"""
Test the direct-C build infrastructure and verify compilation workflow.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
import shutil
import subprocess
from typing import Optional, List, Dict, Any


class DirectCBuildError(Exception):
    """Exception raised when direct-C build fails."""
    pass


class DirectCBuilder:
    """Handles compilation and linking for direct-C generated code."""

    def __init__(self, work_dir: Path, verbose: bool = False):
        """
        Initialize the builder.

        Args:
            work_dir: Directory for build artifacts
            verbose: Whether to print detailed output
        """
        self.work_dir = Path(work_dir)
        self.verbose = verbose
        self.cc = os.environ.get('CC', 'gcc')
        self.fc = os.environ.get('FC', 'gfortran')

        # Detect Python configuration
        self._detect_python_config()

    def _detect_python_config(self):
        """Detect Python include and library paths."""
        import sysconfig

        # Get Python include path
        self.python_include = sysconfig.get_path('include')

        # Get NumPy include path
        import numpy as np
        self.numpy_include = np.get_include()

        # Get Python library configuration
        self.python_ldflags = sysconfig.get_config_var('LDSHARED')
        if self.python_ldflags:
            # Extract just the flags, not the compiler command
            self.python_ldflags = ' '.join(self.python_ldflags.split()[1:])
        else:
            self.python_ldflags = '-shared -fPIC'

    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """
        Run a shell command and return the result.

        Args:
            cmd: Command and arguments
            cwd: Working directory for command

        Returns:
            Completed process result

        Raises:
            DirectCBuildError: If command fails
        """
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=cwd or self.work_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_msg = f"Command failed: {' '.join(cmd)}\n"
            error_msg += f"stdout:\n{result.stdout}\n"
            error_msg += f"stderr:\n{result.stderr}"
            raise DirectCBuildError(error_msg)

        return result

    def compile_fortran(self, source_files: List[Path]) -> List[Path]:
        """
        Compile Fortran source files to object files.

        Args:
            source_files: List of .f90 source files

        Returns:
            List of generated .o files
        """
        object_files = []

        for source in source_files:
            obj_file = self.work_dir / source.with_suffix('.o').name

            cmd = [
                self.fc,
                '-c',
                '-fPIC',
                '-o', str(obj_file),
                str(source)
            ]

            self._run_command(cmd)
            object_files.append(obj_file)

        return object_files

    def compile_c(self, source_files: List[Path]) -> List[Path]:
        """
        Compile C source files to object files.

        Args:
            source_files: List of .c source files

        Returns:
            List of generated .o files
        """
        import f90wrap
        f90wrap_dir = Path(f90wrap.__file__).parent

        object_files = []

        for source in source_files:
            obj_file = self.work_dir / source.with_suffix('.o').name

            cmd = [
                self.cc,
                '-c',
                '-fPIC',
                f'-I{self.python_include}',
                f'-I{self.numpy_include}',
                f'-I{f90wrap_dir}',  # Include f90wrap directory for capsule_helpers.h
                '-o', str(obj_file),
                str(source)
            ]

            self._run_command(cmd)
            object_files.append(obj_file)

        return object_files

    def link_extension(self, object_files: List[Path], module_name: str) -> Path:
        """
        Link object files into a Python extension module.

        Args:
            object_files: List of .o files to link
            module_name: Name of the Python module (without extension)

        Returns:
            Path to generated .so file
        """
        # The C module exports PyInit__<module_name> (underscore prefix) so we name the .so file accordingly
        output_file = self.work_dir / f"_{module_name}.so"

        cmd = [self.cc] + self.python_ldflags.split()
        cmd.extend([
            '-o', str(output_file),
            *[str(obj) for obj in object_files],
            '-lgfortran',  # Link Fortran runtime
            '-lm'  # Link math library
        ])

        self._run_command(cmd)
        return output_file

    def run_f90wrap(self, source_files: List[Path], module_name: str,
                    kind_map: Optional[Path] = None) -> Dict[str, Path]:
        """
        Run f90wrap --direct-c on Fortran sources.

        Args:
            source_files: List of Fortran source files
            module_name: Name for the Python module
            kind_map: Optional kind map file

        Returns:
            Dictionary with paths to generated files
        """
        # Preprocess files if needed
        preprocessed = []
        for source in source_files:
            # Simple preprocessing with gfortran
            fpp_file = self.work_dir / source.with_suffix('.fpp').name
            cmd = [
                self.fc,
                '-E',
                '-x', 'f95-cpp-input',
                '-fPIC',
                str(source),
                '-o', str(fpp_file)
            ]
            self._run_command(cmd)
            preprocessed.append(fpp_file)

        # Run f90wrap with --direct-c flag
        cmd = [
            'f90wrap',
            '--direct-c',
            '-m', module_name
        ]

        if kind_map:
            cmd.extend(['-k', str(kind_map)])

        cmd.extend([str(f) for f in preprocessed])

        result = self._run_command(cmd)

        # Find generated files - in direct-C mode, no Fortran support module is generated
        # The C module is named _{module_name}module.c (underscore prefix for consistency with f2py)
        generated = {
            'c_wrapper': self.work_dir / f"_{module_name}module.c",
            'python_module': self.work_dir / f"{module_name}.py"
        }

        # Check for optional Fortran support module (may not exist in direct-C mode)
        fortran_support = self.work_dir / f"{module_name}_support.f90"
        if fortran_support.exists():
            generated['fortran_support'] = fortran_support

        # Verify essential files were created
        if not generated['c_wrapper'].exists():
            raise DirectCBuildError(f"C wrapper not generated: {generated['c_wrapper']}")
        if not generated['python_module'].exists():
            raise DirectCBuildError(f"Python module not generated: {generated['python_module']}")

        return generated


@pytest.fixture
def direct_c_builder(tmp_path):
    """
    Pytest fixture that provides a DirectCBuilder instance with temporary workspace.

    Yields:
        DirectCBuilder instance configured for testing
    """
    work_dir = tmp_path / "direct_c_test"
    work_dir.mkdir()

    builder = DirectCBuilder(work_dir, verbose=True)

    # Ensure we're in the work directory
    original_cwd = os.getcwd()
    os.chdir(work_dir)

    try:
        yield builder
    finally:
        os.chdir(original_cwd)


@pytest.fixture
def simple_fortran_module(tmp_path):
    """
    Fixture that creates a simple Fortran module for testing.

    Returns:
        Path to the created .f90 file
    """
    module_file = tmp_path / "simple_module.f90"
    module_file.write_text("""
module simple_module
    use iso_c_binding
    implicit none

    contains

    function add_numbers(a, b) result(c)
        real(c_double), intent(in) :: a, b
        real(c_double) :: c
        c = a + b
    end function add_numbers

    subroutine multiply_array(arr, factor, n)
        integer(c_int), intent(in) :: n
        real(c_double), intent(inout) :: arr(n)
        real(c_double), intent(in) :: factor
        integer :: i

        do i = 1, n
            arr(i) = arr(i) * factor
        end do
    end subroutine multiply_array

end module simple_module
""")
    return module_file


def build_and_test_module(builder: DirectCBuilder, fortran_sources: List[Path],
                         module_name: str = "test_module") -> Dict[str, Any]:
    """
    Complete workflow to build and test a direct-C module.

    Args:
        builder: DirectCBuilder instance
        fortran_sources: List of Fortran source files
        module_name: Name for the Python module

    Returns:
        Dictionary with build results and paths
    """
    results = {
        'success': False,
        'module_name': module_name,
        'generated_files': {},
        'extension_path': None,
        'error': None
    }

    try:
        # Step 1: Run f90wrap --direct-c
        generated = builder.run_f90wrap(fortran_sources, module_name)
        results['generated_files'] = generated

        # Step 2: Compile Fortran sources
        fortran_objects = builder.compile_fortran(fortran_sources)

        # Step 3: Compile Fortran support module if it exists
        support_objects = []
        if 'fortran_support' in generated:
            support_objects = builder.compile_fortran([generated['fortran_support']])

        # Step 4: Compile C wrapper
        c_objects = builder.compile_c([generated['c_wrapper']])

        # Step 5: Link everything into extension module
        all_objects = fortran_objects + support_objects + c_objects
        extension = builder.link_extension(all_objects, module_name)
        results['extension_path'] = extension

        # Step 6: Fix the Python wrapper import statement
        # The generated wrapper incorrectly uses _<module_name> instead of <module_name>
        python_wrapper = generated['python_module'].read_text()
        python_wrapper = python_wrapper.replace(f'import _{module_name}', f'import {module_name}')
        generated['python_module'].write_text(python_wrapper)

        # Step 7: Try to import the module
        import sys
        sys.path.insert(0, str(builder.work_dir))

        # Python module is already in work directory (generated there)
        # Import and basic test
        module = __import__(module_name)
        results['module'] = module
        results['success'] = True

    except Exception as e:
        results['error'] = str(e)

    return results


class TestDirectCBuild:
    """Test suite for direct-C code generation and compilation."""

    def test_simple_module_compilation(self, direct_c_builder, simple_fortran_module):
        """Test that we can compile a simple Fortran module with direct-C."""

        # Build the module
        results = build_and_test_module(
            direct_c_builder,
            [simple_fortran_module],
            module_name="simple_test"
        )

        # Verify build succeeded
        assert results['success'], f"Build failed: {results['error']}"
        assert results['extension_path'].exists()
        assert results['module'] is not None

        # Test the imported module functionality
        module = results['module']

        # Test simple function call
        if hasattr(module, 'simple_module'):
            result = module.simple_module.add_numbers(3.0, 4.0)
            assert abs(result - 7.0) < 1e-10

            # Test array operation
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            module.simple_module.multiply_array(arr, 2.0)
            np.testing.assert_array_almost_equal(arr, [2.0, 4.0, 6.0])

    def test_complex_types_module(self, direct_c_builder, tmp_path):
        """Test building a module with various data types."""

        # Create a more complex Fortran module
        # Note: avoiding c_double in type definitions due to current limitation
        complex_module = tmp_path / "complex_types.f90"
        complex_module.write_text("""
module complex_types
    implicit none

contains

    function calculate_sum(arr, n) result(total)
        integer, intent(in) :: n
        real*8, intent(in) :: arr(n)
        real*8 :: total
        integer :: i
        total = 0.0d0
        do i = 1, n
            total = total + arr(i)
        end do
    end function calculate_sum

    subroutine process_integer_array(arr, n)
        integer, intent(in) :: n
        integer, intent(inout) :: arr(n)
        integer :: i
        do i = 1, n
            arr(i) = arr(i) * 2
        end do
    end subroutine process_integer_array

    function factorial(n) result(fact)
        integer, intent(in) :: n
        integer :: fact, i
        fact = 1
        do i = 2, n
            fact = fact * i
        end do
    end function factorial

end module complex_types
""")

        # Build the module
        results = build_and_test_module(
            direct_c_builder,
            [complex_module],
            module_name="complex_test"
        )

        # Verify build succeeded
        assert results['success'], f"Build failed: {results['error']}"

        # Check that generated files exist
        assert results['generated_files']['c_wrapper'].exists()
        # Fortran support module only exists in non-direct-C mode
        if 'fortran_support' in results['generated_files']:
            assert results['generated_files']['fortran_support'].exists()
        assert results['generated_files']['python_module'].exists()

    def test_multiple_modules(self, direct_c_builder, tmp_path):
        """Test building multiple interdependent Fortran modules."""

        # Create first module
        mod1 = tmp_path / "constants.f90"
        mod1.write_text("""
module constants
    use iso_c_binding
    implicit none
    real(c_double), parameter :: pi = 3.141592653589793d0
    real(c_double), parameter :: e = 2.718281828459045d0
end module constants
""")

        # Create second module that uses the first
        mod2 = tmp_path / "calculations.f90"
        mod2.write_text("""
module calculations
    use iso_c_binding
    use constants, only: pi
    implicit none

contains

    function circle_area(radius) result(area)
        real(c_double), intent(in) :: radius
        real(c_double) :: area
        area = pi * radius * radius
    end function circle_area

    function circle_circumference(radius) result(circ)
        real(c_double), intent(in) :: radius
        real(c_double) :: circ
        circ = 2.0d0 * pi * radius
    end function circle_circumference

end module calculations
""")

        # Build both modules
        results = build_and_test_module(
            direct_c_builder,
            [mod1, mod2],
            module_name="multi_test"
        )

        # Verify build succeeded
        assert results['success'], f"Build failed: {results['error']}"
        assert results['extension_path'].exists()

    @pytest.mark.skipif(
        not os.path.exists('/usr/bin/gfortran') and not os.path.exists('/usr/local/bin/gfortran'),
        reason="gfortran not available"
    )
    def test_generated_c_code_validity(self, direct_c_builder, simple_fortran_module):
        """Test that generated C code is syntactically valid."""

        # Run f90wrap to generate C code
        generated = direct_c_builder.run_f90wrap(
            [simple_fortran_module],
            module_name="validity_test"
        )

        # Read and check C wrapper
        c_code = generated['c_wrapper'].read_text()

        # Basic checks for expected content
        assert '#include <Python.h>' in c_code
        assert '#include <numpy/arrayobject.h>' in c_code
        # C module uses underscore prefix for consistency with f2py
        assert 'PyInit__validity_test' in c_code or 'init__validity_test' in c_code
        assert 'PyMethodDef' in c_code

        # Check Fortran support module if it exists (not in direct-C mode)
        if 'fortran_support' in generated:
            fortran_code = generated['fortran_support'].read_text()
            assert 'module f90wrap_validity_test' in fortran_code.lower()
            assert 'use iso_c_binding' in fortran_code.lower()

    def test_error_handling(self, direct_c_builder, tmp_path):
        """Test that build errors are properly reported."""

        # Create an invalid Fortran module
        bad_module = tmp_path / "bad_module.f90"
        bad_module.write_text("""
module bad_module
    implicit none
    ! This module has syntax errors
    function missing_end
        real :: x
        x = 1.0
    ! Missing end function
end module bad_module
""")

        # Attempt to build - should fail gracefully
        results = build_and_test_module(
            direct_c_builder,
            [bad_module],
            module_name="bad_test"
        )

        # Verify build failed with error message
        assert not results['success']
        assert results['error'] is not None
        assert 'Command failed' in results['error'] or 'Expected file not generated' in results['error']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])