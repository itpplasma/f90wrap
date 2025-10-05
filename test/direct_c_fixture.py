"""
Pytest fixture and utilities for testing direct-C code generation.

This module provides automated build infrastructure for testing f90wrap --direct-c
functionality, including compilation and linking of generated C/Fortran code.
"""

import os
import tempfile
import shutil
import subprocess
import json
import pytest
from pathlib import Path
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
        object_files = []

        # Get f90wrap include directory for capsule_helpers.h
        import f90wrap
        f90wrap_include = Path(f90wrap.__file__).parent

        for source in source_files:
            obj_file = self.work_dir / source.with_suffix('.o').name

            cmd = [
                self.cc,
                '-c',
                '-fPIC',
                f'-I{self.python_include}',
                f'-I{self.numpy_include}',
                f'-I{f90wrap_include}',  # Add f90wrap include path
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

        # Find generated files
        generated = {
            'c_wrapper': self.work_dir / f"_{module_name}module.c",
            'fortran_support': self.work_dir / f"{module_name}_support.f90",
            'python_module': self.work_dir / f"{module_name}.py"
        }

        # Verify files were created
        for key, path in generated.items():
            if not path.exists():
                raise DirectCBuildError(f"Expected file not generated: {path}")

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

        # Step 3: Compile Fortran support module
        support_objects = builder.compile_fortran([generated['fortran_support']])

        # Step 4: Compile C wrapper
        c_objects = builder.compile_c([generated['c_wrapper']])

        # Step 5: Link everything into extension module
        all_objects = fortran_objects + support_objects + c_objects
        extension = builder.link_extension(all_objects, module_name)
        results['extension_path'] = extension

        # Step 6: Try to import the module
        import sys
        sys.path.insert(0, str(builder.work_dir))

        # Copy Python wrapper to work directory
        shutil.copy2(generated['python_module'], builder.work_dir)

        # Import and basic test
        module = __import__(module_name)
        results['module'] = module
        results['success'] = True

    except Exception as e:
        results['error'] = str(e)

    return results