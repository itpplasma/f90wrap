"""
Test arrays example using direct-C generation.

This parallel test suite builds the arrays example using f90wrap --direct-c
and verifies that all functionality works correctly with the new backend.
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import subprocess

# Add test directory to path to import the direct_c_fixture
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'test'))
from direct_c_fixture import DirectCBuilder


class TestArraysDirectC(unittest.TestCase):
    """Test the arrays example using direct-C code generation."""

    @classmethod
    def setUpClass(cls):
        """Set up the direct-C build environment once for all tests."""
        cls.work_dir = Path(tempfile.mkdtemp(prefix='arrays_direct_c_'))
        cls.builder = DirectCBuilder(cls.work_dir, verbose=True)

        # Get source files
        example_dir = Path(__file__).parent
        fortran_sources = [
            example_dir / 'parameters.f90',
            example_dir / 'library.f90'
        ]

        # Build with direct-C
        cls.module_name = 'arrays_direct_c'
        try:
            # Run f90wrap with --direct-c
            cmd = [
                'f90wrap',
                '--direct-c',
                '-m', cls.module_name,
                str(fortran_sources[0]),
                str(fortran_sources[1])
            ]

            result = subprocess.run(
                cmd,
                cwd=cls.work_dir,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"f90wrap failed: {result.stderr}")

            # Compile Fortran sources
            fortran_objects = cls.builder.compile_fortran(fortran_sources)

            # With --direct-c, we get a C module with underscore prefix (for f2py consistency)
            c_module = cls.work_dir / f"_{cls.module_name}module.c"
            if not c_module.exists():
                raise RuntimeError(f"C module not found: {c_module}")
            c_objects = cls.builder.compile_c([c_module])

            # Link into extension module - the C module exports with underscore prefix
            all_objects = fortran_objects + c_objects

            # The .so filename must match the exported module name (underscore prefix)
            cmd = [cls.builder.cc] + cls.builder.python_ldflags.split()
            cmd.extend([
                '-o', str(cls.work_dir / f"_{cls.module_name}.so"),
                *[str(obj) for obj in all_objects],
                '-lgfortran',
                '-lm'
            ])
            cls.builder._run_command(cmd)

            # Set up Python path and import module
            sys.path.insert(0, str(cls.work_dir))

            # Import the Python wrapper - fix the import in the wrapper file
            py_wrapper = cls.work_dir / f"{cls.module_name}.py"
            if not py_wrapper.exists():
                raise RuntimeError(f"Python wrapper not found: {py_wrapper}")

            # Fix the import statement in the wrapper to match the actual module name
            wrapper_content = py_wrapper.read_text()
            wrapper_content = wrapper_content.replace(f'import _{cls.module_name}', f'import {cls.module_name}')
            wrapper_content = wrapper_content.replace(f'_{cls.module_name}.', f'{cls.module_name}.')
            py_wrapper.write_text(wrapper_content)

            # Import the wrapper module
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"{cls.module_name}_wrapper", py_wrapper)
            cls.lib = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.lib)

            cls.build_success = True

        except Exception as e:
            cls.build_error = str(e)
            cls.build_success = False
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory."""
        if hasattr(cls, 'work_dir') and cls.work_dir.exists():
            shutil.rmtree(cls.work_dir)

    def setUp(self):
        """Check that build was successful before each test."""
        if not self.build_success:
            self.skipTest(f"Build failed: {self.build_error}")

    def do_array_stuff(self, ndata):
        """Helper method to test array operations."""
        x = np.arange(ndata, dtype=np.float64)
        y = np.arange(ndata, dtype=np.float64)
        br = np.zeros((ndata,), order='F', dtype=np.float64)
        co = np.zeros((4, ndata), order='F', dtype=np.float64)

        self.lib.library.do_array_stuff(n=ndata, x=x, y=y, br=br, co=co)

        for k in range(4):
            np.testing.assert_allclose(x*y + x, co[k,:])
        np.testing.assert_allclose(x/(y+1.0), br)

    def test_basic(self):
        """Test basic array operations with moderate size."""
        self.do_array_stuff(1000)

    def test_verybig_array(self):
        """Test array operations with large arrays."""
        self.do_array_stuff(100000)

    def test_square(self):
        """Test squaring operation on arrays."""
        n = 10000
        x = np.arange(n, dtype=np.float64)
        y = np.arange(n, dtype=np.float64)
        br = np.zeros((n,), order='F', dtype=np.float64)
        co = np.zeros((4, n), order='F', dtype=np.float64)

        self.lib.library.do_array_stuff(n=n, x=x, y=y, br=br, co=co)
        self.lib.library.only_manipulate(n=n, array=co)
        for k in range(4):
            np.testing.assert_allclose((x*y + x)**2, co[k,:])

    def test_return_array(self):
        """Test returning arrays from Fortran."""
        m, n = 10, 4
        arr = np.ndarray((m,n), order='F', dtype=np.int32)
        self.lib.library.return_array(m, n, arr)
        ii, jj = np.mgrid[0:m,0:n]
        ii += 1
        jj += 1
        np.testing.assert_equal(ii*jj + jj, arr)

    def test_set_value(self):
        """Test setting and getting module variables."""
        self.lib.library.ia = 1
        ia = self.lib.library.ia
        np.testing.assert_equal(ia, 1)

    def test_set_array(self):
        """Test setting and getting module arrays."""
        iarray_ref = np.arange(0, 3, dtype=np.int32)
        self.lib.library.iarray = iarray_ref
        iarray = self.lib.library.iarray
        np.testing.assert_allclose(iarray, iarray_ref)


if __name__ == '__main__':
    unittest.main()