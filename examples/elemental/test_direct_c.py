"""
Test elemental example using direct-C generation.

This test suite builds and validates the elemental example with f90wrap --direct-c,
testing elemental function handling.
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


class TestElementalDirectC(unittest.TestCase):
    """Test the elemental example using direct-C code generation."""

    @classmethod
    def setUpClass(cls):
        """Set up the direct-C build environment once for all tests."""
        cls.work_dir = Path(tempfile.mkdtemp(prefix='elemental_direct_c_'))
        cls.builder = DirectCBuilder(cls.work_dir, verbose=True)

        # Get source files
        example_dir = Path(__file__).parent
        fortran_sources = [example_dir / 'elemental_module.f90']

        # Build with direct-C
        cls.module_name = 'elemental_direct_c'
        try:
            # Run f90wrap with --direct-c
            cmd = [
                'f90wrap',
                '--direct-c',
                '-m', cls.module_name,
                str(fortran_sources[0])
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

            # Compile generated Fortran support module
            support_file = cls.work_dir / f"f90wrap_{cls.module_name}.f90"
            support_objects = cls.builder.compile_fortran([support_file])

            # Compile C wrapper
            c_wrapper = cls.work_dir / f"{cls.module_name}_c.c"
            c_objects = cls.builder.compile_c([c_wrapper])

            # Link into extension module
            all_objects = fortran_objects + support_objects + c_objects
            extension = cls.builder.link_extension(all_objects, cls.module_name)

            # Set up Python path and import module
            sys.path.insert(0, str(cls.work_dir))

            # Copy Python wrapper
            py_wrapper = cls.work_dir / f"{cls.module_name}.py"
            if not py_wrapper.exists():
                raise RuntimeError(f"Python wrapper not found: {py_wrapper}")

            cls.lib = __import__(cls.module_name)
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

    def test_sinc_scalar(self):
        """Test elemental sinc function with scalar input."""
        # Test at x=0 (should return 1)
        result = self.lib.elemental_module.sinc(0.0)
        np.testing.assert_almost_equal(result, 1.0)

        # Test at x=pi (should return sin(pi)/pi ≈ 0)
        result = self.lib.elemental_module.sinc(np.pi)
        expected = np.sin(np.pi) / np.pi
        np.testing.assert_almost_equal(result, expected, decimal=10)

        # Test at x=pi/2 (should return sin(pi/2)/(pi/2) = 2/pi)
        result = self.lib.elemental_module.sinc(np.pi/2)
        expected = 2.0 / np.pi
        np.testing.assert_almost_equal(result, expected)

    def test_sinc_array(self):
        """Test elemental sinc function with array input."""
        # Create test array
        x = np.array([0.0, 1e-6, np.pi/2, np.pi, 2*np.pi])

        # Call sinc on array
        result = self.lib.elemental_module.sinc(x)

        # Calculate expected values
        expected = np.zeros_like(x)
        for i, val in enumerate(x):
            if abs(val) > 1e-5:
                expected[i] = np.sin(val) / val
            else:
                expected[i] = 1.0

        np.testing.assert_array_almost_equal(result, expected)

    def test_sinc_2d_array(self):
        """Test elemental sinc function with 2D array input."""
        # Create 2D test array
        x = np.array([[0.0, np.pi/2], [np.pi, 2*np.pi]])

        # Call sinc on 2D array
        result = self.lib.elemental_module.sinc(x)

        # Calculate expected values
        expected = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                val = x[i, j]
                if abs(val) > 1e-5:
                    expected[i, j] = np.sin(val) / val
                else:
                    expected[i, j] = 1.0

        np.testing.assert_array_almost_equal(result, expected)


if __name__ == '__main__':
    unittest.main()