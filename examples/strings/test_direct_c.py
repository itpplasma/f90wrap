"""
Test strings example using direct-C generation.

This test suite builds and validates the strings example with f90wrap --direct-c,
testing string handling between Python and Fortran.
"""

import sys
import os
import unittest
from pathlib import Path
import tempfile
import shutil
import subprocess

# Add test directory to path to import the direct_c_fixture
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'test'))
from direct_c_fixture import DirectCBuilder


class TestStringsDirectC(unittest.TestCase):
    """Test the strings example using direct-C code generation."""

    @classmethod
    def setUpClass(cls):
        """Set up the direct-C build environment once for all tests."""
        cls.work_dir = Path(tempfile.mkdtemp(prefix='strings_direct_c_'))
        cls.builder = DirectCBuilder(cls.work_dir, verbose=True)

        # Get source files
        example_dir = Path(__file__).parent
        fortran_sources = [example_dir / 'string_io.f90']

        # Build with direct-C
        cls.module_name = 'strings_direct_c'
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

    def test_func_generate_string(self):
        """Test function that generates a string."""
        n = 52
        outstring = self.lib.string_io.func_generate_string(n)
        expected = b''.join([chr(k).encode('latin-1') for k in range(34, n+34)])
        self.assertEqual(outstring, expected)

    def test_func_return_string(self):
        """Test function that returns a fixed string."""
        expected = b'-_-::this is a string with ASCII, / and 123...::-_-'
        result = self.lib.string_io.func_return_string()
        self.assertEqual(result.strip(), expected)

    def test_sub_generate_string(self):
        """Test subroutine that generates a string."""
        n = 52
        outstring = self.lib.string_io.generate_string(n)
        expected = b''.join([chr(k).encode('latin-1') for k in range(34, n+34)])
        self.assertEqual(outstring, expected)

    def test_sub_return_string(self):
        """Test subroutine that returns a fixed string."""
        expected = b'-_-::this is a string with ASCII, / and 123...::-_-'
        result = self.lib.string_io.return_string()
        self.assertEqual(result.strip(), expected)

    def test_global_string(self):
        """Test setting and getting global string."""
        test_string = b'Test global string'
        self.lib.string_io.set_global_string(len(test_string), test_string)
        result = self.lib.string_io.global_string
        # The global string is 512 characters, so we need to compare just the relevant part
        self.assertTrue(result.startswith(test_string))

    def test_inout_string(self):
        """Test inout string parameter."""
        input_string = b'Hello World!'
        result = self.lib.string_io.inout_string(len(input_string), input_string)
        expected = b'Z' * len(input_string)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()