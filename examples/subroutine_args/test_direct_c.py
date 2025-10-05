"""
Test subroutine_args example using direct-C generation.

This test suite builds and validates the subroutine_args example with f90wrap --direct-c,
testing handling of multi-line and commented subroutine arguments.
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


class TestSubroutineArgsDirectC(unittest.TestCase):
    """Test the subroutine_args example using direct-C code generation."""

    @classmethod
    def setUpClass(cls):
        """Set up the direct-C build environment once for all tests."""
        cls.work_dir = Path(tempfile.mkdtemp(prefix='subroutine_args_direct_c_'))
        cls.builder = DirectCBuilder(cls.work_dir, verbose=True)

        # Get source files
        example_dir = Path(__file__).parent
        fortran_sources = [example_dir / 'subroutine_mod.f90']

        # Build with direct-C
        cls.module_name = 'subroutine_mod_direct_c'
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

            cls.mod = __import__(cls.module_name)
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

    def test_routine_with_simple_args(self):
        """Test basic subroutine with simple arguments."""
        c, d = self.mod.subroutine_mod.routine_with_simple_args(2, 3)
        self.assertEqual(c, 5)
        self.assertEqual(d, 6)

    def test_routine_with_multiline_args(self):
        """Test subroutine with multi-line argument declaration."""
        c, d = self.mod.subroutine_mod.routine_with_multiline_args(2, 3)
        self.assertEqual(c, 5)
        self.assertEqual(d, 6)

    def test_routine_with_commented_args(self):
        """Test subroutine with comments in argument list."""
        c, d = self.mod.subroutine_mod.routine_with_commented_args(2, 3)
        self.assertEqual(c, 5)
        self.assertEqual(d, 6)

    def test_routine_with_more_commented_args(self):
        """Test subroutine with multiple comments in argument list."""
        c, d = self.mod.subroutine_mod.routine_with_more_commented_args(2, 3)
        self.assertEqual(c, 5)
        self.assertEqual(d, 6)


if __name__ == '__main__':
    unittest.main()