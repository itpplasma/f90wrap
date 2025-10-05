"""
End-to-end test: Generate wrappers for a module with derived types and verify Fortran support.
"""

import unittest
import tempfile
import os
import subprocess
from pathlib import Path


class TestEndToEndFortranSupport(unittest.TestCase):
    """Full end-to-end test with f90wrap CLI."""

    def setUp(self):
        """Create test directory and sample Fortran code."""
        self.test_dir = tempfile.mkdtemp(prefix='f90wrap_e2e_')

        # Create a sample Fortran module with derived types
        self.fortran_file = os.path.join(self.test_dir, 'shapes.f90')
        with open(self.fortran_file, 'w') as f:
            f.write('''
module shapes
    use iso_c_binding
    implicit none

    type :: rectangle
        real(8) :: width
        real(8) :: height
    end type rectangle

    type :: circle
        real(8) :: radius
        real(8) :: center_x
        real(8) :: center_y
    end type circle

contains

    function create_rectangle(w, h) result(rect)
        real(8), intent(in) :: w, h
        type(rectangle) :: rect
        rect%width = w
        rect%height = h
    end function create_rectangle

    function rectangle_area(rect) result(area)
        type(rectangle), intent(in) :: rect
        real(8) :: area
        area = rect%width * rect%height
    end function rectangle_area

    subroutine scale_rectangle(rect, factor)
        type(rectangle), intent(inout) :: rect
        real(8), intent(in) :: factor
        rect%width = rect%width * factor
        rect%height = rect%height * factor
    end subroutine scale_rectangle

end module shapes
''')

    def tearDown(self):
        """Clean up test directory."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_direct_c_mode_generates_fortran_support(self):
        """Test that f90wrap --direct-c generates both C and Fortran support."""
        # Run f90wrap in direct-c mode
        result = subprocess.run(
            ['python', '-m', 'f90wrap', '--direct-c', self.fortran_file],
            cwd=self.test_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)

        self.assertEqual(result.returncode, 0, f"f90wrap failed: {result.stderr}")

        # Check that C module was generated (default name is 'modmodule.c')
        c_module = os.path.join(self.test_dir, 'modmodule.c')
        self.assertTrue(os.path.exists(c_module), "C module not generated")

        # Check that Fortran support was generated (default name is 'mod_support.f90')
        fortran_support = os.path.join(self.test_dir, 'mod_support.f90')
        self.assertTrue(os.path.exists(fortran_support), "Fortran support module not generated")

        # Read and verify Fortran support content
        with open(fortran_support, 'r') as f:
            support_content = f.read()

        # Check for allocator/deallocator routines
        self.assertIn('f90wrap_rectangle__allocate', support_content)
        self.assertIn('f90wrap_rectangle__deallocate', support_content)
        self.assertIn('f90wrap_circle__allocate', support_content)
        self.assertIn('f90wrap_circle__deallocate', support_content)

        # Check for proper Fortran structure
        self.assertIn('module f90wrap_support', support_content)
        self.assertIn('use shapes', support_content)
        self.assertIn('use iso_c_binding', support_content)
        self.assertIn('type(rectangle), pointer :: fptr', support_content)
        self.assertIn('type(circle), pointer :: fptr', support_content)

        # Verify bind(C) interfaces
        self.assertIn('bind(C, name=', support_content)

        # Check that compilation would work (if gfortran available)
        try:
            # Try to compile the original module
            result = subprocess.run(
                ['gfortran', '-c', self.fortran_file],
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Try to compile the support module
                result = subprocess.run(
                    ['gfortran', '-c', fortran_support],
                    cwd=self.test_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                self.assertEqual(result.returncode, 0,
                               f"Support module compilation failed: {result.stderr}")
        except FileNotFoundError:
            # gfortran not available, skip compilation test
            pass

    def test_regular_mode_no_fortran_support(self):
        """Test that regular mode (non direct-c) doesn't generate Fortran support."""
        # Run f90wrap in regular mode (without --direct-c)
        result = subprocess.run(
            ['python', '-m', 'f90wrap', self.fortran_file],
            cwd=self.test_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        self.assertEqual(result.returncode, 0, f"f90wrap failed: {result.stderr}")

        # Check that Fortran support was NOT generated
        fortran_support = os.path.join(self.test_dir, 'mod_support.f90')
        self.assertFalse(os.path.exists(fortran_support),
                        "Fortran support should not be generated in regular mode")

        # Check that f90wrap files were generated instead
        f90wrap_file = os.path.join(self.test_dir, 'f90wrap_shapes.f90')
        self.assertTrue(os.path.exists(f90wrap_file),
                       "f90wrap Fortran file not generated in regular mode")


if __name__ == '__main__':
    unittest.main()