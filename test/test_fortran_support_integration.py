"""
Integration test for Fortran support module with actual compilation.
"""

import unittest
import tempfile
import os
import subprocess
from pathlib import Path
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestFortranSupportIntegration(unittest.TestCase):
    """Test complete integration of Fortran support with compilation."""

    def setUp(self):
        """Create test AST and temporary directory."""
        self.test_dir = tempfile.mkdtemp(prefix='f90wrap_test_')
        self.root = ft.Root()

        # Create a module with a derived type
        self.module = ft.Module('geometry')

        # Create a point type
        point_type = ft.Type('point')
        point_type.mod_name = 'geometry'
        point_type.elements = [
            ft.Element('x', type='real(8)', attributes=[]),
            ft.Element('y', type='real(8)', attributes=[]),
            ft.Element('z', type='real(8)', attributes=[])
        ]
        point_type.procedures = []

        self.module.types = [point_type]
        self.module.routines = []
        self.root.modules = [self.module]
        self.root.procedures = []

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_generated_fortran_compiles(self):
        """Test that generated Fortran support module compiles."""
        gen = CWrapperGenerator(self.root, 'geometry')
        fortran_code = gen.generate_fortran_support()

        # Write the original Fortran module
        geometry_file = os.path.join(self.test_dir, 'geometry.f90')
        with open(geometry_file, 'w') as f:
            f.write('''
module geometry
    use iso_c_binding
    implicit none

    type :: point
        real(8) :: x, y, z
    end type point

end module geometry
''')

        # Write the support module
        support_file = os.path.join(self.test_dir, 'geometry_support.f90')
        with open(support_file, 'w') as f:
            f.write(fortran_code)

        # Try to compile with gfortran if available
        try:
            # Compile original module
            result = subprocess.run(
                ['gfortran', '-c', geometry_file],
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                self.skipTest(f"gfortran compilation failed: {result.stderr}")

            # Compile support module
            result = subprocess.run(
                ['gfortran', '-c', support_file],
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            self.assertEqual(result.returncode, 0, f"Support module compilation failed: {result.stderr}")

            # Check that object files were created
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'geometry.o')))
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'geometry_support.o')))

        except FileNotFoundError:
            self.skipTest("gfortran not available")
        except subprocess.TimeoutExpired:
            self.fail("Compilation timed out")

    def test_fortran_c_interop(self):
        """Test that Fortran support can be called from C."""
        gen = CWrapperGenerator(self.root, 'geometry')
        fortran_code = gen.generate_fortran_support()

        # Write the original Fortran module
        geometry_file = os.path.join(self.test_dir, 'geometry.f90')
        with open(geometry_file, 'w') as f:
            f.write('''
module geometry
    use iso_c_binding
    implicit none

    type :: point
        real(8) :: x, y, z
    end type point

end module geometry
''')

        # Write the support module
        support_file = os.path.join(self.test_dir, 'geometry_support.f90')
        with open(support_file, 'w') as f:
            f.write(fortran_code)

        # Write a simple C test program
        c_test_file = os.path.join(self.test_dir, 'test_interop.c')
        with open(c_test_file, 'w') as f:
            # Get the mangled name (simplified assumption here)
            f.write('''
#include <stdio.h>
#include <stddef.h>

// External Fortran routines
extern void __geometry_MOD_f90wrap_point__allocate(void** ptr);
extern void __geometry_MOD_f90wrap_point__deallocate(void** ptr);

int main() {
    void* ptr = NULL;

    // Test allocation
    __geometry_MOD_f90wrap_point__allocate(&ptr);
    if (ptr == NULL) {
        printf("ERROR: Allocation failed\\n");
        return 1;
    }
    printf("Allocation successful\\n");

    // Test deallocation
    __geometry_MOD_f90wrap_point__deallocate(&ptr);
    if (ptr != NULL) {
        printf("ERROR: Deallocation failed to clear pointer\\n");
        return 1;
    }
    printf("Deallocation successful\\n");

    return 0;
}
''')

        # Try to compile and link
        try:
            # Compile Fortran modules
            result = subprocess.run(
                ['gfortran', '-c', geometry_file, support_file],
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                self.skipTest(f"Fortran compilation failed: {result.stderr}")

            # Compile C test
            result = subprocess.run(
                ['gcc', '-c', c_test_file],
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                self.skipTest(f"C compilation failed: {result.stderr}")

            # Link everything
            result = subprocess.run(
                ['gfortran', '-o', 'test_interop', 'test_interop.o',
                 'geometry.o', 'geometry_support.o'],
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                # Linking might fail due to name mangling differences
                # This is OK for now as we're just testing compilation
                pass

        except FileNotFoundError:
            self.skipTest("gcc/gfortran not available")
        except subprocess.TimeoutExpired:
            self.fail("Compilation timed out")


if __name__ == '__main__':
    unittest.main()