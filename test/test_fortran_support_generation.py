"""
Test Fortran support module generation for direct C mode.
"""

import unittest
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestFortranSupportGeneration(unittest.TestCase):
    """Test generation of Fortran helper routines for derived types."""

    def setUp(self):
        """Create test AST with a derived type."""
        self.root = ft.Root()

        # Create a module with a derived type
        self.module = ft.Module('test_module')

        # Create a simple derived type
        self.dtype = ft.Type('my_type')
        self.dtype.mod_name = 'test_module'
        self.dtype.elements = [
            ft.Element('value', type='real(8)', attributes=[]),
            ft.Element('count', type='integer', attributes=[])
        ]
        self.dtype.procedures = []

        self.module.types = [self.dtype]
        self.module.routines = []
        self.root.modules = [self.module]
        self.root.procedures = []

    def test_fortran_support_generated(self):
        """Test that Fortran support module is generated with allocators."""
        gen = CWrapperGenerator(self.root, 'test_module')
        fortran_code = gen.generate_fortran_support()

        # Check module declaration
        self.assertIn('module f90wrap_support', fortran_code)
        self.assertIn('use test_module', fortran_code)
        self.assertIn('use iso_c_binding', fortran_code)

        # Check allocator routine
        self.assertIn('subroutine f90wrap_my_type__allocate(ptr)', fortran_code)
        self.assertIn('type(c_ptr), intent(out) :: ptr', fortran_code)
        self.assertIn('type(my_type), pointer :: fptr', fortran_code)
        self.assertIn('allocate(fptr)', fortran_code)
        self.assertIn('ptr = c_loc(fptr)', fortran_code)

        # Check deallocator routine
        self.assertIn('subroutine f90wrap_my_type__deallocate(ptr)', fortran_code)
        self.assertIn('type(c_ptr), intent(inout) :: ptr', fortran_code)
        self.assertIn('if (c_associated(ptr)) then', fortran_code)
        self.assertIn('call c_f_pointer(ptr, fptr)', fortran_code)
        self.assertIn('deallocate(fptr)', fortran_code)
        self.assertIn('ptr = c_null_ptr', fortran_code)

    def test_multiple_types(self):
        """Test support generation for multiple derived types."""
        # Add another type
        dtype2 = ft.Type('another_type')
        dtype2.mod_name = 'test_module'
        dtype2.elements = [
            ft.Element('data', type='real(8)', attributes=[])
        ]
        dtype2.procedures = []
        self.module.types.append(dtype2)

        gen = CWrapperGenerator(self.root, 'test_module')
        fortran_code = gen.generate_fortran_support()

        # Check both types have allocators
        self.assertIn('f90wrap_my_type__allocate', fortran_code)
        self.assertIn('f90wrap_my_type__deallocate', fortran_code)
        self.assertIn('f90wrap_another_type__allocate', fortran_code)
        self.assertIn('f90wrap_another_type__deallocate', fortran_code)

    def test_multiple_modules(self):
        """Test support generation for types from multiple modules."""
        # Create second module with a type
        module2 = ft.Module('second_module')
        dtype2 = ft.Type('second_type')
        dtype2.mod_name = 'second_module'
        dtype2.elements = [
            ft.Element('field', type='integer', attributes=[])
        ]
        dtype2.procedures = []
        module2.types = [dtype2]
        module2.routines = []

        self.root.modules.append(module2)

        gen = CWrapperGenerator(self.root, 'test_module')
        fortran_code = gen.generate_fortran_support()

        # Check both modules are used
        self.assertIn('use test_module', fortran_code)
        self.assertIn('use second_module', fortran_code)

        # Check types from both modules have support
        self.assertIn('f90wrap_my_type__allocate', fortran_code)
        self.assertIn('f90wrap_second_type__allocate', fortran_code)

    def test_no_types(self):
        """Test that no support module is generated when no types exist."""
        # Clear types
        self.module.types = []

        gen = CWrapperGenerator(self.root, 'test_module')
        fortran_code = gen.generate_fortran_support()

        # Should return empty string
        self.assertEqual(fortran_code, '')

    def test_bind_c_names(self):
        """Test that bind(C) names are properly mangled."""
        gen = CWrapperGenerator(self.root, 'test_module')
        fortran_code = gen.generate_fortran_support()

        # Check for bind(C) with mangled names
        self.assertIn("bind(C, name='", fortran_code)
        # The exact mangled name depends on the name mangler implementation
        # but it should contain the type name
        self.assertIn('my_type', fortran_code)


if __name__ == '__main__':
    unittest.main()