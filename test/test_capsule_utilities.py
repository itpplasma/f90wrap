"""
Test that shared capsule utilities reduce code duplication across generated modules.
"""

import unittest
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestCapsuleUtilities(unittest.TestCase):
    """Test that capsule utilities reduce code duplication."""

    def setUp(self):
        """Create test AST with multiple modules and derived types."""
        self.root = ft.Root()

        # Create first module with two derived types
        module1 = ft.Module('module_a')

        type_a = ft.Type('type_a')
        type_a.mod_name = 'module_a'
        type_a.elements = [
            ft.Element('x', type='real(8)', attributes=[])
        ]

        type_b = ft.Type('type_b')
        type_b.mod_name = 'module_a'
        type_b.elements = [
            ft.Element('y', type='integer', attributes=[])
        ]

        module1.types = [type_a, type_b]

        # Add functions that use these types
        func1 = ft.Function('create_type_a', filename='test.f90')
        func1.arguments = []
        func1.ret_val = ft.Argument(name='result', type='type(type_a)')
        func1.mod_name = 'module_a'

        func2 = ft.Function('create_type_b', filename='test.f90')
        func2.arguments = []
        func2.ret_val = ft.Argument(name='result', type='type(type_b)')
        func2.mod_name = 'module_a'

        sub1 = ft.Subroutine('process_type_a', filename='test.f90')
        arg = ft.Argument(name='obj', type='type(type_a)')
        arg.intent = 'in'
        arg.attributes = ['intent(in)']
        sub1.arguments = [arg]
        sub1.mod_name = 'module_a'

        module1.routines = [func1, func2, sub1]

        # Create second module with another derived type
        module2 = ft.Module('module_b')

        type_c = ft.Type('type_c')
        type_c.mod_name = 'module_b'
        type_c.elements = [
            ft.Element('z', type='real(8)', attributes=[])
        ]

        module2.types = [type_c]

        func3 = ft.Function('create_type_c', filename='test.f90')
        func3.arguments = []
        func3.ret_val = ft.Argument(name='result', type='type(type_c)')
        func3.mod_name = 'module_b'

        module2.routines = [func3]

        self.root.modules = [module1, module2]

    def test_shared_capsule_helpers_included(self):
        """Test that generated code includes shared capsule helpers header."""
        gen = CWrapperGenerator(self.root, 'module_a')
        code = gen.generate()

        # Check header inclusion
        self.assertIn('#include "capsule_helpers.h"', code)

    def test_uses_shared_create_capsule(self):
        """Test that generated code uses shared f90wrap_create_capsule function."""
        gen = CWrapperGenerator(self.root, 'module_a')
        code = gen.generate()

        # Should use f90wrap_create_capsule instead of direct PyCapsule_New
        self.assertIn('f90wrap_create_capsule', code)
        # Direct PyCapsule_New should not appear (except maybe in comments)
        lines = [l.strip() for l in code.split('\n') if not l.strip().startswith('/*') and not l.strip().startswith('*')]
        for line in lines:
            if 'PyCapsule_New(' in line:
                self.fail(f"Found direct PyCapsule_New call: {line}")

    def test_uses_shared_unwrap_capsule(self):
        """Test that generated code uses shared f90wrap_unwrap_capsule function."""
        gen = CWrapperGenerator(self.root, 'module_a')
        code = gen.generate()

        # Should use f90wrap_unwrap_capsule for extracting pointers
        self.assertIn('f90wrap_unwrap_capsule', code)

    def test_uses_destructor_macro(self):
        """Test that generated code uses F90WRAP_DEFINE_SIMPLE_DESTRUCTOR macro."""
        gen = CWrapperGenerator(self.root, 'module_a')
        code = gen.generate()

        # Each type should have its destructor defined using the macro
        self.assertIn('F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(type_a)', code)
        self.assertIn('F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(type_b)', code)

        # Should not have manual destructor implementations
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'static void' in line and '_capsule_destructor(PyObject *capsule) {' in line:
                # Check it's not the macro-generated one by looking for manual implementation
                if i + 1 < len(lines) and 'PyCapsule_GetPointer' in lines[i + 1]:
                    self.fail(f"Found manual destructor implementation at line {i}: {line}")

    def test_code_size_reduction(self):
        """Test that shared utilities reduce generated code size."""
        # Generate code with multiple derived types
        gen = CWrapperGenerator(self.root, 'module_a')
        code = gen.generate()

        # Count occurrences of common patterns that should be reduced
        lines = code.split('\n')

        # These patterns should appear only in the header, not duplicated
        generic_typedef_count = sum(1 for l in lines if 'typedef struct { PyObject_HEAD void* fortran_ptr; } GenericDerivedType;' in l)
        self.assertEqual(generic_typedef_count, 0, "GenericDerivedType should not be duplicated")

        # The destructor logic should use macro, not be duplicated
        destructor_impl_count = sum(1 for l in lines if 'void* ptr = PyCapsule_GetPointer(capsule,' in l)
        self.assertEqual(destructor_impl_count, 0, "Manual destructor implementations should not exist")

    def test_multiple_modules_share_utilities(self):
        """Test that multiple modules can share the same utilities."""
        # Generate code for both modules
        gen1 = CWrapperGenerator(self.root, 'module_a')
        code1 = gen1.generate()

        gen2 = CWrapperGenerator(self.root, 'module_b')
        code2 = gen2.generate()

        # Both should include the same header
        self.assertIn('#include "capsule_helpers.h"', code1)
        self.assertIn('#include "capsule_helpers.h"', code2)

        # Both should use the shared functions
        self.assertIn('f90wrap_create_capsule', code1)
        self.assertIn('f90wrap_create_capsule', code2)
        self.assertIn('f90wrap_unwrap_capsule', code1)
        self.assertIn('f90wrap_unwrap_capsule', code2)

    def test_clear_capsule_usage(self):
        """Test that f90wrap_clear_capsule is used for preventing double-free."""
        # Add a destructor routine to test
        module = self.root.modules[0]

        sub_destructor = ft.Subroutine('destroy_type_a', filename='test.f90')
        arg = ft.Argument(name='obj', type='type(type_a)')
        arg.intent = 'inout'
        arg.attributes = ['intent(inout)']
        sub_destructor.arguments = [arg]
        sub_destructor.mod_name = 'module_a'
        # Mark as destructor
        sub_destructor.is_destructor = True

        module.routines.append(sub_destructor)

        gen = CWrapperGenerator(self.root, 'module_a')
        code = gen.generate()

        # Should use f90wrap_clear_capsule
        self.assertIn('f90wrap_clear_capsule', code)


if __name__ == '__main__':
    unittest.main()