"""
Test derived type lifecycle management - constructors, destructors, and PyCapsule handling.
"""

import unittest
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestDerivedTypeLifecycle(unittest.TestCase):
    """Test derived type constructor/destructor generation."""

    def setUp(self):
        """Create test AST with a derived type."""
        self.root = ft.Root()

        # Create a module with a derived type
        self.module = ft.Module('test_module')

        # Create a simple derived type
        self.dtype = ft.Type('my_type')
        self.dtype.mod_name = 'test_module'
        self.dtype.elements = [
            ft.Element('value', type='real(8)', attributes=[])
        ]
        self.dtype.procedures = []

        self.module.types = [self.dtype]
        self.module.routines = []

        # Add a function that returns this derived type
        self.func = ft.Function('create_my_type', filename='test.f90')
        self.func.arguments = []
        self.func.ret_val = ft.Argument(name='result', type='type(my_type)')
        self.func.mod_name = 'test_module'
        self.module.routines.append(self.func)

        # Add a subroutine that takes the type as input
        self.sub = ft.Subroutine('process_my_type', filename='test.f90')
        arg = ft.Argument(name='obj', type='type(my_type)')
        arg.intent = 'inout'
        arg.attributes = ['intent(inout)']
        self.sub.arguments = [arg]
        self.sub.mod_name = 'test_module'
        self.module.routines.append(self.sub)

        # Add a subroutine that outputs the type
        self.sub_out = ft.Subroutine('make_my_type', filename='test.f90')
        out_arg = ft.Argument(name='obj', type='type(my_type)')
        out_arg.intent = 'out'
        out_arg.attributes = ['intent(out)']
        self.sub_out.arguments = [out_arg]
        self.sub_out.mod_name = 'test_module'
        self.module.routines.append(self.sub_out)

        self.root.modules = [self.module]
        self.root.procedures = []

    def test_capsule_destructor_generated(self):
        """Test that PyCapsule destructor is generated for derived types."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check that destructor macro is used
        self.assertIn('F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(my_type)', code)

        # Check that shared helpers are included
        self.assertIn('#include "capsule_helpers.h"', code)

    def test_constructor_wrapper_generated(self):
        """Test that constructor wrapper is generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check constructor wrapper
        self.assertIn('wrap_my_type_create', code)
        self.assertIn('Create a new my_type instance', code)
        self.assertIn('f90wrap_my_type__allocate', code)
        # Now using shared helper instead of direct PyCapsule_New
        self.assertIn('f90wrap_create_capsule(ptr, "my_type_capsule", my_type_capsule_destructor)', code)

    def test_destructor_wrapper_generated(self):
        """Test that destructor wrapper is generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check destructor wrapper
        self.assertIn('wrap_my_type_destroy', code)
        self.assertIn('Destroy a my_type instance', code)
        self.assertIn('f90wrap_my_type__deallocate', code)
        # Now using shared helper instead of direct PyCapsule_SetPointer
        self.assertIn('f90wrap_clear_capsule(py_capsule)', code)

    def test_function_returning_type(self):
        """Test that functions returning derived types use PyCapsule."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Function should return PyCapsule with destructor
        self.assertIn('wrap_create_my_type', code)
        self.assertIn('Return derived type my_type as PyCapsule', code)
        # Now using shared helper
        self.assertIn('f90wrap_create_capsule(result, "my_type_capsule", my_type_capsule_destructor)', code)

    def test_subroutine_output_type(self):
        """Test that subroutines with output type arguments use PyCapsule."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Output argument should be wrapped in PyCapsule
        self.assertIn('wrap_make_my_type', code)
        self.assertIn('Return derived type my_type as PyCapsule', code)
        # Now using shared helper
        self.assertIn('f90wrap_create_capsule(obj, "my_type_capsule", my_type_capsule_destructor)', code)

    def test_subroutine_input_type(self):
        """Test that subroutines accepting derived types unwrap PyCapsules."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Input argument should unwrap PyCapsule
        self.assertIn('wrap_process_my_type', code)
        self.assertIn('Unwrap PyCapsule for derived type my_type', code)
        # Now using shared helper
        self.assertIn('f90wrap_unwrap_capsule(py_obj, "my_type")', code)

    def test_multiple_output_types(self):
        """Test handling of multiple output derived types."""
        # Add a subroutine with multiple outputs
        sub_multi = ft.Subroutine('make_two_types', filename='test.f90')
        out1 = ft.Argument(name='obj1', type='type(my_type)')
        out1.intent = 'out'
        out1.attributes = ['intent(out)']
        out2 = ft.Argument(name='obj2', type='type(my_type)')
        out2.intent = 'out'
        out2.attributes = ['intent(out)']
        sub_multi.arguments = [out1, out2]
        sub_multi.mod_name = 'test_module'
        self.module.routines.append(sub_multi)

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check tuple creation with multiple PyCapsules
        self.assertIn('wrap_make_two_types', code)
        self.assertIn('PyTuple_New(2)', code)
        # Now using shared helper
        self.assertIn('f90wrap_create_capsule(obj1, "my_type_capsule", my_type_capsule_destructor)', code)
        self.assertIn('f90wrap_create_capsule(obj2, "my_type_capsule", my_type_capsule_destructor)', code)

    def test_null_handling(self):
        """Test proper NULL pointer handling."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check NULL checks in return values
        self.assertIn('if (result == NULL)', code)
        self.assertIn('Py_RETURN_NONE', code)

        # Check NULL checks in constructor
        self.assertIn('if (ptr == NULL)', code)
        self.assertIn('PyErr_SetString(PyExc_MemoryError', code)


if __name__ == '__main__':
    unittest.main()