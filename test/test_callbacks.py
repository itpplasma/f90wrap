"""
Unit tests for callback wrapper generation in cwrapgen.py.

Tests callback argument handling, capsule unwrapping, and Python callable
integration with Fortran procedures.
"""

import unittest
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestCallbackWrapperGeneration(unittest.TestCase):
    """Test callback wrapper emission."""

    def setUp(self):
        """Create AST with callback procedures."""
        self.root = ft.Root('test')
        self.module = ft.Module('callback_mod', filename='test.f90')

    def test_simple_callback_argument(self):
        """Test basic callback argument handling."""
        # Create subroutine with callback argument
        sub = ft.Subroutine('call_callback', filename='test.f90')
        sub.mod_name = 'callback_mod'

        # Add a callback argument
        callback_arg = ft.Argument('my_callback', filename='test.f90')
        callback_arg.type = 'callback'
        callback_arg.attributes = ['callback']

        # Add a regular argument
        data_arg = ft.Argument('data', filename='test.f90')
        data_arg.type = 'real(8)'
        data_arg.attributes = ['intent(in)']

        sub.arguments = [callback_arg, data_arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check callback validation
        self.assertIn('PyCallable_Check', code)
        self.assertIn('Argument my_callback must be callable', code)

        # Check callback storage
        self.assertIn('Py_XINCREF(py_my_callback)', code)
        self.assertIn('void* my_callback = (void*)py_my_callback', code)

        # Check format string includes 'O' for callback
        self.assertIn('PyArg_ParseTuple(args, "Od"', code)

    def test_external_procedure_callback(self):
        """Test external procedure as callback."""
        sub = ft.Subroutine('use_external', filename='test.f90')
        sub.mod_name = 'callback_mod'

        # External procedure argument
        ext_arg = ft.Argument('external_func', filename='test.f90')
        ext_arg.type = 'callback'
        ext_arg.attributes = ['external', 'callback']

        sub.arguments = [ext_arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check external handling
        self.assertIn('Callback argument external_func', code)
        self.assertIn('PyCallable_Check(py_external_func)', code)
        self.assertIn('Pass Python callable as opaque pointer', code)

    def test_derived_type_capsule_unwrap(self):
        """Test PyCapsule unwrapping for derived types."""
        # Create a derived type
        dtype = ft.Type('my_type', filename='test.f90')
        dtype.mod_name = 'callback_mod'
        dtype.elements = []
        self.module.types = [dtype]

        # Create subroutine with derived type argument
        sub = ft.Subroutine('use_type', filename='test.f90')
        sub.mod_name = 'callback_mod'

        type_arg = ft.Argument('obj', filename='test.f90')
        type_arg.type = 'type(my_type)'
        type_arg.attributes = ['intent(in)']

        sub.arguments = [type_arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check capsule unwrapping using shared helper
        self.assertIn('Unwrap PyCapsule for derived type my_type', code)
        self.assertIn('f90wrap_unwrap_capsule', code)
        self.assertIn('my_type', code)

    def test_multiple_callbacks_in_subroutine(self):
        """Test subroutine with multiple callback arguments."""
        sub = ft.Subroutine('multi_callback', filename='test.f90')
        sub.mod_name = 'callback_mod'

        # First callback
        callback1 = ft.Argument('callback_one', filename='test.f90')
        callback1.type = 'callback'
        callback1.attributes = ['callback']

        # Second callback
        callback2 = ft.Argument('callback_two', filename='test.f90')
        callback2.type = 'callback'
        callback2.attributes = ['callback']

        # Data argument
        data = ft.Argument('input', filename='test.f90')
        data.type = 'integer'
        data.attributes = ['intent(in)']

        sub.arguments = [callback1, callback2, data]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check both callbacks are handled
        self.assertIn('Callback argument callback_one', code)
        self.assertIn('Callback argument callback_two', code)
        self.assertIn('PyCallable_Check(py_callback_one)', code)
        self.assertIn('PyCallable_Check(py_callback_two)', code)

        # Check format string has two 'O's for callbacks and 'i' for integer
        self.assertIn('PyArg_ParseTuple(args, "OOi"', code)

    def test_mixed_capsule_and_callback(self):
        """Test procedure with both capsule and callback arguments."""
        # Add a type for capsule testing
        dtype = ft.Type('data_type', filename='test.f90')
        dtype.mod_name = 'callback_mod'
        dtype.elements = []
        self.module.types = [dtype]

        sub = ft.Subroutine('process', filename='test.f90')
        sub.mod_name = 'callback_mod'

        # Capsule argument
        capsule_arg = ft.Argument('data_obj', filename='test.f90')
        capsule_arg.type = 'type(data_type)'
        capsule_arg.attributes = ['intent(in)']

        # Callback argument
        callback_arg = ft.Argument('handler', filename='test.f90')
        callback_arg.type = 'callback'
        callback_arg.attributes = ['callback']

        sub.arguments = [capsule_arg, callback_arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check capsule handling
        self.assertIn('Unwrap PyCapsule for derived type data_type', code)

        # Check callback handling
        self.assertIn('Callback argument handler', code)

        # Both should use 'O' format
        self.assertIn('PyArg_ParseTuple(args, "OO"', code)

    def test_callback_no_stub_generated(self):
        """Verify no stubs or placeholders in callback code."""
        sub = ft.Subroutine('with_callback', filename='test.f90')
        sub.mod_name = 'callback_mod'

        callback_arg = ft.Argument('func', filename='test.f90')
        callback_arg.type = 'callback'
        callback_arg.attributes = ['callback', 'intent(in)']

        sub.arguments = [callback_arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # No stubs or placeholders should be present
        self.assertNotIn('TODO', code)
        self.assertNotIn('FIXME', code)
        self.assertNotIn('NotImplemented', code)
        self.assertNotIn('stub', code.lower())

        # Should have complete implementation
        self.assertIn('PyCallable_Check', code)
        self.assertIn('Py_XINCREF', code)

    def test_callback_in_type_bound_procedure(self):
        """Test callback argument in type-bound procedure."""
        # Create a type with a method that takes a callback
        dtype = ft.Type('processor', filename='test.f90')
        dtype.mod_name = 'callback_mod'
        dtype.elements = []

        method = ft.Subroutine('process_with_callback', filename='test.f90')

        callback_arg = ft.Argument('handler', filename='test.f90')
        callback_arg.type = 'callback'
        callback_arg.attributes = ['callback']

        method.arguments = [callback_arg]
        dtype.procedures = [method]

        self.module.types = [dtype]
        self.module.routines = []
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check type-bound method with callback
        self.assertIn('processor_process_with_callback', code)
        self.assertIn('PyCallable_Check', code)
        self.assertIn('self->fortran_ptr', code)


if __name__ == '__main__':
    unittest.main()