"""
Extended test scenarios for comprehensive feature coverage.

Tests edge cases and complex combinations that extend the base unit tests.
"""

import unittest
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestExtendedScenarios(unittest.TestCase):
    """Test extended scenarios beyond basic coverage."""

    def test_type_bound_procedure_simple(self):
        """Test simple type-bound procedure generation."""
        dtype = ft.Type(name='calculator_t')
        dtype.elements = [
            ft.Element(name='value', type='real(8)')
        ]

        # Add type-bound procedure
        proc = ft.Function(name='compute')
        proc.ret_val = ft.Argument(name='result', type='real(8)')
        proc.arguments = [
            ft.Argument(name='self', type='type(calculator_t)'),
            ft.Argument(name='x', type='real(8)')
        ]
        proc.method_name = 'compute'
        dtype.procedures = [proc]

        module = ft.Module(name='calc_mod')
        module.types = [dtype]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'calc_mod')
        code = generator.generate()

        # Check type-bound procedure wrapper
        self.assertIn('calculator_t', code)
        self.assertIn('compute', code)

    def test_multi_return_via_intent_out(self):
        """Test function with multiple returns via intent(out) arguments."""
        proc = ft.Subroutine(name='decompose')
        proc.arguments = [
            ft.Argument(name='matrix', type='real(8)',
                       attributes=['dimension(3,3)', 'intent(in)']),
            ft.Argument(name='eigenvalues', type='real(8)',
                       attributes=['dimension(3)', 'intent(out)']),
            ft.Argument(name='determinant', type='real(8)', attributes=['intent(out)']),
            ft.Argument(name='rank', type='integer', attributes=['intent(out)'])
        ]

        module = ft.Module(name='linalg')
        module.subroutines = [proc]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'linalg')
        code = generator.generate()

        # Check multiple output handling
        self.assertIn('decompose', code)
        self.assertIn('eigenvalues', code)
        self.assertIn('determinant', code)
        self.assertIn('rank', code)

    def test_character_string_fixed_length(self):
        """Test fixed-length character string handling."""
        proc = ft.Subroutine(name='format_output')
        proc.arguments = [
            ft.Argument(name='input', type='character(len=50)', attributes=['intent(in)']),
            ft.Argument(name='output', type='character(len=100)', attributes=['intent(out)'])
        ]

        module = ft.Module(name='strings')
        module.subroutines = [proc]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'strings')
        code = generator.generate()

        # Check string handling
        self.assertIn('format_output', code)
        self.assertIn('char', code)

    def test_assumed_shape_arrays(self):
        """Test assumed-shape array handling."""
        proc = ft.Function(name='array_sum')
        proc.ret_val = ft.Argument(name='result', type='real(8)')
        proc.arguments = [
            ft.Argument(name='data', type='real(8)',
                       attributes=['dimension(:,:)', 'intent(in)'])
        ]

        module = ft.Module(name='array_ops')
        module.functions = [proc]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'array_ops')
        code = generator.generate()

        # Check array handling
        self.assertIn('array_sum', code)
        self.assertIn('PyArrayObject', code)

    def test_complex_derived_type_nesting(self):
        """Test nested derived types with allocatable components."""
        inner_type = ft.Type(name='point_t')
        inner_type.elements = [
            ft.Element(name='x', type='real(8)'),
            ft.Element(name='y', type='real(8)'),
            ft.Element(name='z', type='real(8)')
        ]

        outer_type = ft.Type(name='trajectory_t')
        outer_type.elements = [
            ft.Element(name='points', type='type(point_t)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='times', type='real(8)',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='n_points', type='integer')
        ]

        module = ft.Module(name='trajectory')
        module.types = [inner_type, outer_type]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'trajectory')
        code = generator.generate()

        # Check nested type handling
        self.assertIn('point_t', code)
        self.assertIn('trajectory_t', code)
        self.assertIn('allocate', code)

    def test_callback_with_derived_type_arg(self):
        """Test callback procedure with derived type argument."""
        # Define type
        data_type = ft.Type(name='data_t')
        data_type.elements = [
            ft.Element(name='values', type='real(8)',
                      attributes=['dimension(:)', 'allocatable'])
        ]

        # Define callback interface
        callback = ft.Interface(name='processor')
        callback_proc = ft.Subroutine(name='process')
        callback_proc.arguments = [
            ft.Argument(name='data', type='type(data_t)', attributes=['intent(inout)'])
        ]
        callback.procedures = [callback_proc]

        # Function using callback
        func = ft.Subroutine(name='apply_processor')
        func.arguments = [
            ft.Argument(name='data', type='type(data_t)', attributes=['intent(inout)']),
            ft.Argument(name='proc', type='procedure(processor)')
        ]

        module = ft.Module(name='processing')
        module.types = [data_type]
        module.interfaces = [callback]
        module.subroutines = [func]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'processing')
        code = generator.generate()

        # Check callback handling
        self.assertIn('processor', code)
        self.assertIn('apply_processor', code)

    def test_optional_array_with_default(self):
        """Test optional array argument with default behavior."""
        proc = ft.Subroutine(name='process_data')
        proc.arguments = [
            ft.Argument(name='input', type='real(8)',
                       attributes=['dimension(:)', 'intent(in)']),
            ft.Argument(name='weights', type='real(8)',
                       attributes=['dimension(:)', 'intent(in)', 'optional']),
            ft.Argument(name='output', type='real(8)',
                       attributes=['dimension(:)', 'intent(out)'])
        ]

        module = ft.Module(name='weighted_ops')
        module.subroutines = [proc]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'weighted_ops')
        code = generator.generate()

        # Check optional array handling
        self.assertIn('process_data', code)
        self.assertIn('weights', code)
        self.assertIn('present', code)

    def test_7d_array_handling(self):
        """Test maximum dimension (7D) array handling."""
        proc = ft.Subroutine(name='process_tensor')
        proc.arguments = [
            ft.Argument(name='tensor', type='real(4)',
                       attributes=['dimension(:,:,:,:,:,:,:)', 'intent(inout)'])
        ]

        module = ft.Module(name='tensor_ops')
        module.subroutines = [proc]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'tensor_ops')
        code = generator.generate()

        # Check 7D array handling
        self.assertIn('process_tensor', code)
        self.assertIn('PyArrayObject', code)

    def test_complex_number_arrays(self):
        """Test complex number array handling."""
        proc = ft.Function(name='fft')
        proc.ret_val = ft.Argument(name='result', type='complex(8)',
                                  attributes=['dimension(:)'])
        proc.arguments = [
            ft.Argument(name='input', type='complex(8)',
                       attributes=['dimension(:)', 'intent(in)'])
        ]

        module = ft.Module(name='fft_mod')
        module.functions = [proc]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'fft_mod')
        code = generator.generate()

        # Check complex type handling
        self.assertIn('fft', code)
        self.assertIn('NPY_COMPLEX', code)

    def test_module_with_global_allocatable(self):
        """Test module with allocatable module variables."""
        module = ft.Module(name='globals')
        module.elements = [
            ft.Element(name='global_buffer', type='real(8)',
                      attributes=['dimension(:,:)', 'allocatable']),
            ft.Element(name='buffer_size', type='integer')
        ]

        # Add initialization routine
        init_sub = ft.Subroutine(name='init_globals')
        init_sub.arguments = [
            ft.Argument(name='size', type='integer', attributes=['intent(in)'])
        ]
        module.subroutines = [init_sub]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'globals')
        code = generator.generate()

        # Check module variable handling
        self.assertIn('global_buffer', code)
        self.assertIn('buffer_size', code)
        self.assertIn('init_globals', code)

    def test_recursive_type_reference(self):
        """Test type with recursive reference (e.g., linked list node)."""
        node_type = ft.Type(name='node_t')
        node_type.elements = [
            ft.Element(name='value', type='integer'),
            ft.Element(name='next', type='type(node_t)', attributes=['pointer'])
        ]

        module = ft.Module(name='linked_list')
        module.types = [node_type]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'linked_list')
        code = generator.generate()

        # Check recursive type handling
        self.assertIn('node_t', code)
        self.assertIn('next', code)

    def test_procedure_pointer_in_type(self):
        """Test derived type with procedure pointer component."""
        # Define interface for procedure pointer
        op_interface = ft.Interface(name='binary_op')
        op_proc = ft.Function(name='op')
        op_proc.ret_val = ft.Argument(name='result', type='real(8)')
        op_proc.arguments = [
            ft.Argument(name='a', type='real(8)', attributes=['intent(in)']),
            ft.Argument(name='b', type='real(8)', attributes=['intent(in)'])
        ]
        op_interface.procedures = [op_proc]

        # Type with procedure pointer
        calc_type = ft.Type(name='calculator_t')
        calc_type.elements = [
            ft.Element(name='operation', type='procedure(binary_op)',
                      attributes=['pointer']),
            ft.Element(name='result', type='real(8)')
        ]

        module = ft.Module(name='proc_ptr_test')
        module.interfaces = [op_interface]
        module.types = [calc_type]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'proc_ptr_test')
        code = generator.generate()

        # Check procedure pointer handling
        self.assertIn('binary_op', code)
        self.assertIn('calculator_t', code)

    def test_character_array_2d(self):
        """Test 2D character array handling."""
        proc = ft.Subroutine(name='process_names')
        proc.arguments = [
            ft.Argument(name='names', type='character(len=30)',
                       attributes=['dimension(10,5)', 'intent(inout)'])
        ]

        module = ft.Module(name='name_grid')
        module.subroutines = [proc]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'name_grid')
        code = generator.generate()

        # Check 2D character array handling
        self.assertIn('process_names', code)
        self.assertIn('char', code)

    def test_mixed_kind_parameters(self):
        """Test various kind parameters for intrinsic types."""
        proc = ft.Subroutine(name='mixed_precision')
        proc.arguments = [
            ft.Argument(name='i1', type='integer(1)', attributes=['intent(in)']),
            ft.Argument(name='i2', type='integer(2)', attributes=['intent(in)']),
            ft.Argument(name='i4', type='integer(4)', attributes=['intent(in)']),
            ft.Argument(name='i8', type='integer(8)', attributes=['intent(in)']),
            ft.Argument(name='r4', type='real(4)', attributes=['intent(in)']),
            ft.Argument(name='r8', type='real(8)', attributes=['intent(in)'])
        ]

        module = ft.Module(name='precision_test')
        module.subroutines = [proc]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'precision_test')
        code = generator.generate()

        # Check type mapping for different kinds
        self.assertIn('mixed_precision', code)
        # Should handle different integer and real kinds

    def test_empty_derived_type(self):
        """Test empty derived type (edge case)."""
        empty_type = ft.Type(name='empty_t')
        empty_type.elements = []

        module = ft.Module(name='empty_test')
        module.types = [empty_type]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'empty_test')
        code = generator.generate()

        # Should handle empty type gracefully
        self.assertIn('empty_t', code)


if __name__ == '__main__':
    unittest.main()