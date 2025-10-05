"""
Comprehensive test scenarios for edge cases and complex features.

Tests complex scenarios including:
- Type-bound procedures with various signatures
- Multi-return functions
- Character string handling (fixed and assumed-length)
- Array arguments (1D, 2D, assumed-shape)
- Combinations of features
"""

import unittest
from f90wrap import fortran as ft
from f90wrap.cwrapgen import CWrapperGenerator


class TestTypeBoundProceduresComplex(unittest.TestCase):
    """Test complex type-bound procedure scenarios."""

    def test_type_bound_procedure_returning_derived_type(self):
        """Test type-bound procedure returning another derived type."""
        # Create types
        result_type = ft.Type(name='result_t')
        result_type.elements = [
            ft.Element(name='value', type='real', attributes=['dimension(3)']),
            ft.Element(name='status', type='integer')
        ]

        main_type = ft.Type(name='processor_t')
        main_type.elements = [
            ft.Element(name='data', type='real', attributes=['dimension(:)', 'allocatable'])
        ]

        # Add type-bound procedure
        proc = ft.Procedure(name='process', type='type(result_t)')
        proc.arguments = [
            ft.Argument(name='self', type='type(processor_t)'),
            ft.Argument(name='factor', type='real')
        ]
        proc.method_name = 'process'
        main_type.procedures.append(proc)

        # Generate wrappers
        module = ft.Module(name='complex_tbp')
        module.types = [result_type, main_type]

        root = ft.Root()
        root.modules = [module]

        generator = CWrapperGenerator(root, 'complex_tbp')
        c_code = generator.generate()

        # Verify C wrapper
        self.assertIn('f90wrap_processor_t__process', c_code)
        self.assertIn('result_t', c_code)
        self.assertIn('processor_t', c_code)

    def test_type_bound_subroutine_modifying_self(self):
        """Test type-bound subroutine that modifies the object."""
        dtype = ft.Type(name='counter_t')
        dtype.elements = [
            ft.Element(name='count', type='integer'),
            ft.Element(name='max_count', type='integer')
        ]

        # Add incrementing subroutine
        proc = ft.Subroutine(name='increment')
        proc.arguments = [
            ft.Argument(name='self', type='type(counter_t)', intent='inout'),
            ft.Argument(name='step', type='integer', intent='in', value='1', optional=True)
        ]
        proc.method_name = 'increment'
        dtype.procedures.append(proc)

        module = ft.Module(name='counter_mod')
        module.types = [dtype]

        c_code = self.generator.generate_type(dtype, module)

        # Check for proper intent handling
        self.assertIn('counter_t* self', c_code)
        self.assertIn('int* step', c_code)
        self.assertIn('int* step_present', c_code)

    def test_type_bound_with_callback_argument(self):
        """Test type-bound procedure accepting callback."""
        dtype = ft.Type(name='solver_t')
        dtype.elements = [
            ft.Element(name='tolerance', type='real(8)')
        ]

        # Callback interface
        callback_iface = ft.Interface(name='func_interface')
        callback_proc = ft.Procedure(name='func', type='real(8)')
        callback_proc.arguments = [
            ft.Argument(name='x', type='real(8)', intent='in')
        ]
        callback_iface.procedures.append(callback_proc)

        # Type-bound procedure with callback
        proc = ft.Procedure(name='solve', type='real(8)')
        proc.arguments = [
            ft.Argument(name='self', type='type(solver_t)'),
            ft.Argument(name='func', type='procedure(func_interface)'),
            ft.Argument(name='x0', type='real(8)')
        ]
        proc.method_name = 'solve'
        dtype.procedures.append(proc)

        module = ft.Module(name='solver_mod')
        module.types = [dtype]
        module.interfaces = [callback_iface]

        c_code = self.generator.generate_type(dtype, module)

        self.assertIn('typedef double (*func_interface_t)', c_code)
        self.assertIn('func_interface_t func', c_code)
        self.assertIn('double* x0', c_code)


class TestMultiReturnFunctions(unittest.TestCase):
    """Test functions with multiple return values via arguments."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_subroutine_multiple_outputs(self):
        """Test subroutine with multiple output arguments."""
        proc = ft.Subroutine(name='decompose')
        proc.arguments = [
            ft.Argument(name='input_matrix', type='real(8)',
                       attributes=['dimension(:,:)'], intent='in'),
            ft.Argument(name='eigenvalues', type='real(8)',
                       attributes=['dimension(:)'], intent='out'),
            ft.Argument(name='eigenvectors', type='real(8)',
                       attributes=['dimension(:,:)'], intent='out'),
            ft.Argument(name='status', type='integer', intent='out')
        ]

        module = ft.Module(name='linalg')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Check all outputs are properly handled
        self.assertIn('double* input_matrix', c_code)
        self.assertIn('double* eigenvalues', c_code)
        self.assertIn('double* eigenvectors', c_code)
        self.assertIn('int* status', c_code)

    def test_function_with_intent_out_args(self):
        """Test function with additional intent(out) arguments."""
        proc = ft.Function(name='integrate', type='real(8)')
        proc.arguments = [
            ft.Argument(name='func', type='real(8)', attributes=['external']),
            ft.Argument(name='a', type='real(8)', intent='in'),
            ft.Argument(name='b', type='real(8)', intent='in'),
            ft.Argument(name='error_est', type='real(8)', intent='out'),
            ft.Argument(name='n_evals', type='integer', intent='out')
        ]

        module = ft.Module(name='quadrature')
        module.functions = [proc]

        c_code = self.generator.generate_function(proc, module)

        # Verify return value and out parameters
        self.assertIn('double integrate_', c_code)
        self.assertIn('double* error_est', c_code)
        self.assertIn('int* n_evals', c_code)
        self.assertIn('return result', c_code)

    def test_mixed_intent_arrays(self):
        """Test procedure with mixed intent array arguments."""
        proc = ft.Subroutine(name='filter_data')
        proc.arguments = [
            ft.Argument(name='input', type='real',
                       attributes=['dimension(:)'], intent='in'),
            ft.Argument(name='mask', type='logical',
                       attributes=['dimension(:)'], intent='in'),
            ft.Argument(name='output', type='real',
                       attributes=['dimension(:)'], intent='out'),
            ft.Argument(name='stats', type='real',
                       attributes=['dimension(5)'], intent='out')
        ]

        module = ft.Module(name='filters')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)
        # Fortran support would be generated via self.generator.generate_subroutine_support(proc, module)

        # Check array handling
        self.assertIn('float* input', c_code)
        self.assertIn('int* mask', c_code)  # logical -> int in C
        self.assertIn('float* output', c_code)
        self.assertIn('float* stats', c_code)


class TestCharacterStringHandling(unittest.TestCase):
    """Test character string argument handling."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_fixed_length_character(self):
        """Test fixed-length character arguments."""
        proc = ft.Subroutine(name='format_name')
        proc.arguments = [
            ft.Argument(name='first', type='character(len=20)', intent='in'),
            ft.Argument(name='last', type='character(len=30)', intent='in'),
            ft.Argument(name='full', type='character(len=60)', intent='out')
        ]

        module = ft.Module(name='strings')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Check string handling with lengths
        self.assertIn('char* first', c_code)
        self.assertIn('char* last', c_code)
        self.assertIn('char* full', c_code)
        self.assertIn('int first_len', c_code)
        self.assertIn('int last_len', c_code)
        self.assertIn('int full_len', c_code)

    def test_assumed_length_character(self):
        """Test assumed-length character(*) arguments."""
        proc = ft.Function(name='trim_string', type='integer')
        proc.arguments = [
            ft.Argument(name='input_str', type='character(*)', intent='in'),
            ft.Argument(name='output_str', type='character(*)', intent='out')
        ]

        module = ft.Module(name='string_utils')
        module.functions = [proc]

        c_code = self.generator.generate_function(proc, module)

        # Verify assumed-length handling
        self.assertIn('char* input_str', c_code)
        self.assertIn('int input_str_len', c_code)
        self.assertIn('char* output_str', c_code)
        self.assertIn('int output_str_len', c_code)

    def test_character_in_derived_type(self):
        """Test character components in derived types."""
        dtype = ft.Type(name='person_t')
        dtype.elements = [
            ft.Element(name='name', type='character(len=50)'),
            ft.Element(name='title', type='character(len=20)'),
            ft.Element(name='id', type='integer')
        ]

        module = ft.Module(name='person_mod')
        module.types = [dtype]

        c_code = self.generator.generate_type(dtype, module)

        # Check getters/setters for character components
        self.assertIn('person_t_get_name', c_code)
        self.assertIn('person_t_set_name', c_code)
        self.assertIn('char* name', c_code)
        self.assertIn('int name_len', c_code)

    def test_character_array(self):
        """Test character array arguments."""
        proc = ft.Subroutine(name='process_names')
        proc.arguments = [
            ft.Argument(name='names', type='character(len=30)',
                       attributes=['dimension(:)'], intent='inout'),
            ft.Argument(name='count', type='integer', intent='out')
        ]

        module = ft.Module(name='name_processor')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Character arrays need special handling
        self.assertIn('char* names', c_code)
        self.assertIn('int names_len', c_code)
        self.assertIn('int* count', c_code)


class TestArrayArguments(unittest.TestCase):
    """Test various array argument scenarios."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_assumed_shape_1d_array(self):
        """Test 1D assumed-shape array arguments."""
        proc = ft.Subroutine(name='normalize')
        proc.arguments = [
            ft.Argument(name='data', type='real(8)',
                       attributes=['dimension(:)'], intent='inout'),
            ft.Argument(name='norm_type', type='integer', intent='in', value='2')
        ]

        module = ft.Module(name='array_ops')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        self.assertIn('double* data', c_code)
        self.assertIn('int* data_shape', c_code)
        self.assertIn('int* norm_type', c_code)

    def test_assumed_shape_2d_array(self):
        """Test 2D assumed-shape array arguments."""
        proc = ft.Function(name='matrix_trace', type='real(8)')
        proc.arguments = [
            ft.Argument(name='matrix', type='real(8)',
                       attributes=['dimension(:,:)'], intent='in')
        ]

        module = ft.Module(name='matrix_ops')
        module.functions = [proc]

        c_code = self.generator.generate_function(proc, module)

        self.assertIn('double* matrix', c_code)
        self.assertIn('int* matrix_shape', c_code)
        self.assertIn('double matrix_trace_', c_code)

    def test_explicit_shape_array(self):
        """Test explicit shape array arguments."""
        proc = ft.Subroutine(name='convolve')
        proc.arguments = [
            ft.Argument(name='input', type='real',
                       attributes=['dimension(100)'], intent='in'),
            ft.Argument(name='kernel', type='real',
                       attributes=['dimension(5)'], intent='in'),
            ft.Argument(name='output', type='real',
                       attributes=['dimension(96)'], intent='out')
        ]

        module = ft.Module(name='signal')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Fixed-size arrays don't need shape parameters
        self.assertIn('float* input', c_code)
        self.assertIn('float* kernel', c_code)
        self.assertIn('float* output', c_code)
        self.assertNotIn('input_shape', c_code)

    def test_allocatable_array_in_type(self):
        """Test allocatable array component in derived type."""
        dtype = ft.Type(name='dataset_t')
        dtype.elements = [
            ft.Element(name='data', type='real(8)',
                      attributes=['dimension(:,:)', 'allocatable']),
            ft.Element(name='labels', type='integer',
                      attributes=['dimension(:)', 'allocatable']),
            ft.Element(name='n_samples', type='integer')
        ]

        module = ft.Module(name='dataset')
        module.types = [dtype]

        c_code = self.generator.generate_type(dtype, module)

        # Check allocatable array handling
        self.assertIn('dataset_t_allocate_data', c_code)
        self.assertIn('dataset_t_deallocate_data', c_code)
        self.assertIn('dataset_t_get_data', c_code)
        self.assertIn('int* shape', c_code)

    def test_pointer_array(self):
        """Test pointer array arguments."""
        proc = ft.Subroutine(name='link_arrays')
        proc.arguments = [
            ft.Argument(name='source', type='real(8)',
                       attributes=['dimension(:)', 'target'], intent='in'),
            ft.Argument(name='ptr', type='real(8)',
                       attributes=['dimension(:)', 'pointer'], intent='out')
        ]

        module = ft.Module(name='pointers')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Pointer arrays handled similarly but may need special care
        self.assertIn('double* source', c_code)
        self.assertIn('double** ptr', c_code)  # Pointer to pointer for Fortran pointer


class TestComplexCombinations(unittest.TestCase):
    """Test combinations of features."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_optional_character_array(self):
        """Test optional character array argument."""
        proc = ft.Subroutine(name='print_messages')
        proc.arguments = [
            ft.Argument(name='messages', type='character(len=80)',
                       attributes=['dimension(:)'], intent='in', optional=True),
            ft.Argument(name='prefix', type='character(*)',
                       intent='in', optional=True, value='"INFO: "')
        ]

        module = ft.Module(name='logger')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Check optional array and character handling
        self.assertIn('char* messages', c_code)
        self.assertIn('int messages_len', c_code)
        self.assertIn('int* messages_shape', c_code)
        self.assertIn('int* messages_present', c_code)
        self.assertIn('char* prefix', c_code)
        self.assertIn('int* prefix_present', c_code)

    def test_derived_type_with_procedure_pointer(self):
        """Test derived type with procedure pointer component."""
        # Interface for procedure pointer
        iface = ft.Interface(name='op_interface')
        iface_proc = ft.Function(name='op', type='real(8)')
        iface_proc.arguments = [
            ft.Argument(name='x', type='real(8)', intent='in'),
            ft.Argument(name='y', type='real(8)', intent='in')
        ]
        iface.procedures.append(iface_proc)

        # Type with procedure pointer
        dtype = ft.Type(name='calculator_t')
        dtype.elements = [
            ft.Element(name='operation', type='procedure(op_interface)',
                      attributes=['pointer']),
            ft.Element(name='result', type='real(8)')
        ]

        module = ft.Module(name='calculator')
        module.interfaces = [iface]
        module.types = [dtype]

        c_code = self.generator.generate_type(dtype, module)

        # Check procedure pointer handling
        self.assertIn('typedef double (*op_interface_t)', c_code)
        self.assertIn('calculator_t_set_operation', c_code)
        self.assertIn('op_interface_t operation', c_code)

    def test_nested_derived_types_with_arrays(self):
        """Test nested derived types containing arrays."""
        inner_type = ft.Type(name='vector_t')
        inner_type.elements = [
            ft.Element(name='components', type='real(8)',
                      attributes=['dimension(3)']),
            ft.Element(name='magnitude', type='real(8)')
        ]

        outer_type = ft.Type(name='particle_t')
        outer_type.elements = [
            ft.Element(name='position', type='type(vector_t)'),
            ft.Element(name='velocity', type='type(vector_t)'),
            ft.Element(name='history', type='type(vector_t)',
                      attributes=['dimension(:)', 'allocatable'])
        ]

        module = ft.Module(name='physics')
        module.types = [inner_type, outer_type]

        c_code = self.generator.generate_type(outer_type, module)

        # Check nested type handling
        self.assertIn('particle_t_get_position', c_code)
        self.assertIn('particle_t_set_velocity', c_code)
        self.assertIn('particle_t_allocate_history', c_code)
        self.assertIn('vector_t*', c_code)

    def test_callback_returning_derived_type(self):
        """Test callback that returns a derived type."""
        result_type = ft.Type(name='status_t')
        result_type.elements = [
            ft.Element(name='code', type='integer'),
            ft.Element(name='message', type='character(len=100)')
        ]

        # Callback interface returning derived type
        callback_iface = ft.Interface(name='handler_interface')
        callback_proc = ft.Function(name='handler', type='type(status_t)')
        callback_proc.arguments = [
            ft.Argument(name='event_code', type='integer', intent='in')
        ]
        callback_iface.procedures.append(callback_proc)

        # Function using the callback
        proc = ft.Subroutine(name='process_events')
        proc.arguments = [
            ft.Argument(name='events', type='integer',
                       attributes=['dimension(:)'], intent='in'),
            ft.Argument(name='handler', type='procedure(handler_interface)'),
            ft.Argument(name='results', type='type(status_t)',
                       attributes=['dimension(:)'], intent='out')
        ]

        module = ft.Module(name='event_system')
        module.types = [result_type]
        module.interfaces = [callback_iface]
        module.subroutines = [proc]

        c_code = self.generator.generate_module(module)

        # Verify complex callback handling
        self.assertIn('typedef status_t* (*handler_interface_t)', c_code)
        self.assertIn('handler_interface_t handler', c_code)
        self.assertIn('status_t* results', c_code)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        self.generator = CWrapperGenerator()

    def test_empty_derived_type(self):
        """Test empty derived type (no components)."""
        dtype = ft.Type(name='empty_t')
        dtype.elements = []

        module = ft.Module(name='empty_mod')
        module.types = [dtype]

        c_code = self.generator.generate_type(dtype, module)

        # Should still generate basic structure
        self.assertIn('typedef struct', c_code)
        self.assertIn('empty_t', c_code)

    def test_recursive_type_reference(self):
        """Test type with recursive reference (linked list)."""
        dtype = ft.Type(name='node_t')
        dtype.elements = [
            ft.Element(name='value', type='integer'),
            ft.Element(name='next', type='type(node_t)', attributes=['pointer'])
        ]

        module = ft.Module(name='list_mod')
        module.types = [dtype]

        c_code = self.generator.generate_type(dtype, module)

        # Check forward declaration and pointer handling
        self.assertIn('node_t*', c_code)
        self.assertIn('node_t_get_next', c_code)

    def test_max_dimensions_array(self):
        """Test maximum dimension arrays (7D in Fortran)."""
        proc = ft.Subroutine(name='process_7d')
        proc.arguments = [
            ft.Argument(name='data', type='real',
                       attributes=['dimension(:,:,:,:,:,:,:)'], intent='inout')
        ]

        module = ft.Module(name='multidim')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Should handle up to 7D
        self.assertIn('float* data', c_code)
        self.assertIn('int* data_shape', c_code)
        self.assertIn('data_shape[6]', c_code)  # 7th dimension

    def test_complex_number_types(self):
        """Test complex number type handling."""
        proc = ft.Function(name='compute_fft', type='complex(8)')
        proc.arguments = [
            ft.Argument(name='input', type='complex(8)',
                       attributes=['dimension(:)'], intent='in'),
            ft.Argument(name='inverse', type='logical', intent='in',
                       optional=True, value='.false.')
        ]

        module = ft.Module(name='fft_mod')
        module.functions = [proc]

        c_code = self.generator.generate_function(proc, module)

        # Complex types map to C complex
        self.assertIn('double complex', c_code)
        self.assertIn('double complex* input', c_code)
        self.assertIn('int* inverse', c_code)

    def test_kind_parameters(self):
        """Test various kind parameters for intrinsic types."""
        proc = ft.Subroutine(name='mixed_kinds')
        proc.arguments = [
            ft.Argument(name='i1', type='integer(1)', intent='in'),
            ft.Argument(name='i2', type='integer(2)', intent='in'),
            ft.Argument(name='i4', type='integer(4)', intent='in'),
            ft.Argument(name='i8', type='integer(8)', intent='in'),
            ft.Argument(name='r4', type='real(4)', intent='in'),
            ft.Argument(name='r8', type='real(8)', intent='in'),
            ft.Argument(name='c4', type='complex(4)', intent='in'),
            ft.Argument(name='c8', type='complex(8)', intent='in')
        ]

        module = ft.Module(name='kinds')
        module.subroutines = [proc]

        c_code = self.generator.generate_subroutine(proc, module)

        # Check proper C type mapping
        self.assertIn('signed char* i1', c_code)
        self.assertIn('short* i2', c_code)
        self.assertIn('int* i4', c_code)
        self.assertIn('long long* i8', c_code)
        self.assertIn('float* r4', c_code)
        self.assertIn('double* r8', c_code)
        self.assertIn('float complex* c4', c_code)
        self.assertIn('double complex* c8', c_code)


if __name__ == '__main__':
    unittest.main()