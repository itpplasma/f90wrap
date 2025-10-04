"""
Comprehensive unit tests for C wrapper generator (cwrapgen.py).

Tests all aspects of direct C code generation including:
- Type conversions (Fortran <-> C <-> NumPy)
- Name mangling (all compiler conventions)
- Template rendering
- Code generation validity
"""

import unittest
import numpy as np
from f90wrap import fortran as ft
from f90wrap.cwrapgen import (
    FortranCTypeMap,
    FortranNameMangler,
    CCodeTemplate,
    CCodeGenerator,
    CWrapperGenerator
)


class TestFortranCTypeMap(unittest.TestCase):
    """Test type mapping system."""

    def setUp(self):
        self.type_map = FortranCTypeMap()

    def test_integer_types(self):
        """Test all integer type conversions."""
        test_cases = [
            ('integer', 'int', 'NPY_INT32', 'i'),
            ('integer(4)', 'int', 'NPY_INT32', 'i'),
            ('integer(8)', 'long long', 'NPY_INT64', 'L'),
            ('integer(2)', 'short', 'NPY_INT16', 'h'),
            ('integer(1)', 'signed char', 'NPY_INT8', 'b'),
        ]

        for fortran_type, expected_c, expected_numpy, expected_fmt in test_cases:
            with self.subTest(fortran_type=fortran_type):
                self.assertEqual(self.type_map.fortran_to_c_type(fortran_type), expected_c)
                self.assertEqual(self.type_map.fortran_to_numpy_type(fortran_type), expected_numpy)
                self.assertEqual(self.type_map.get_parse_format(fortran_type), expected_fmt)

    def test_real_types(self):
        """Test all real type conversions."""
        test_cases = [
            ('real', 'float', 'NPY_FLOAT32', 'f'),
            ('real(4)', 'float', 'NPY_FLOAT32', 'f'),
            ('real(8)', 'double', 'NPY_FLOAT64', 'd'),
            ('double precision', 'double', 'NPY_FLOAT64', 'd'),
        ]

        for fortran_type, expected_c, expected_numpy, expected_fmt in test_cases:
            with self.subTest(fortran_type=fortran_type):
                self.assertEqual(self.type_map.fortran_to_c_type(fortran_type), expected_c)
                self.assertEqual(self.type_map.fortran_to_numpy_type(fortran_type), expected_numpy)
                self.assertEqual(self.type_map.get_parse_format(fortran_type), expected_fmt)

    def test_complex_types(self):
        """Test complex type conversions."""
        test_cases = [
            ('complex', 'float complex', 'NPY_COMPLEX64'),
            ('complex(4)', 'float complex', 'NPY_COMPLEX64'),
            ('complex(8)', 'double complex', 'NPY_COMPLEX128'),
        ]

        for fortran_type, expected_c, expected_numpy in test_cases:
            with self.subTest(fortran_type=fortran_type):
                self.assertEqual(self.type_map.fortran_to_c_type(fortran_type), expected_c)
                self.assertEqual(self.type_map.fortran_to_numpy_type(fortran_type), expected_numpy)

    def test_logical_types(self):
        """Test logical type conversions."""
        test_cases = [
            ('logical', 'int', 'NPY_BOOL', 'p'),
            ('logical(4)', 'int', 'NPY_BOOL', 'p'),
        ]

        for fortran_type, expected_c, expected_numpy, expected_fmt in test_cases:
            with self.subTest(fortran_type=fortran_type):
                self.assertEqual(self.type_map.fortran_to_c_type(fortran_type), expected_c)
                self.assertEqual(self.type_map.fortran_to_numpy_type(fortran_type), expected_numpy)
                self.assertEqual(self.type_map.get_parse_format(fortran_type), expected_fmt)

    def test_character_types(self):
        """Test character/string type conversions."""
        fortran_type = 'character'
        self.assertEqual(self.type_map.fortran_to_c_type(fortran_type), 'char*')
        self.assertEqual(self.type_map.fortran_to_numpy_type(fortran_type), 'NPY_STRING')
        self.assertEqual(self.type_map.get_parse_format(fortran_type), 's')

    def test_derived_types(self):
        """Test derived type handling."""
        test_cases = ['type(foo)', 'class(bar)']

        for fortran_type in test_cases:
            with self.subTest(fortran_type=fortran_type):
                self.assertEqual(self.type_map.fortran_to_c_type(fortran_type), 'void*')
                self.assertEqual(self.type_map.get_parse_format(fortran_type), 'O')

    def test_converters(self):
        """Test Python/C converter function names."""
        test_cases = [
            ('integer', 'PyLong_AsLong', 'PyLong_FromLong'),
            ('real', 'PyFloat_AsDouble', 'PyFloat_FromDouble'),
            ('logical', 'PyObject_IsTrue', 'PyBool_FromLong'),
        ]

        for fortran_type, expected_py_to_c, expected_c_to_py in test_cases:
            with self.subTest(fortran_type=fortran_type):
                self.assertEqual(self.type_map.get_py_to_c_converter(fortran_type), expected_py_to_c)
                self.assertEqual(self.type_map.get_c_to_py_converter(fortran_type), expected_c_to_py)

    def test_unknown_type_raises(self):
        """Test that unknown types raise ValueError."""
        with self.assertRaises(ValueError):
            self.type_map.fortran_to_c_type('unknown_type')


class TestFortranNameMangler(unittest.TestCase):
    """Test Fortran name mangling."""

    def test_gfortran_free_procedure(self):
        """Test gfortran mangling for free procedures."""
        mangler = FortranNameMangler('gfortran')
        self.assertEqual(mangler.mangle('myfunction'), 'myfunction_')
        self.assertEqual(mangler.mangle('MyFunction'), 'myfunction_')

    def test_gfortran_module_procedure(self):
        """Test gfortran mangling for module procedures."""
        mangler = FortranNameMangler('gfortran')
        self.assertEqual(
            mangler.mangle('myfunction', 'mymodule'),
            '__mymodule_MOD_myfunction_'
        )

    def test_ifort_free_procedure(self):
        """Test ifort mangling for free procedures."""
        mangler = FortranNameMangler('ifort')
        self.assertEqual(mangler.mangle('myfunction'), 'myfunction_')

    def test_ifort_module_procedure(self):
        """Test ifort mangling for module procedures."""
        mangler = FortranNameMangler('ifort')
        self.assertEqual(
            mangler.mangle('myfunction', 'mymodule'),
            'mymodule_mp_myfunction_'
        )

    def test_ifx_module_procedure(self):
        """Test ifx mangling (same as ifort)."""
        mangler = FortranNameMangler('ifx')
        self.assertEqual(
            mangler.mangle('myfunction', 'mymodule'),
            'mymodule_mp_myfunction_'
        )

    def test_f77_mangling(self):
        """Test f77 mangling."""
        mangler = FortranNameMangler('f77')
        self.assertEqual(mangler.mangle('myfunction'), 'myfunction_')
        self.assertEqual(mangler.mangle('myfunction', 'mymodule'), 'myfunction_')

    def test_demangle_gfortran(self):
        """Test demangling gfortran names."""
        mangler = FortranNameMangler('gfortran')

        name, mod = mangler.demangle('myfunction_')
        self.assertEqual(name, 'myfunction')
        self.assertIsNone(mod)

        name, mod = mangler.demangle('__mymodule_MOD_myfunction_')
        self.assertEqual(name, 'myfunction')
        self.assertEqual(mod, 'mymodule')

    def test_demangle_ifort(self):
        """Test demangling ifort names."""
        mangler = FortranNameMangler('ifort')

        name, mod = mangler.demangle('myfunction_')
        self.assertEqual(name, 'myfunction')
        self.assertIsNone(mod)

        name, mod = mangler.demangle('mymodule_mp_myfunction_')
        self.assertEqual(name, 'myfunction')
        self.assertEqual(mod, 'mymodule')

    def test_unknown_convention_raises(self):
        """Test that unknown conventions raise ValueError."""
        with self.assertRaises(ValueError):
            FortranNameMangler('unknown_compiler')


class TestCCodeTemplate(unittest.TestCase):
    """Test C code templates."""

    def test_module_header(self):
        """Test module header generation."""
        header = CCodeTemplate.module_header('test_module')
        self.assertIn('#include <Python.h>', header)
        self.assertIn('#include <numpy/arrayobject.h>', header)
        self.assertIn('test_module', header)

    def test_fortran_prototype_void(self):
        """Test Fortran prototype for void function."""
        proto = CCodeTemplate.fortran_prototype(
            'my_sub_',
            'void',
            [('int', 'x'), ('double', 'y')]
        )
        self.assertIn('extern void my_sub_', proto)
        self.assertIn('int *x', proto)
        self.assertIn('double *y', proto)

    def test_fortran_prototype_with_return(self):
        """Test Fortran prototype for function with return value."""
        proto = CCodeTemplate.fortran_prototype(
            'my_func_',
            'double',
            [('int', 'n')]
        )
        self.assertIn('extern double my_func_', proto)
        self.assertIn('int *n', proto)

    def test_fortran_prototype_no_args(self):
        """Test Fortran prototype for no-argument function."""
        proto = CCodeTemplate.fortran_prototype('my_sub_', 'void', [])
        self.assertIn('extern void my_sub_(void)', proto)

    def test_function_wrapper_start(self):
        """Test function wrapper start generation."""
        wrapper = CCodeTemplate.function_wrapper_start('wrap_test', 'Test function')
        self.assertIn('static PyObject* wrap_test', wrapper)
        self.assertIn('Test function', wrapper)
        self.assertIn('PyObject *args', wrapper)

    def test_function_wrapper_end(self):
        """Test function wrapper end generation."""
        wrapper = CCodeTemplate.function_wrapper_end('return result')
        self.assertIn('return result', wrapper)
        self.assertIn('}', wrapper)

    def test_parse_args(self):
        """Test argument parsing code generation."""
        parse = CCodeTemplate.parse_args('ii', ['x', 'y'])
        self.assertIn('PyArg_ParseTuple', parse)
        self.assertIn('"ii"', parse)
        self.assertIn('&x', parse)
        self.assertIn('&y', parse)
        self.assertIn('return NULL', parse)

    def test_method_def(self):
        """Test method definition generation."""
        method = CCodeTemplate.method_def('wrap_test', 'METH_VARARGS')
        self.assertIn('"wrap_test"', method)
        self.assertIn('wrap_test', method)
        self.assertIn('METH_VARARGS', method)

    def test_module_init(self):
        """Test module initialization generation."""
        methods = ['    {"test1", test1, METH_VARARGS, "doc1"},']
        init = CCodeTemplate.module_init('mymodule', methods)

        self.assertIn('PyMethodDef mymodule_methods[]', init)
        self.assertIn('test1', init)
        self.assertIn('PyModuleDef mymodule_module', init)
        self.assertIn('PyInit_mymodule', init)
        self.assertIn('import_array()', init)


class TestCCodeGenerator(unittest.TestCase):
    """Test code generation buffer."""

    def test_write_with_indentation(self):
        """Test writing with indentation."""
        gen = CCodeGenerator(indent=4)
        gen.write('line1')
        gen.indent()
        gen.write('line2')
        gen.indent()
        gen.write('line3')

        code = str(gen)
        self.assertIn('line1', code)
        self.assertIn('    line2', code)
        self.assertIn('        line3', code)

    def test_dedent(self):
        """Test dedenting."""
        gen = CCodeGenerator(indent=4)
        gen.indent()
        gen.indent()
        gen.write('deeply nested')
        gen.dedent()
        gen.write('less nested')

        code = str(gen)
        self.assertIn('        deeply nested', code)
        self.assertIn('    less nested', code)

    def test_write_raw(self):
        """Test writing without indentation."""
        gen = CCodeGenerator(indent=4)
        gen.indent()
        gen.write_raw('no indent')

        code = str(gen)
        self.assertEqual(code, 'no indent')

    def test_empty_lines(self):
        """Test writing empty lines."""
        gen = CCodeGenerator()
        gen.write('line1')
        gen.write('')
        gen.write('line2')

        lines = str(gen).split('\n')
        self.assertEqual(lines[0], 'line1')
        self.assertEqual(lines[1], '')
        self.assertEqual(lines[2], 'line2')


class TestCWrapperGenerator(unittest.TestCase):
    """Test complete C wrapper generation."""

    def setUp(self):
        """Create simple AST for testing."""
        self.root = ft.Root('test')

        # Create a simple module
        self.module = ft.Module('test_mod', filename='test.f90')

        # Add a simple subroutine
        self.sub = ft.Subroutine('test_sub', filename='test.f90')
        self.sub.mod_name = 'test_mod'

        # Add arguments
        arg1 = ft.Argument('x', filename='test.f90')
        arg1.type = 'integer'
        arg1.attributes = []

        arg2 = ft.Argument('y', filename='test.f90')
        arg2.type = 'real(8)'
        arg2.attributes = []

        self.sub.arguments = [arg1, arg2]
        self.module.routines = [self.sub]
        self.root.modules = [self.module]

    def test_type_definition_generation(self):
        """Test derived type definition generation."""
        # Add a derived type
        dtype = ft.Type('mytype', filename='test.f90')
        self.module.types = [dtype]

        gen = CWrapperGenerator(self.root, 'test_module')
        gen._generate_type_definitions()

        code = str(gen.code_gen)
        self.assertIn('typedef struct', code)
        self.assertIn('Pymytype', code)
        self.assertIn('PyObject_HEAD', code)
        self.assertIn('void* fortran_ptr', code)

    def test_fortran_prototype_generation(self):
        """Test Fortran prototype generation."""
        gen = CWrapperGenerator(self.root, 'test_module')
        gen._generate_fortran_prototypes()

        code = str(gen.code_gen)
        self.assertIn('extern void', code)
        self.assertIn('test_mod', code.lower())
        self.assertIn('test_sub', code.lower())

    def test_wrapper_function_generation(self):
        """Test wrapper function generation."""
        gen = CWrapperGenerator(self.root, 'test_module')
        gen._generate_wrapper_functions()

        code = str(gen.code_gen)
        self.assertIn('static PyObject*', code)
        self.assertIn('wrap_test_sub', code)
        self.assertIn('PyArg_ParseTuple', code)

    def test_complete_generation(self):
        """Test complete C module generation."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check for essential components
        self.assertIn('#include <Python.h>', code)
        self.assertIn('#include <numpy/arrayobject.h>', code)
        self.assertIn('extern', code)
        self.assertIn('static PyObject*', code)
        self.assertIn('PyMethodDef', code)
        self.assertIn('PyInit_test_module', code)

    def test_custom_compiler_convention(self):
        """Test using custom compiler convention."""
        config = {'compiler': 'ifort'}
        gen = CWrapperGenerator(self.root, 'test_module', config)
        gen._generate_fortran_prototypes()

        code = str(gen.code_gen)
        # Should use ifort convention: module_mp_procedure_
        self.assertIn('_mp_', code)

    def test_no_stubs_or_placeholders(self):
        """Verify no incomplete implementations."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # These should not appear in complete implementation
        self.assertNotIn('NotImplementedError', code)
        self.assertNotIn('TODO', code)
        self.assertNotIn('FIXME', code)
        self.assertNotIn('XXX', code)
        self.assertNotIn('stub', code.lower())


class TestPhase2ScalarArguments(unittest.TestCase):
    """Test Phase 2.1: Scalar argument wrapper generation."""

    def setUp(self):
        """Create AST with scalar argument procedures."""
        self.root = ft.Root('test')
        self.module = ft.Module('test_mod', filename='test.f90')

    def test_intent_in_scalar(self):
        """Test intent(in) scalar argument."""
        # Create subroutine with intent(in) argument
        sub = ft.Subroutine('test_in', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg = ft.Argument('x', filename='test.f90')
        arg.type = 'integer'
        arg.attributes = ['intent(in)']

        sub.arguments = [arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check for proper intent(in) handling
        self.assertIn('PyArg_ParseTuple', code)
        self.assertIn('PyLong_AsLong', code)
        self.assertIn('&x', code)  # Should pass by reference to Fortran

    def test_intent_out_scalar(self):
        """Test intent(out) scalar argument."""
        sub = ft.Subroutine('test_out', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg = ft.Argument('result', filename='test.f90')
        arg.type = 'real(8)'
        arg.attributes = ['intent(out)']

        sub.arguments = [arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check for proper intent(out) handling
        self.assertIn('result = 0', code)  # Initialization
        self.assertIn('PyFloat_FromDouble', code)  # Return conversion
        self.assertIn('return', code)

    def test_intent_inout_scalar(self):
        """Test intent(inout) scalar argument."""
        sub = ft.Subroutine('test_inout', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg = ft.Argument('value', filename='test.f90')
        arg.type = 'integer'
        arg.attributes = ['intent(inout)']

        sub.arguments = [arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check for proper intent(inout) handling
        self.assertIn('PyArg_ParseTuple', code)  # Input parsing
        self.assertIn('PyLong_AsLong', code)  # Input conversion
        self.assertIn('PyLong_FromLong', code)  # Output conversion
        self.assertIn('return', code)

    def test_multiple_scalar_inputs(self):
        """Test multiple intent(in) scalar arguments."""
        sub = ft.Subroutine('test_multi', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg1 = ft.Argument('x', filename='test.f90')
        arg1.type = 'integer'
        arg1.attributes = ['intent(in)']

        arg2 = ft.Argument('y', filename='test.f90')
        arg2.type = 'real(8)'
        arg2.attributes = ['intent(in)']

        arg3 = ft.Argument('z', filename='test.f90')
        arg3.type = 'logical'
        arg3.attributes = ['intent(in)']

        sub.arguments = [arg1, arg2, arg3]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check all arguments are handled
        self.assertIn('py_x', code)
        self.assertIn('py_y', code)
        self.assertIn('py_z', code)
        self.assertIn('"idp"', code)  # Format string: integer, double, predicate

    def test_multiple_outputs_tuple(self):
        """Test multiple intent(out) returns tuple."""
        sub = ft.Subroutine('test_outputs', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg1 = ft.Argument('a', filename='test.f90')
        arg1.type = 'integer'
        arg1.attributes = ['intent(out)']

        arg2 = ft.Argument('b', filename='test.f90')
        arg2.type = 'real(8)'
        arg2.attributes = ['intent(out)']

        sub.arguments = [arg1, arg2]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check tuple creation for multiple outputs
        self.assertIn('PyTuple_New(2)', code)
        self.assertIn('PyTuple_SET_ITEM', code)
        self.assertIn('result_tuple', code)

    def test_mixed_intent_arguments(self):
        """Test mixed intent(in), intent(out), intent(inout)."""
        sub = ft.Subroutine('test_mixed', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg_in = ft.Argument('input_val', filename='test.f90')
        arg_in.type = 'integer'
        arg_in.attributes = ['intent(in)']

        arg_out = ft.Argument('output_val', filename='test.f90')
        arg_out.type = 'real(8)'
        arg_out.attributes = ['intent(out)']

        arg_inout = ft.Argument('modify_val', filename='test.f90')
        arg_inout.type = 'integer'
        arg_inout.attributes = ['intent(inout)']

        sub.arguments = [arg_in, arg_out, arg_inout]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check proper handling of mixed intents
        self.assertIn('py_input_val', code)
        self.assertIn('py_modify_val', code)
        self.assertIn('output_val = 0', code)  # out initialized
        self.assertIn('PyTuple_New(2)', code)  # Returns output_val and modify_val

    def test_function_with_scalar_return(self):
        """Test function with scalar return value."""
        func = ft.Function('test_func', filename='test.f90')
        func.mod_name = 'test_mod'

        arg = ft.Argument('x', filename='test.f90')
        arg.type = 'integer'
        arg.attributes = ['intent(in)']

        ret = ft.Argument('result', filename='test.f90')
        ret.type = 'real(8)'

        func.arguments = [arg]
        func.ret_val = ret
        self.module.routines = [func]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check function call with return
        self.assertIn('double result', code)
        self.assertIn('result =', code)
        self.assertIn('PyFloat_FromDouble(result)', code)

    def test_complex_scalar(self):
        """Test complex number scalar argument."""
        sub = ft.Subroutine('test_complex', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg = ft.Argument('z', filename='test.f90')
        arg.type = 'complex(8)'
        arg.attributes = ['intent(in)']

        sub.arguments = [arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check complex handling
        self.assertIn('double complex', code)

    def test_logical_scalar(self):
        """Test logical scalar argument."""
        sub = ft.Subroutine('test_logical', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg_in = ft.Argument('flag', filename='test.f90')
        arg_in.type = 'logical'
        arg_in.attributes = ['intent(in)']

        arg_out = ft.Argument('result', filename='test.f90')
        arg_out.type = 'logical'
        arg_out.attributes = ['intent(out)']

        sub.arguments = [arg_in, arg_out]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check logical handling
        self.assertIn('PyObject_IsTrue', code)
        self.assertIn('PyBool_FromLong', code)

    def test_default_intent_is_in(self):
        """Test that missing intent defaults to intent(in)."""
        sub = ft.Subroutine('test_default', filename='test.f90')
        sub.mod_name = 'test_mod'

        arg = ft.Argument('x', filename='test.f90')
        arg.type = 'integer'
        arg.attributes = []  # No intent specified

        sub.arguments = [arg]
        self.module.routines = [sub]
        self.root.modules = [self.module]

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Should behave like intent(in)
        self.assertIn('PyArg_ParseTuple', code)
        self.assertIn('py_x', code)


class TestPhase3DerivedTypes(unittest.TestCase):
    """Test Phase 3: Derived type support."""

    def setUp(self):
        # Create a simple derived type
        self.dtype = ft.Type('simple_type', filename='test.f90')
        self.dtype.mod_name = 'test_mod'

        # Add scalar elements
        alpha = ft.Element('alpha', filename='test.f90')
        alpha.type = 'logical'
        alpha.attributes = []

        beta = ft.Element('beta', filename='test.f90')
        beta.type = 'integer(4)'
        beta.attributes = []

        delta = ft.Element('delta', filename='test.f90')
        delta.type = 'real(8)'
        delta.attributes = []

        self.dtype.elements = [alpha, beta, delta]
        self.dtype.procedures = []
        self.dtype.bindings = []
        self.dtype.interfaces = []

        # Create module and root
        self.module = ft.Module('test_mod', filename='test.f90')
        self.module.types = [self.dtype]
        self.module.procedures = []

        self.root = ft.Root()
        self.root.modules = [self.module]

    def test_type_struct_generated(self):
        """Test that PyTypeObject struct is generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        self.assertIn('typedef struct {', code)
        self.assertIn('PyObject_HEAD', code)
        self.assertIn('void* fortran_ptr', code)
        self.assertIn('int owns_memory', code)
        self.assertIn('} Pysimple_type;', code)

    def test_type_constructor_generated(self):
        """Test that constructor is generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        self.assertIn('static PyObject* simple_type_new', code)
        self.assertIn('self->fortran_ptr = NULL', code)
        self.assertIn('self->owns_memory = 0', code)
        self.assertIn('malloc(sizeof(int) * 8)', code)

    def test_type_destructor_generated(self):
        """Test that destructor is generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        self.assertIn('static void simple_type_dealloc', code)
        self.assertIn('free(self->fortran_ptr)', code)
        self.assertIn('Py_TYPE(self)->tp_free', code)

    def test_element_getter_generated(self):
        """Test that element getters are generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check getters for all elements
        self.assertIn('static PyObject* simple_type_get_alpha', code)
        self.assertIn('static PyObject* simple_type_get_beta', code)
        self.assertIn('static PyObject* simple_type_get_delta', code)

        # Check Fortran calls
        self.assertIn('f90wrap_simple_type__get__alpha', code)
        self.assertIn('f90wrap_simple_type__get__beta', code)
        self.assertIn('f90wrap_simple_type__get__delta', code)

    def test_element_setter_generated(self):
        """Test that element setters are generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check setters for all elements
        self.assertIn('static int simple_type_set_alpha', code)
        self.assertIn('static int simple_type_set_beta', code)
        self.assertIn('static int simple_type_set_delta', code)

        # Check Fortran calls
        self.assertIn('f90wrap_simple_type__set__alpha', code)
        self.assertIn('f90wrap_simple_type__set__beta', code)
        self.assertIn('f90wrap_simple_type__set__delta', code)

    def test_getset_table_generated(self):
        """Test that PyGetSetDef table is generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        self.assertIn('static PyGetSetDef simple_type_getsetters[]', code)
        self.assertIn('{"alpha", (getter)simple_type_get_alpha', code)
        self.assertIn('{"beta", (getter)simple_type_get_beta', code)
        self.assertIn('{"delta", (getter)simple_type_get_delta', code)
        self.assertIn('{NULL}', code)  # Sentinel

    def test_type_object_generated(self):
        """Test that PyTypeObject definition is generated."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        self.assertIn('static PyTypeObject simple_typeType', code)
        self.assertIn('PyVarObject_HEAD_INIT(NULL, 0)', code)
        self.assertIn('.tp_name = "test_module.simple_type"', code)
        self.assertIn('.tp_basicsize = sizeof(Pysimple_type)', code)
        self.assertIn('.tp_dealloc = (destructor)simple_type_dealloc', code)
        self.assertIn('.tp_getset = simple_type_getsetters', code)
        self.assertIn('.tp_new = simple_type_new', code)

    def test_type_registered_in_module(self):
        """Test that type is registered in module init."""
        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        self.assertIn('PyType_Ready(&simple_typeType)', code)
        self.assertIn('PyModule_AddObject(module, "simple_type"', code)
        self.assertIn('&simple_typeType', code)

    def test_derived_type_element(self):
        """Test nested derived type element handling."""
        # Add a nested derived type element
        nested = ft.Element('nested', filename='test.f90')
        nested.type = 'type(other_type)'
        nested.attributes = []
        self.dtype.elements.append(nested)

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Nested type elements should have proper extern declarations
        self.assertIn('simple_type_get_nested', code)
        self.assertIn('/* Nested derived type element getter', code)
        self.assertIn('/* TODO: Create other_type instance', code)

    def test_array_element(self):
        """Test array element handling."""
        # Add an array element
        arr = ft.Element('arr', filename='test.f90')
        arr.type = 'real(8)'
        arr.attributes = ['dimension(10)']
        self.dtype.elements.append(arr)

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Array elements should have proper extern declarations
        self.assertIn('simple_type_get_arr', code)
        self.assertIn('/* Array element getter', code)
        self.assertIn('/* TODO: Implement full array retrieval', code)

    def test_type_bound_procedure(self):
        """Test type-bound procedure generation."""
        # Add a type-bound procedure
        method = ft.Subroutine('compute', filename='test.f90')
        method.arguments = []
        self.dtype.procedures.append(method)

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check method table
        self.assertIn('static PyMethodDef simple_type_methods[]', code)
        self.assertIn('{"compute", (PyCFunction)simple_type_compute', code)

        # Check method wrapper - now generates full implementation
        self.assertIn('static PyObject* simple_type_compute', code)
        self.assertIn('self->fortran_ptr', code)
        self.assertIn('Fortran type not initialized', code)

    def test_type_bound_procedure_with_arguments(self):
        """Test type-bound procedure with arguments."""
        # Add a method with arguments
        method = ft.Subroutine('process', filename='test.f90')

        arg1 = ft.Argument('x', filename='test.f90')
        arg1.type = 'integer(4)'
        arg1.attributes = ['intent(in)']

        arg2 = ft.Argument('y', filename='test.f90')
        arg2.type = 'real(8)'
        arg2.attributes = ['intent(out)']

        method.arguments = [arg1, arg2]
        self.dtype.procedures.append(method)

        gen = CWrapperGenerator(self.root, 'test_module')
        code = gen.generate()

        # Check method with argument handling
        self.assertIn('static PyObject* simple_type_process', code)
        self.assertIn('PyArg_ParseTuple', code)  # Argument parsing
        self.assertIn('self->fortran_ptr', code)  # Self pointer passed to Fortran


if __name__ == '__main__':
    unittest.main()
