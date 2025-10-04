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
        self.assertIn('void* fortran_handle', code)

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


if __name__ == '__main__':
    unittest.main()
