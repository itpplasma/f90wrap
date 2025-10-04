#  f90wrap: F90 to Python interface generator with derived type support
#
#  Copyright James Kermode 2011-2018
#
#  This file is part of f90wrap
#  For the latest version see github.com/jameskermode/f90wrap
#
#  f90wrap is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  f90wrap is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with f90wrap. If not, see <http://www.gnu.org/licenses/>.
#
#  If you would like to license the source code under different terms,
#  please contact James Kermode, james.kermode@gmail.com

"""
Error handling and exception support for C wrapper generation.

Provides Python exception raising from C and Fortran error propagation
using setjmp/longjmp for f90wrap_abort support.
"""

import logging
from f90wrap.cwrapgen import CCodeGenerator

log = logging.getLogger(__name__)


class CErrorHandler:
    """
    Generate error handling code for C wrappers.

    Handles:
    - Python exception checking and propagation
    - Fortran abort mechanism (f90wrap_abort)
    - Resource cleanup on error paths
    - Memory leak prevention
    """

    @staticmethod
    def generate_exception_check(code_gen: CCodeGenerator,
                                 operation: str = "operation") -> None:
        """
        Generate code to check if a Python exception occurred.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        operation : str
            Description of the operation for error messages
        """
        code_gen.write(f'/* Check for Python exception after {operation} */')
        code_gen.write('if (PyErr_Occurred()) {')
        code_gen.indent()
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

    @staticmethod
    def generate_abort_handler_header(code_gen: CCodeGenerator) -> None:
        """
        Generate header code for f90wrap_abort mechanism.

        Uses setjmp/longjmp to catch Fortran abort calls.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        """
        code_gen.write_raw('''
/* f90wrap_abort error handling mechanism */
#include <setjmp.h>

static jmp_buf f90wrap_abort_jmp_buf;
static int f90wrap_abort_active = 0;
static char f90wrap_abort_message[1024] = "";

/* Fortran-callable abort function */
extern void f90wrap_abort_(const char *msg, int msg_len);

void f90wrap_abort_(const char *msg, int msg_len) {
    /* Copy error message */
    int copy_len = msg_len < 1023 ? msg_len : 1023;
    strncpy(f90wrap_abort_message, msg, copy_len);
    f90wrap_abort_message[copy_len] = '\\0';

    /* Jump back to wrapper if active */
    if (f90wrap_abort_active) {
        longjmp(f90wrap_abort_jmp_buf, 1);
    } else {
        /* No handler active - print to stderr and exit */
        fprintf(stderr, "FORTRAN ERROR: %s\\n", f90wrap_abort_message);
        exit(1);
    }
}

''')

    @staticmethod
    def generate_abort_wrapper_start(code_gen: CCodeGenerator,
                                      func_name: str) -> None:
        """
        Generate start of abort-protected wrapper function.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        func_name : str
            Name of the wrapped function
        """
        code_gen.write(f'/* Set up f90wrap_abort handler for {func_name} */')
        code_gen.write('f90wrap_abort_active = 1;')
        code_gen.write('')
        code_gen.write('if (setjmp(f90wrap_abort_jmp_buf)) {')
        code_gen.indent()
        code_gen.write('/* Fortran abort was called */')
        code_gen.write('f90wrap_abort_active = 0;')
        code_gen.write('PyErr_SetString(PyExc_RuntimeError, f90wrap_abort_message);')
        code_gen.write('/* Clean up resources here */')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

    @staticmethod
    def generate_abort_wrapper_end(code_gen: CCodeGenerator) -> None:
        """
        Generate end of abort-protected wrapper function.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        """
        code_gen.write('')
        code_gen.write('/* Deactivate abort handler */')
        code_gen.write('f90wrap_abort_active = 0;')
        code_gen.write('')

    @staticmethod
    def generate_cleanup_label(code_gen: CCodeGenerator,
                               cleanup_vars: list) -> None:
        """
        Generate cleanup code with goto label for error handling.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        cleanup_vars : list of str
            Variables to clean up (free, decref, etc.)
        """
        code_gen.write('cleanup:')
        code_gen.indent()

        for var in cleanup_vars:
            if var.startswith('py_'):
                # Python object - decref
                code_gen.write(f'Py_XDECREF({var});')
            else:
                # C pointer - free
                code_gen.write(f'if ({var}) free({var});')

        code_gen.dedent()
        code_gen.write('')

    @staticmethod
    def generate_null_check(code_gen: CCodeGenerator,
                           var: str,
                           error_msg: str,
                           cleanup_label: str = None) -> None:
        """
        Generate NULL pointer check with error handling.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        var : str
            Variable to check
        error_msg : str
            Error message if NULL
        cleanup_label : str, optional
            Cleanup label to jump to on error
        """
        code_gen.write(f'if ({var} == NULL) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_SetString(PyExc_RuntimeError, "{error_msg}");')

        if cleanup_label:
            code_gen.write(f'goto {cleanup_label};')
        else:
            code_gen.write('return NULL;')

        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

    @staticmethod
    def generate_array_check(code_gen: CCodeGenerator,
                            var: str,
                            expected_ndim: int,
                            expected_dtype: str,
                            arg_name: str) -> None:
        """
        Generate comprehensive array validation checks.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        var : str
            Array variable name
        expected_ndim : int
            Expected number of dimensions
        expected_dtype : str
            Expected NumPy dtype code
        arg_name : str
            Argument name for error messages
        """
        # Type check
        code_gen.write(f'/* Validate array argument: {arg_name} */')
        code_gen.write(f'if (!PyArray_Check({var})) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_Format(PyExc_TypeError,')
        code_gen.write(f'             "Argument \'{arg_name}\' must be a NumPy array");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

        # Dimension check
        code_gen.write(f'if (PyArray_NDIM((PyArrayObject*){var}) != {expected_ndim}) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_Format(PyExc_ValueError,')
        code_gen.write(f'             "Array \'{arg_name}\' must have {expected_ndim} dimensions, got %d",')
        code_gen.write(f'             PyArray_NDIM((PyArrayObject*){var}));')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

        # Type check
        code_gen.write(f'if (PyArray_TYPE((PyArrayObject*){var}) != {expected_dtype}) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_Format(PyExc_TypeError,')
        code_gen.write(f'             "Array \'{arg_name}\' has wrong dtype (expected {expected_dtype})");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

    @staticmethod
    def generate_bounds_check(code_gen: CCodeGenerator,
                              index_var: str,
                              size_var: str,
                              array_name: str) -> None:
        """
        Generate array bounds checking code.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        index_var : str
            Index variable to check
        size_var : str
            Size/length variable
        array_name : str
            Array name for error messages
        """
        code_gen.write(f'/* Bounds check for {array_name} */')
        code_gen.write(f'if ({index_var} < 0 || {index_var} >= {size_var}) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_Format(PyExc_IndexError,')
        code_gen.write(f'             "Index %ld out of bounds for array \'{array_name}\' (size %ld)",')
        code_gen.write(f'             (long){index_var}, (long){size_var});')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

    @staticmethod
    def generate_type_check(code_gen: CCodeGenerator,
                           var: str,
                           py_type: str,
                           arg_name: str) -> None:
        """
        Generate Python type checking code.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        var : str
            Variable to check
        py_type : str
            Expected Python type (PyLong_Check, PyFloat_Check, etc.)
        arg_name : str
            Argument name for error messages
        """
        code_gen.write(f'/* Type check for {arg_name} */')
        code_gen.write(f'if (!{py_type}({var})) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_Format(PyExc_TypeError,')
        code_gen.write(f'             "Argument \'{arg_name}\' has wrong type");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

    @staticmethod
    def generate_overflow_check(code_gen: CCodeGenerator,
                               value_var: str,
                               min_value: str,
                               max_value: str,
                               arg_name: str) -> None:
        """
        Generate numeric overflow checking code.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        value_var : str
            Value variable to check
        min_value : str
            Minimum allowed value
        max_value : str
            Maximum allowed value
        arg_name : str
            Argument name for error messages
        """
        code_gen.write(f'/* Overflow check for {arg_name} */')
        code_gen.write(f'if ({value_var} < {min_value} || {value_var} > {max_value}) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_Format(PyExc_OverflowError,')
        code_gen.write(f'             "Value for \'{arg_name}\' is out of range");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

    @staticmethod
    def generate_memory_error(code_gen: CCodeGenerator,
                             allocation_desc: str,
                             cleanup_label: str = None) -> None:
        """
        Generate memory allocation failure handling.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator instance
        allocation_desc : str
            Description of what failed to allocate
        cleanup_label : str, optional
            Cleanup label to jump to
        """
        code_gen.write(f'PyErr_Format(PyExc_MemoryError,')
        code_gen.write(f'             "Failed to allocate {allocation_desc}");')

        if cleanup_label:
            code_gen.write(f'goto {cleanup_label};')
        else:
            code_gen.write('return NULL;')

        code_gen.write('')
