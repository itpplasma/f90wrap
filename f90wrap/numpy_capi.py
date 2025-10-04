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
NumPy C API integration for f90wrap.

Handles NumPy array creation, conversion, and memory management
for interfacing with Fortran arrays.
"""

import logging
from typing import List, Optional
from f90wrap import fortran as ft
from f90wrap.cwrapgen import FortranCTypeMap, CCodeGenerator

log = logging.getLogger(__name__)


class NumpyArrayHandler:
    """
    Handle NumPy C API array operations.

    Provides methods to generate C code for:
    - Creating NumPy arrays from Fortran data
    - Extracting Fortran data from NumPy arrays
    - Handling memory ownership and reference counting
    - Converting between column-major (Fortran) and row-major (NumPy) layouts
    """

    def __init__(self, type_map: FortranCTypeMap):
        """
        Initialize NumPy array handler.

        Parameters
        ----------
        type_map : FortranCTypeMap
            Type mapping system
        """
        self.type_map = type_map

    def generate_array_from_fortran(self, arg: ft.Argument, code_gen: CCodeGenerator,
                                     var_name: str = None) -> str:
        """
        Generate C code to create NumPy array from Fortran pointer.

        Parameters
        ----------
        arg : fortran.Argument
            Fortran array argument
        code_gen : CCodeGenerator
            Code generator instance
        var_name : str, optional
            Variable name for the array (default: arg.name)

        Returns
        -------
        str
            Name of the generated PyObject* variable
        """
        if var_name is None:
            var_name = arg.name

        py_var = f"py_{var_name}"
        numpy_type = self.type_map.fortran_to_numpy_type(arg.type)

        # Get array dimensions
        dims = self._extract_dimensions(arg)
        ndim = len(dims)

        code_gen.write(f'/* Create NumPy array from Fortran array {var_name} */')
        code_gen.write(f'npy_intp dims_{var_name}[{ndim}];')

        # Set dimensions
        for i, dim in enumerate(dims):
            if isinstance(dim, str):
                code_gen.write(f'dims_{var_name}[{i}] = {dim};')
            else:
                code_gen.write(f'dims_{var_name}[{i}] = {dim};')

        # Create array with Fortran memory ordering
        code_gen.write(f'PyObject* {py_var} = PyArray_New(&PyArray_Type, {ndim}, dims_{var_name},')
        code_gen.write(f'                      {numpy_type}, NULL, (void*){var_name},')
        code_gen.write(f'                      0, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_WRITEABLE, NULL);')

        code_gen.write(f'if ({py_var} == NULL) {{')
        code_gen.indent()
        code_gen.write('PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

        return py_var

    def generate_fortran_from_array(self, arg: ft.Argument, code_gen: CCodeGenerator,
                                     py_var: str, c_var: str) -> None:
        """
        Generate C code to extract Fortran data from NumPy array.

        Parameters
        ----------
        arg : fortran.Argument
            Fortran array argument
        code_gen : CCodeGenerator
            Code generator instance
        py_var : str
            Name of Python object variable (PyObject*)
        c_var : str
            Name of C variable to receive pointer
        """
        numpy_type = self.type_map.fortran_to_numpy_type(arg.type)
        c_type = self.type_map.fortran_to_c_type(arg.type)

        dims = self._extract_dimensions(arg)
        ndim = len(dims)

        code_gen.write(f'/* Extract Fortran array from NumPy {py_var} */')

        # Type check
        code_gen.write(f'if (!PyArray_Check({py_var})) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_SetString(PyExc_TypeError, "Expected NumPy array for {arg.name}");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

        # Dimension check
        code_gen.write(f'if (PyArray_NDIM((PyArrayObject*){py_var}) != {ndim}) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_Format(PyExc_ValueError, "Array {arg.name} must have {ndim} dimensions, got %d",')
        code_gen.write(f'             PyArray_NDIM((PyArrayObject*){py_var}));')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

        # Type check
        code_gen.write(f'if (PyArray_TYPE((PyArrayObject*){py_var}) != {numpy_type}) {{')
        code_gen.indent()
        code_gen.write(f'PyErr_SetString(PyExc_TypeError, "Array {arg.name} has wrong dtype");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

        # Handle contiguity - convert to Fortran order if needed
        code_gen.write(f'PyArrayObject *{c_var}_array = (PyArrayObject*){py_var};')
        code_gen.write(f'if (!PyArray_IS_F_CONTIGUOUS({c_var}_array)) {{')
        code_gen.indent()
        code_gen.write(f'{c_var}_array = (PyArrayObject*)PyArray_FromArray(')
        code_gen.write(f'    {c_var}_array, NULL, NPY_ARRAY_F_CONTIGUOUS);')
        code_gen.write(f'if ({c_var}_array == NULL) {{')
        code_gen.indent()
        code_gen.write('PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

        # Extract data pointer
        code_gen.write(f'{c_type}* {c_var} = ({c_type}*)PyArray_DATA({c_var}_array);')
        code_gen.write('')

    def generate_dimension_checks(self, arg: ft.Argument, code_gen: CCodeGenerator,
                                   py_var: str) -> None:
        """
        Generate dimension checking code for array arguments.

        Parameters
        ----------
        arg : fortran.Argument
            Array argument
        code_gen : CCodeGenerator
            Code generator
        py_var : str
            Python object variable name
        """
        dims = self._extract_dimensions(arg)

        for i, dim in enumerate(dims):
            if isinstance(dim, int):
                # Fixed dimension - check it
                code_gen.write(f'if (PyArray_DIM((PyArrayObject*){py_var}, {i}) != {dim}) {{')
                code_gen.indent()
                code_gen.write(f'PyErr_Format(PyExc_ValueError,')
                code_gen.write(f'             "Dimension {i} of {arg.name} must be {dim}, got %ld",')
                code_gen.write(f'             (long)PyArray_DIM((PyArrayObject*){py_var}, {i}));')
                code_gen.write('return NULL;')
                code_gen.dedent()
                code_gen.write('}')

    def generate_array_copy(self, arg: ft.Argument, code_gen: CCodeGenerator,
                            src_var: str, dst_var: str, intent: str) -> None:
        """
        Generate code to copy array data (for intent handling).

        Parameters
        ----------
        arg : fortran.Argument
            Array argument
        code_gen : CCodeGenerator
            Code generator
        src_var : str
            Source array variable
        dst_var : str
            Destination array variable
        intent : str
            Argument intent ('in', 'out', 'inout')
        """
        c_type = self.type_map.fortran_to_c_type(arg.type)
        dims = self._extract_dimensions(arg)

        # Calculate total size
        size_expr = ' * '.join(str(d) if isinstance(d, int) else d for d in dims)

        if intent in ('in', 'inout'):
            code_gen.write(f'/* Copy input array data */')
            code_gen.write(f'memcpy({dst_var}, {src_var}, ({size_expr}) * sizeof({c_type}));')
            code_gen.write('')

        if intent in ('out', 'inout'):
            code_gen.write(f'/* Copy output array data */')
            code_gen.write(f'memcpy({src_var}, {dst_var}, ({size_expr}) * sizeof({c_type}));')
            code_gen.write('')

    def generate_array_alloc(self, arg: ft.Argument, code_gen: CCodeGenerator,
                            var_name: str) -> str:
        """
        Generate code to allocate temporary array storage.

        Parameters
        ----------
        arg : fortran.Argument
            Array argument
        code_gen : CCodeGenerator
            Code generator
        var_name : str
            Variable name for the array

        Returns
        -------
        str
            Name of allocated variable
        """
        c_type = self.type_map.fortran_to_c_type(arg.type)
        dims = self._extract_dimensions(arg)

        size_expr = ' * '.join(str(d) if isinstance(d, int) else d for d in dims)

        code_gen.write(f'/* Allocate temporary array for {var_name} */')
        code_gen.write(f'{c_type}* {var_name} = ({c_type}*)malloc(({size_expr}) * sizeof({c_type}));')
        code_gen.write(f'if ({var_name} == NULL) {{')
        code_gen.indent()
        code_gen.write('PyErr_SetString(PyExc_MemoryError, "Failed to allocate array");')
        code_gen.write('return NULL;')
        code_gen.dedent()
        code_gen.write('}')
        code_gen.write('')

        return var_name

    def generate_array_free(self, code_gen: CCodeGenerator, var_name: str) -> None:
        """
        Generate code to free temporary array storage.

        Parameters
        ----------
        code_gen : CCodeGenerator
            Code generator
        var_name : str
            Variable name to free
        """
        code_gen.write(f'free({var_name});')

    def _extract_dimensions(self, arg: ft.Argument) -> List:
        """
        Extract dimension information from argument.

        Parameters
        ----------
        arg : fortran.Argument
            Array argument

        Returns
        -------
        list
            List of dimensions (integers or variable names)
        """
        dims = []

        # Look for dimension attribute
        for attr in arg.attributes:
            if attr.startswith('dimension'):
                # Extract dimension spec: dimension(n1, n2, ...)
                dim_spec = attr[attr.index('(')+1:attr.rindex(')')]
                dim_parts = [d.strip() for d in dim_spec.split(',')]

                for part in dim_parts:
                    if ':' in part:
                        # Assumed shape: 1:n or :
                        if part == ':':
                            dims.append('shape_unknown')
                        else:
                            bounds = part.split(':')
                            if bounds[1]:
                                # Try to parse as integer
                                try:
                                    dims.append(int(bounds[1]))
                                except ValueError:
                                    dims.append(bounds[1])
                            else:
                                dims.append('shape_unknown')
                    else:
                        # Fixed size
                        try:
                            dims.append(int(part))
                        except ValueError:
                            # Variable dimension
                            dims.append(part)
                break

        if not dims:
            # No dimension attribute found - treat as scalar
            return []

        return dims

    def generate_stride_conversion(self, arg: ft.Argument, code_gen: CCodeGenerator,
                                    fortran_var: str, numpy_var: str) -> None:
        """
        Generate code to handle stride/layout conversion between Fortran and NumPy.

        Parameters
        ----------
        arg : fortran.Argument
            Array argument
        code_gen : CCodeGenerator
            Code generator
        fortran_var : str
            Fortran array variable (column-major)
        numpy_var : str
            NumPy array variable (row-major)
        """
        dims = self._extract_dimensions(arg)
        ndim = len(dims)

        if ndim <= 1:
            # No conversion needed for 1D arrays
            return

        c_type = self.type_map.fortran_to_c_type(arg.type)

        code_gen.write(f'/* Convert array layout between Fortran (column-major) and NumPy (row-major) */')

        # Generate nested loops for transpose
        # For simplicity, use PyArray transpose when available
        code_gen.write(f'/* Note: Using F_CONTIGUOUS flag handles layout automatically */')
        code_gen.write('')
