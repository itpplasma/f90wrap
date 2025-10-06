"""Direct-C C code generator for f90wrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from f90wrap import codegen as cg
from f90wrap import fortran as ft
from f90wrap.directc import InteropInfo, ProcedureKey
from f90wrap.transform import shorten_long_name
from f90wrap.numpy_utils import (
    build_arg_format,
    c_type_from_fortran,
    numpy_type_from_fortran,
    parse_arg_format,
)


@dataclass
class DirectCGenerator(cg.CodeGenerator):
    """Generate C extension module code calling f90wrap helpers."""

    root: ft.Root
    interop_info: Dict[ProcedureKey, InteropInfo]
    kind_map: Dict[str, Dict[str, str]]
    prefix: str = "f90wrap_"

    def __post_init__(self):
        """Initialize CodeGenerator parent after dataclass init."""
        cg.CodeGenerator.__init__(self, indent="    ", max_length=120,
                                   continuation="\\", comment="//")

    def generate_module(
        self,
        mod_name: str,
        procedures: Optional[Iterable[ft.Procedure]] = None,
    ) -> str:
        """Generate complete _module.c file content."""

        # Reset code buffer
        self.code = []
        self._write_headers()

        selected = self._collect_procedures(mod_name, procedures)
        if not selected:
            return ""

        self._write_external_declarations(selected)

        # Generate wrapper functions
        for proc in selected:
            self._write_wrapper_function(proc, mod_name)

        # Module method table and init
        self._write_method_table(selected, mod_name)
        self._write_module_init(mod_name)

        return str(self)

    def _collect_procedures(
        self,
        mod_name: str,
        procedures: Optional[Iterable[ft.Procedure]] = None,
    ) -> List[ft.Procedure]:
        """Collect procedures that require helper wrappers for a module."""

        if procedures is None:
            proc_list: List[ft.Procedure] = []
            for module in self.root.modules:
                if module.name == mod_name:
                    proc_list.extend(module.procedures)
        else:
            proc_list = list(procedures)

        selected: List[ft.Procedure] = []
        for proc in proc_list:
            key = ProcedureKey(
                proc.mod_name,
                getattr(proc, "type_name", None),
                proc.name,
            )
            info = self.interop_info.get(key)
            if info:
                selected.append(proc)

        return selected

    def _write_headers(self) -> None:
        """Write standard C headers and Python/NumPy includes."""

        self.write("#include <Python.h>")
        self.write("#include <stdbool.h>")
        self.write("#include <stdlib.h>")
        self.write("#include <string.h>")
        self.write("#include <complex.h>")
        self.write("")
        self.write("#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION")
        self.write("#include <numpy/arrayobject.h>")
        self.write("")
        self.write("#define F90WRAP_F_SYMBOL(name) name##_")
        self.write("")

    def _write_external_declarations(self, procedures: Iterable[ft.Procedure]) -> None:
        """Write extern declarations for f90wrap helper functions."""

        self.write("/* External f90wrap helper functions */")
        for proc in procedures:
            self._write_helper_declaration(proc)
        self.write("")

    def _helper_name(self, proc: ft.Procedure) -> str:
        """Return the bare f90wrap helper name for a procedure."""

        parts: List[str] = [self.prefix]
        if proc.mod_name:
            parts.append(f"{proc.mod_name}__")
        type_name = getattr(proc, "type_name", None)
        if type_name:
            parts.append(f"{type_name}__")
        parts.append(proc.name)
        return shorten_long_name("".join(parts))

    def _helper_symbol(self, proc: ft.Procedure) -> str:
        """Return helper name with C macro for symbol mangling."""

        return f"F90WRAP_F_SYMBOL({self._helper_name(proc)})"

    def _wrapper_name(self, module_label: str, proc: ft.Procedure) -> str:
        """Return a unique wrapper function name for a procedure."""

        components: List[str] = [module_label]
        type_name = getattr(proc, "type_name", None)
        if type_name:
            components.append(type_name)
        if proc.mod_name and proc.mod_name != module_label:
            components.append(proc.mod_name)
        components.append(proc.name)
        safe_components = [comp for comp in components if comp]
        return "wrap_" + "_".join(safe_components)

    def _write_helper_declaration(self, proc: ft.Procedure) -> None:
        """Write extern declaration for a f90wrap helper function."""

        # Build parameter list
        params = []
        for arg in proc.arguments:
            if self._is_hidden_argument(arg):
                params.append("int* " + arg.name)
            elif self._is_array(arg):
                c_type = c_type_from_fortran(arg.type, self.kind_map)
                params.append(f"{c_type}* {arg.name}")
            elif arg.type.lower().startswith("character"):
                params.append(f"char* {arg.name}")
                params.append(f"int {arg.name}_len")
            else:
                c_type = c_type_from_fortran(arg.type, self.kind_map)
                params.append(f"{c_type}* {arg.name}")

        # Add return value for functions
        if isinstance(proc, ft.Function):
            c_type = c_type_from_fortran(proc.ret_val.type, self.kind_map)
            params.insert(0, f"{c_type}* result")

        helper_symbol = self._helper_symbol(proc)
        if params:
            self.write(f"extern void {helper_symbol}({', '.join(params)});")
        else:
            self.write(f"extern void {helper_symbol}(void);")

    def _write_wrapper_function(self, proc: ft.Procedure, mod_name: str) -> None:
        """Write Python C API wrapper function for a procedure."""

        wrapper_name = self._wrapper_name(mod_name, proc)

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()

        # Parse Python arguments
        self._write_arg_parsing(proc)

        # Prepare arguments for helper call
        self._write_arg_preparation(proc)

        # Call the helper function
        self._write_helper_call(proc)

        # Build return value
        self._write_return_value(proc)

        self.dedent()
        self.write("}")
        self.write("")

    def _write_arg_parsing(self, proc: ft.Procedure) -> None:
        """Generate PyArg_ParseTuple code for procedure arguments."""

        hidden_args = [arg for arg in proc.arguments if self._is_hidden_argument(arg)]
        for hidden in hidden_args:
            self.write(f"int {hidden.name}_val = 0;")

        if not proc.arguments:
            return

        # Build format string and variable list
        format_parts = []
        parse_vars = []
        kw_names = []

        for arg in proc.arguments:
            if self._is_hidden_argument(arg):
                continue
            if self._is_array(arg):
                format_parts.append("O")  # NumPy array as PyObject
                self.write(f"PyObject* py_{arg.name};")
                parse_vars.append(f"&py_{arg.name}")
                kw_names.append(f'"{arg.name}"')
            elif arg.type.lower().startswith("character"):
                format_parts.append("s")
                self.write(f"const char* {arg.name}_str;")
                parse_vars.append(f"&{arg.name}_str")
                kw_names.append(f'"{arg.name}"')
            else:
                fmt = parse_arg_format(arg.type)
                format_parts.append(fmt)
                c_type = c_type_from_fortran(arg.type, self.kind_map)
                self.write(f"{c_type} {arg.name}_val;")
                parse_vars.append(f"&{arg.name}_val")
                kw_names.append(f'"{arg.name}"')

        format_str = "".join(format_parts)
        kwlist = ", ".join(kw_names) if kw_names else ""
        self.write(f"static char *kwlist[] = {{{kwlist}{', ' if kwlist else ''}NULL}};")
        self.write("")
        self.write(
            f'if (!PyArg_ParseTupleAndKeywords(args, kwargs, "{format_str}", kwlist, '
            f"{', '.join(parse_vars)})) {{"
        )
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("")

    def _write_arg_preparation(self, proc: ft.Procedure) -> None:
        """Prepare arguments for helper function call."""

        for arg in proc.arguments:
            if self._is_array(arg):
                self._write_array_preparation(arg)
            elif arg.type.lower().startswith("character"):
                # Copy string to buffer
                self.write(f"int {arg.name}_len = strlen({arg.name}_str);")
                self.write(f"char* {arg.name} = (char*)malloc({arg.name}_len + 1);")
                self.write(f"strcpy({arg.name}, {arg.name}_str);")
            # Scalars already prepared

    def _write_array_preparation(self, arg: ft.Argument) -> None:
        """Extract array data from NumPy array."""

        numpy_type = numpy_type_from_fortran(arg.type, self.kind_map)
        c_type = c_type_from_fortran(arg.type, self.kind_map)

        self.write(f"/* Extract {arg.name} array data */")
        self.write(f"if (!PyArray_Check(py_{arg.name})) {{")
        self.indent()
        self.write(f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a NumPy array");')
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        # Convert to Fortran-contiguous if needed
        self.write(f"PyArrayObject* {arg.name}_arr = (PyArrayObject*)PyArray_FROM_OTF(")
        self.write(f"    py_{arg.name}, {numpy_type}, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);")
        self.write(f"if ({arg.name}_arr == NULL) {{")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"{c_type}* {arg.name} = ({c_type}*)PyArray_DATA({arg.name}_arr);")

        # Get dimensions if needed
        dims = self._extract_dimensions(arg)
        if dims:
            for i in range(len(dims)):
                self.write(f"int n{i}_{arg.name} = (int)PyArray_DIM({arg.name}_arr, {i});")
            for i, dim in enumerate(dims):
                dim_name = dim.strip()
                if dim_name and dim_name.startswith("f90wrap_"):
                    self.write(f"{dim_name}_val = n{i}_{arg.name};")

        self.write("")

    def _write_helper_call(self, proc: ft.Procedure) -> None:
        """Generate the call to the f90wrap helper function."""

        call_args = []
        helper_symbol = self._helper_symbol(proc)

        # Add result parameter for functions
        if isinstance(proc, ft.Function):
            c_type = c_type_from_fortran(proc.ret_val.type, self.kind_map)
            if self._is_array(proc.ret_val):
                self.write(f"{c_type}* result;")
            else:
                self.write(f"{c_type} result;")
            call_args.append("&result")

        # Add regular arguments
        for arg in proc.arguments:
            if self._is_hidden_argument(arg):
                call_args.append(f"&{arg.name}_val")
            elif self._is_array(arg):
                call_args.append(arg.name)
            elif arg.type.lower().startswith("character"):
                call_args.append(arg.name)
                call_args.append(f"&{arg.name}_len")
            else:
                call_args.append(f"&{arg.name}_val")

        self.write(f"/* Call f90wrap helper */")
        if call_args:
            self.write(f"{helper_symbol}({', '.join(call_args)});")
        else:
            self.write(f"{helper_symbol}();")
        self.write("")

    def _write_return_value(self, proc: ft.Procedure) -> None:
        """Build and return the Python return value."""

        # Clean up allocated memory
        for arg in proc.arguments:
            if arg.type.lower().startswith("character"):
                self.write(f"free({arg.name});")
            elif self._is_array(arg):
                self.write(f"Py_DECREF({arg.name}_arr);")

        if isinstance(proc, ft.Function):
            ret_type = proc.ret_val.type.lower()
            if self._is_array(proc.ret_val):
                self._write_array_return(proc.ret_val, "result")
            elif ret_type.startswith("logical"):
                self.write("return PyBool_FromLong(result);")
            else:
                fmt = build_arg_format(proc.ret_val.type)
                self.write(f'return Py_BuildValue("{fmt}", result);')
        else:
            self.write("Py_RETURN_NONE;")

    def _write_array_return(self, ret_val: ft.Argument, var_name: str) -> None:
        """Create NumPy array from returned Fortran array."""

        numpy_type = numpy_type_from_fortran(ret_val.type, self.kind_map)
        dims = self._extract_dimensions(ret_val)
        ndim = len(dims) if dims else 1

        self.write(f"/* Create NumPy array from result */")
        self.write(f"npy_intp result_dims[{ndim}];")
        for i, dim in enumerate(dims or [1]):
            self.write(f"result_dims[{i}] = {dim};")

        self.write(f"PyObject* result_arr = PyArray_New(&PyArray_Type, {ndim}, result_dims,")
        self.write(f"    {numpy_type}, NULL, (void*){var_name},")
        self.write(f"    0, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA, NULL);")
        self.write("return result_arr;")

    def _write_method_table(self, procedures: List[ft.Procedure], mod_name: str) -> None:
        """Write the module method table."""

        self.write(f"/* Method table for {mod_name} module */")
        self.write(f"static PyMethodDef {mod_name}_methods[] = {{")
        self.indent()

        for proc in procedures:
            wrapper_name = self._wrapper_name(mod_name, proc)
            method_name = self._helper_name(proc)
            docstring = proc.doc[0] if proc.doc else f"Wrapper for {proc.name}"
            # Escape any quotes in docstring
            docstring = docstring.replace('"', '\\"')
            self.write(
                f'{{"{method_name}", (PyCFunction){wrapper_name}, '
                f"METH_VARARGS | METH_KEYWORDS, \"{docstring}\"}},"
            )

        self.write("{NULL, NULL, 0, NULL}  /* Sentinel */")
        self.dedent()
        self.write("};")
        self.write("")

    def _write_module_init(self, mod_name: str) -> None:
        """Write the module initialization function."""

        py_mod_name = mod_name.lstrip("_") or mod_name
        self.write(f"/* Module definition */")
        self.write(f"static struct PyModuleDef {mod_name}module = {{")
        self.indent()
        self.write("PyModuleDef_HEAD_INIT,")
        self.write(f'"{py_mod_name}",')
        self.write(f'"Direct-C wrapper for {mod_name} module",')
        self.write("-1,")
        self.write(f"{mod_name}_methods")
        self.dedent()
        self.write("};")
        self.write("")

        self.write(f"/* Module initialization */")
        self.write(f"PyMODINIT_FUNC PyInit_{py_mod_name}(void)")
        self.write("{")
        self.indent()
        self.write("import_array();  /* Initialize NumPy */")
        self.write(f"return PyModule_Create(&{mod_name}module);")
        self.dedent()
        self.write("}")

    def _is_array(self, arg: ft.Argument) -> bool:
        """Check if argument is an array."""
        return any("dimension" in attr for attr in arg.attributes)

    def _is_hidden_argument(self, arg: ft.Argument) -> bool:
        """Return True if argument is hidden from the Python API."""

        return any(attr.startswith("intent(hide)") for attr in arg.attributes)

    def _extract_dimensions(self, arg: ft.Argument) -> List[str]:
        """Extract array dimensions from argument attributes."""

        for attr in arg.attributes:
            if attr.startswith("dimension("):
                dim_str = attr[len("dimension("):-1]
                return [d.strip() for d in dim_str.split(",")]
        return []
