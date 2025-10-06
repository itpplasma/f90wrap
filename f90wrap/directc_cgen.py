"""Direct-C C code generator for f90wrap."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Tuple

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
class ModuleHelper:
    """Metadata for module-level helper routines."""

    module: str
    name: str
    kind: str  # 'get', 'set', 'array', 'array_getitem', 'array_setitem', 'array_len', 'get_derived', 'set_derived'
    element: ft.Element
    is_type_member: bool = False


@dataclass
class DirectCGenerator(cg.CodeGenerator):
    """Generate C extension module code calling f90wrap helpers."""

    root: ft.Root
    interop_info: Dict[ProcedureKey, InteropInfo]
    kind_map: Dict[str, Dict[str, str]]
    prefix: str = "f90wrap_"
    handle_size: int = 4
    error_num_arg: Optional[str] = None
    error_msg_arg: Optional[str] = None
    callbacks: Optional[Iterable[str]] = None
    shape_hints: Optional[Dict[Tuple[str, Optional[str], str, str], List[str]]] = None

    def __post_init__(self):
        """Initialize CodeGenerator parent after dataclass init."""
        cg.CodeGenerator.__init__(self, indent="    ", max_length=120,
                                   continuation="\\", comment="//")
        if self.callbacks:
            seen = set()
            ordered: List[str] = []
            for name in self.callbacks:
                if not name:
                    continue
                normalised = name.strip()
                if not normalised:
                    continue
                key = normalised.lower()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(normalised)
            self.callbacks = ordered
        else:
            self.callbacks = []
        if self.shape_hints is None:
            self.shape_hints = {}

    def generate_module(
        self,
        mod_name: str,
        procedures: Optional[Iterable[ft.Procedure]] = None,
    ) -> str:
        """Generate complete _module.c file content."""

        # Reset code buffer
        self.code = []
        self._write_headers(mod_name)

        module_label = mod_name.lstrip("_") or mod_name

        selected = self._collect_procedures(module_label, procedures)
        module_helpers = self._collect_module_helpers(module_label)

        if not selected and not module_helpers:
            return ""

        self._write_external_declarations(selected, module_helpers)

        # Generate wrapper functions
        for proc in selected:
            self._write_wrapper_function(proc, mod_name)

        for helper in module_helpers:
            self._write_module_helper_wrapper(helper)

        # Module method table and init
        self._write_method_table(selected, module_helpers, mod_name)
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
        else:
            proc_list = list(procedures)

        target_module: Optional[ft.Module] = None
        for module in self.root.modules:
            if module.name == mod_name:
                target_module = module
                if procedures is None:
                    proc_list.extend(module.procedures)
                break

        if target_module is not None:
            for derived in getattr(target_module, "types", []):
                for proc in getattr(derived, "procedures", []):
                    if proc not in proc_list:
                        proc_list.append(proc)

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

    def _collect_module_helpers(self, mod_name: str) -> List[ModuleHelper]:
        """Collect helper metadata for module-level variables."""

        target_module: Optional[ft.Module] = None
        for module in self.root.modules:
            if module.name == mod_name:
                target_module = module
                break

        if target_module is None:
            return []

        helpers: List[ModuleHelper] = []
        for element in getattr(target_module, "elements", []):
            is_array = any(attr.startswith("dimension(") for attr in element.attributes)
            is_parameter = any(attr.startswith("parameter") for attr in element.attributes)
            element_type = element.type.strip().lower()
            is_derived = element_type.startswith("type(") or element_type.startswith("class(")

            if not is_array:
                if is_derived:
                    helpers.append(ModuleHelper(mod_name, element.name, "get_derived", element, False))
                    if not is_parameter:
                        helpers.append(ModuleHelper(mod_name, element.name, "set_derived", element, False))
                else:
                    helpers.append(ModuleHelper(mod_name, element.name, "get", element, False))
                    if not is_parameter:
                        helpers.append(ModuleHelper(mod_name, element.name, "set", element, False))
            elif is_derived:
                helpers.append(ModuleHelper(mod_name, element.name, "array_getitem", element))
                if not is_parameter:
                    helpers.append(ModuleHelper(mod_name, element.name, "array_setitem", element))
                helpers.append(ModuleHelper(mod_name, element.name, "array_len", element))
            else:
                helpers.append(ModuleHelper(mod_name, element.name, "array", element))

        for derived in getattr(target_module, "types", []):
            type_mod = derived.name
            for element in getattr(derived, "elements", []):
                is_array = any(attr.startswith("dimension(") for attr in element.attributes)
                is_parameter = any(attr.startswith("parameter") for attr in element.attributes)
                element_type = element.type.strip().lower()
                is_derived = element_type.startswith("type(") or element_type.startswith("class(")

                if not is_array:
                    if is_derived:
                        helpers.append(ModuleHelper(type_mod, element.name, "get_derived", element, True))
                        if not is_parameter:
                            helpers.append(ModuleHelper(type_mod, element.name, "set_derived", element, True))
                    else:
                        helpers.append(ModuleHelper(type_mod, element.name, "get", element, True))
                        if not is_parameter:
                            helpers.append(ModuleHelper(type_mod, element.name, "set", element, True))
                    continue

                if is_derived:
                    helpers.append(ModuleHelper(type_mod, element.name, "array_getitem", element))
                    if not is_parameter:
                        helpers.append(ModuleHelper(type_mod, element.name, "array_setitem", element))
                    helpers.append(ModuleHelper(type_mod, element.name, "array_len", element))
                else:
                    helpers.append(ModuleHelper(type_mod, element.name, "array", element))

        return helpers

    def _write_headers(self, module_name: str) -> None:
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
        self._write_abort_helpers()
        self._write_callback_trampolines(module_name)

    def _write_abort_helpers(self) -> None:
        """Emit minimal abort handler expected by f90wrap helpers."""

        self.write("void f90wrap_abort_(char *message, int len_message)")
        self.write("{")
        self.indent()
        self.write("if (message == NULL) {")
        self.indent()
        self.write("PyErr_SetString(PyExc_RuntimeError, \"f90wrap_abort called\");")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("while (len_message > 0 && message[len_message - 1] == ' ') {")
        self.indent()
        self.write("--len_message;")
        self.dedent()
        self.write("}")
        self.write("if (len_message <= 0) {")
        self.indent()
        self.write("PyErr_SetString(PyExc_RuntimeError, \"f90wrap_abort called\");")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyObject* unicode = PyUnicode_FromStringAndSize(message, len_message);")
        self.write("if (unicode == NULL) {")
        self.indent()
        self.write("PyErr_SetString(PyExc_RuntimeError, \"f90wrap_abort called\");")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyErr_SetObject(PyExc_RuntimeError, unicode);")
        self.write("Py_DECREF(unicode);")
        self.dedent()
        self.write("}")
        self.write("")
        self.write("void f90wrap_abort__(char *message, int len_message)")
        self.write("{")
        self.indent()
        self.write("f90wrap_abort_(message, len_message);")
        self.dedent()
        self.write("}")
        self.write("")

    def _write_callback_trampolines(self, module_name: str) -> None:
        """Emit helper and trampoline functions for registered callbacks."""

        if not self.callbacks:
            return

        self.write("/* Callback trampolines */")
        self.write(
            "static void f90wrap_invoke_callback(const char *callback_name, char *message, int len_message)"
        )
        self.write("{")
        self.indent()
        self.write("PyGILState_STATE gstate = PyGILState_Ensure();")
        self.write(f"PyObject* module = PyImport_AddModule(\"{module_name}\");")
        self.write("if (module == NULL) {")
        self.indent()
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyObject* callable = PyObject_GetAttrString(module, callback_name);")
        self.write("if (callable == NULL) {")
        self.indent()
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("if (callable == Py_None) {")
        self.indent()
        self.write("Py_DECREF(callable);")
        self.write(
            "PyErr_Format(PyExc_RuntimeError, \"cb: Callback %s not defined (as an argument or module attribute).\", callback_name);"
        )
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("int actual_len = len_message;")
        self.write("while (actual_len > 0 && message[actual_len - 1] == ' ') {")
        self.indent()
        self.write("--actual_len;")
        self.dedent()
        self.write("}")
        self.write("PyObject* arg_bytes = PyBytes_FromStringAndSize(message, actual_len);")
        self.write("if (arg_bytes == NULL) {")
        self.indent()
        self.write("Py_DECREF(callable);")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyObject* arg = PyUnicode_DecodeLatin1(PyBytes_AS_STRING(arg_bytes), actual_len, \"strict\");")
        self.write("Py_DECREF(arg_bytes);")
        self.write("if (arg == NULL) {")
        self.indent()
        self.write("Py_DECREF(callable);")
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("PyObject* result = PyObject_CallFunctionObjArgs(callable, arg, NULL);")
        self.write("Py_DECREF(arg);")
        self.write("Py_DECREF(callable);")
        self.write("if (result == NULL) {")
        self.indent()
        self.write("PyGILState_Release(gstate);")
        self.write("return;")
        self.dedent()
        self.write("}")
        self.write("Py_DECREF(result);")
        self.write("PyGILState_Release(gstate);")
        self.dedent()
        self.write("}")
        self.write("")

        for callback_name in self.callbacks:
            symbol = f"{callback_name.lower()}_"
            self.write(f"void {symbol}(char *message, int len_message)")
            self.write("{")
            self.indent()
            self.write(f"f90wrap_invoke_callback(\"{callback_name}\", message, len_message);")
            self.dedent()
            self.write("}")
            self.write("")

    def _write_external_declarations(
        self,
        procedures: Iterable[ft.Procedure],
        module_helpers: Iterable[ModuleHelper],
    ) -> None:
        """Write extern declarations for f90wrap helper functions."""

        self.write("/* External f90wrap helper functions */")
        for proc in procedures:
            self._write_helper_declaration(proc)
        for helper in module_helpers:
            self._write_module_helper_declaration(helper)
        self.write("")

    def _helper_name(self, proc: ft.Procedure) -> str:
        """Return the bare f90wrap helper name for a procedure."""

        parts: List[str] = [self.prefix]
        if proc.mod_name:
            parts.append(f"{proc.mod_name}__")
        parts.append(proc.name)
        return shorten_long_name("".join(parts))

    def _helper_symbol(self, proc: ft.Procedure) -> str:
        """Return helper name with C macro for symbol mangling."""

        return f"F90WRAP_F_SYMBOL({self._helper_name(proc)})"

    def _module_helper_name(self, helper: ModuleHelper) -> str:
        """Return helper name for module-level helper routines."""

        base = f"{self.prefix}{helper.module}__"
        if helper.kind in {"get", "set", "get_derived", "set_derived"}:
            kind_label = "get" if helper.kind.startswith("get") else "set"
            base += f"{kind_label}__{helper.name}"
        elif helper.kind == "array":
            base += f"array__{helper.name}"
        elif helper.kind == "array_getitem":
            base += f"array_getitem__{helper.name}"
        elif helper.kind == "array_setitem":
            base += f"array_setitem__{helper.name}"
        elif helper.kind == "array_len":
            base += f"array_len__{helper.name}"
        return shorten_long_name(base)

    def _module_helper_symbol(self, helper: ModuleHelper) -> str:
        """Return C symbol macro for module helper."""

        return f"F90WRAP_F_SYMBOL({self._module_helper_name(helper)})"

    @staticmethod
    def _character_length_expr(type_spec: str) -> Optional[str]:
        """Extract a fixed character length literal when available."""

        text = type_spec.strip().lower()
        if not text.startswith("character"):
            return None

        if "(" not in text or ")" not in text:
            return None

        inside = text[text.find("(") + 1 : text.rfind(")")].strip()
        if inside.startswith("len="):
            inside = inside[4:].strip()
        if inside == "*" or not inside:
            return None
        inside = inside.replace(" ", "")
        if re.fullmatch(r"\d+", inside):
            return inside
        return None

    @staticmethod
    def _static_array_shape(arg: ft.Argument) -> Optional[Tuple[int, ...]]:
        """Return constant shape for dimension(...) attribute when available."""

        for attr in arg.attributes:
            if attr.startswith("dimension(") and attr.endswith(")"):
                entries = [entry.strip() for entry in attr[len("dimension("):-1].split(",")]
                dims: List[int] = []
                for entry in entries:
                    if not entry or not entry.isdigit():
                        return None
                    dims.append(int(entry))
                return tuple(dims)
        return None

    def _module_helper_wrapper_name(self, helper: ModuleHelper) -> str:
        """Return wrapper function name for a module helper."""

        return f"wrap_{helper.module}_{helper.kind}_{helper.name}"

    def _write_module_helper_wrapper(self, helper: ModuleHelper) -> None:
        """Emit specialised wrappers for module get/set/array helpers."""

        if helper.kind == "array_getitem":
            self._write_module_array_getitem_wrapper(helper)
            return
        if helper.kind == "array_setitem":
            self._write_module_array_setitem_wrapper(helper)
            return
        if helper.kind == "array_len":
            self._write_module_array_len_wrapper(helper)
            return
        if helper.kind == "get_derived":
            self._write_module_get_derived_wrapper(helper)
            return
        if helper.kind == "set_derived":
            self._write_module_set_derived_wrapper(helper)
            return

        wrapper_name = self._module_helper_wrapper_name(helper)
        helper_symbol = self._module_helper_symbol(helper)

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")

        if helper.kind == "get":
            if helper.is_type_member:
                self._write_type_member_get_wrapper(helper, helper_symbol)
            else:
                self._write_module_scalar_get_wrapper(helper, helper_symbol)
            self.dedent()
            self.write("}")
            self.write("")
            return

        if helper.kind == "set":
            if helper.is_type_member:
                self._write_type_member_set_wrapper(helper, helper_symbol)
            else:
                self._write_module_scalar_set_wrapper(helper, helper_symbol)
            self.dedent()
            self.write("}")
            self.write("")
            return

        else:  # array helper
            self.write("PyObject* dummy_handle = Py_None;")
            self.write("static char *kwlist[] = {\"handle\", NULL};")
            self.write(
                "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"|O\", kwlist, &dummy_handle)) {"
            )
            self.indent()
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("")

            self.write("int dummy_this[4] = {0, 0, 0, 0};")
            self.write("if (dummy_handle != Py_None) {")
            self.indent()
            self.write("PyObject* handle_sequence = PySequence_Fast(dummy_handle, \"Handle must be a sequence\");")
            self.write("if (handle_sequence == NULL) {")
            self.indent()
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);")
            self.write(f"if (handle_len != {self.handle_size}) {{")
            self.indent()
            self.write("Py_DECREF(handle_sequence);")
            self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
            self.indent()
            self.write("PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);")
            self.write("if (item == NULL) {")
            self.indent()
            self.write("Py_DECREF(handle_sequence);")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("dummy_this[i] = (int)PyLong_AsLong(item);")
            self.write("if (PyErr_Occurred()) {")
            self.indent()
            self.write("Py_DECREF(handle_sequence);")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.dedent()
            self.write("}")
            self.write("Py_DECREF(handle_sequence);")
            self.dedent()
            self.write("}")
            self.write("int nd = 0;")
            self.write("int dtype = 0;")
            self.write("int dshape[10] = {0};")
            self.write("long long handle = 0;")
            self.write(f"{helper_symbol}(dummy_this, &nd, &dtype, dshape, &handle);")

            self.write("if (nd < 0 || nd > 10) {")
            self.indent()
            self.write("PyErr_SetString(PyExc_ValueError, \"Invalid dimensionality\");")
            self.write("return NULL;")
            self.dedent()
            self.write("}")

            self.write("PyObject* shape_tuple = PyTuple_New(nd);")
            self.write("if (shape_tuple == NULL) {")
            self.indent()
            self.write("return NULL;")
            self.dedent()
            self.write("}")

            self.write("for (int i = 0; i < nd; ++i) {")
            self.indent()
            self.write("PyObject* dim = PyLong_FromLong((long)dshape[i]);")
            self.write("if (dim == NULL) {")
            self.indent()
            self.write("Py_DECREF(shape_tuple);")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("PyTuple_SET_ITEM(shape_tuple, i, dim);")
            self.dedent()
            self.write("}")

            self.write("PyObject* result = PyTuple_New(4);")
            self.write("if (result == NULL) {")
            self.indent()
            self.write("Py_DECREF(shape_tuple);")
            self.write("return NULL;")
            self.dedent()
            self.write("}")

            self.write("PyObject* nd_obj = PyLong_FromLong((long)nd);")
            self.write("if (nd_obj == NULL) {")
            self.indent()
            self.write("Py_DECREF(shape_tuple);")
            self.write("Py_DECREF(result);")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("PyTuple_SET_ITEM(result, 0, nd_obj);")

            self.write("PyObject* dtype_obj = PyLong_FromLong((long)dtype);")
            self.write("if (dtype_obj == NULL) {")
            self.indent()
            self.write("Py_DECREF(shape_tuple);")
            self.write("Py_DECREF(result);")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("PyTuple_SET_ITEM(result, 1, dtype_obj);")

            self.write("PyTuple_SET_ITEM(result, 2, shape_tuple);")
            self.write("shape_tuple = NULL;")

            self.write("PyObject* handle_obj = PyLong_FromLongLong(handle);")
            self.write("if (handle_obj == NULL) {")
            self.indent()
            self.write("Py_DECREF(result);")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("PyTuple_SET_ITEM(result, 3, handle_obj);")
            self.write("return result;")

        self.dedent()
        self.write("}")
        self.write("")

    def _write_module_scalar_get_wrapper(self, helper: ModuleHelper, helper_symbol: str) -> None:
        self.write(
            "if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {"
        )
        self.indent()
        self.write(
            "PyErr_SetString(PyExc_TypeError, \"This helper does not accept arguments\");"
        )
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("")

        fmt = build_arg_format(helper.element.type)
        if fmt == "s":
            length_expr = self._character_length_expr(helper.element.type) or "1024"
            self.write(f"int value_len = {length_expr};")
            self.write("if (value_len <= 0) {")
            self.indent()
            self.write(
                "PyErr_SetString(PyExc_ValueError, \"Character helper length must be positive\");"
            )
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("char* buffer = (char*)malloc((size_t)value_len + 1);")
            self.write("if (buffer == NULL) {")
            self.indent()
            self.write("PyErr_NoMemory();")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("memset(buffer, ' ', value_len);")
            self.write("buffer[value_len] = '\\0';")
            self.write(f"{helper_symbol}(buffer, value_len);")
            self.write("int actual_len = value_len;")
            self.write("while (actual_len > 0 && buffer[actual_len - 1] == ' ') {")
            self.indent()
            self.write("--actual_len;")
            self.dedent()
            self.write("}")
            self.write("PyObject* result = PyUnicode_FromStringAndSize(buffer, actual_len);")
            self.write("free(buffer);")
            self.write("if (result == NULL) {")
            self.indent()
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("return result;")
            return

        needs_bool = helper.element.type.strip().lower().startswith("logical")
        c_type = c_type_from_fortran(helper.element.type, self.kind_map)
        self.write(f"{c_type} value;")
        self.write(f"{helper_symbol}(&value);")
        if needs_bool:
            self.write("return PyBool_FromLong(value);")
        else:
            self.write(f"return Py_BuildValue(\"{fmt}\", value);")

    def _write_module_scalar_set_wrapper(self, helper: ModuleHelper, helper_symbol: str) -> None:
        fmt = parse_arg_format(helper.element.type)
        if fmt == "s":
            kw_name = helper.element.name
            self.write("const char* value_str;")
            self.write(f"static char *kwlist[] = {{\"{kw_name}\", NULL}};")
            self.write(
                "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"s\", kwlist, &value_str)) {"
            )
            self.indent()
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("int value_len = (int)strlen(value_str);")
            self.write("if (value_len < 0) value_len = 0;")
            self.write("char* value = (char*)malloc((size_t)value_len + 1);")
            self.write("if (value == NULL) {")
            self.indent()
            self.write("PyErr_NoMemory();")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("memcpy(value, value_str, (size_t)value_len);")
            self.write("value[value_len] = '\\0';")
            self.write(f"{helper_symbol}(value, value_len);")
            self.write("free(value);")
            self.write("Py_RETURN_NONE;")
            return

        c_type = c_type_from_fortran(helper.element.type, self.kind_map)
        kw_name = helper.element.name
        self.write(f"{c_type} value;")
        self.write(f"static char *kwlist[] = {{\"{kw_name}\", NULL}};")
        self.write(
            f"if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"{fmt}\", kwlist, &value)) {{"
        )
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(f"{helper_symbol}(&value);")
        self.write("Py_RETURN_NONE;")

    def _write_type_member_get_wrapper(self, helper: ModuleHelper, helper_symbol: str) -> None:
        fmt = build_arg_format(helper.element.type)
        self.write("PyObject* py_handle;")
        self.write("static char *kwlist[] = {\"handle\", NULL};")
        self.write("if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O\", kwlist, &py_handle)) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write("PyObject* handle_sequence = PySequence_Fast(py_handle, \"Handle must be a sequence\");")
        self.write("if (handle_sequence == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);")
        self.write(f"if (handle_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(handle_sequence);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(f"int this_handle[{self.handle_size}] = {{0}};")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);")
        self.write("if (item == NULL) {")
        self.indent()
        self.write("Py_DECREF(handle_sequence);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("this_handle[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("Py_DECREF(handle_sequence);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")
        self.write("Py_DECREF(handle_sequence);")

        if fmt == "s":
            length_expr = self._character_length_expr(helper.element.type) or "1024"
            self.write(f"int value_len = {length_expr};")
            self.write("if (value_len <= 0) {")
            self.indent()
            self.write(
                "PyErr_SetString(PyExc_ValueError, \"Character helper length must be positive\");"
            )
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("char* buffer = (char*)malloc((size_t)value_len + 1);")
            self.write("if (buffer == NULL) {")
            self.indent()
            self.write("PyErr_NoMemory();")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("memset(buffer, ' ', value_len);")
            self.write("buffer[value_len] = '\\0';")
            self.write(f"{helper_symbol}(this_handle, buffer, value_len);")
            self.write("int actual_len = value_len;")
            self.write("while (actual_len > 0 && buffer[actual_len - 1] == ' ') {")
            self.indent()
            self.write("--actual_len;")
            self.dedent()
            self.write("}")
            self.write("PyObject* result = PyUnicode_FromStringAndSize(buffer, actual_len);")
            self.write("free(buffer);")
            self.write("return result;")
            return

        c_type = c_type_from_fortran(helper.element.type, self.kind_map)
        self.write(f"{c_type} value;")
        self.write(f"{helper_symbol}(this_handle, &value);")
        self.write(f"return Py_BuildValue(\"{fmt}\", value);")

    def _write_type_member_set_wrapper(self, helper: ModuleHelper, helper_symbol: str) -> None:
        fmt = parse_arg_format(helper.element.type)
        self.write("PyObject* py_handle;")
        self.write("PyObject* py_value;")
        self.write("static char *kwlist[] = {\"handle\", \"value\", NULL};")
        self.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"OO\", kwlist, &py_handle, &py_value)) {"
        )
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write("PyObject* handle_sequence = PySequence_Fast(py_handle, \"Handle must be a sequence\");")
        self.write("if (handle_sequence == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);")
        self.write(f"if (handle_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(handle_sequence);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(f"int this_handle[{self.handle_size}] = {{0}};")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);")
        self.write("if (item == NULL) {")
        self.indent()
        self.write("Py_DECREF(handle_sequence);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("this_handle[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("Py_DECREF(handle_sequence);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")
        self.write("Py_DECREF(handle_sequence);")

        if fmt == "s":
            self.write("if (!PyUnicode_Check(py_value)) {")
            self.indent()
            self.write("PyErr_SetString(PyExc_TypeError, \"Value must be str\");")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("PyObject* bytes = PyUnicode_AsUTF8String(py_value);")
            self.write("if (bytes == NULL) {")
            self.indent()
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write("const char* buffer = PyBytes_AsString(bytes);")
            self.write("Py_ssize_t len = PyBytes_GET_SIZE(bytes);")
            self.write(f"{helper_symbol}(this_handle, buffer, (int)len);")
            self.write("Py_DECREF(bytes);")
            self.write("Py_RETURN_NONE;")
            return

        c_type = c_type_from_fortran(helper.element.type, self.kind_map)
        self.write(f"{c_type} value;")
        self.write(
            f"if (!PyArg_Parse(py_value, \"{fmt}\", &value)) {{ return NULL; }}"
        )
        self.write(f"{helper_symbol}(this_handle, &value);")
        self.write("Py_RETURN_NONE;")

    def _write_module_array_getitem_wrapper(self, helper: ModuleHelper) -> None:
        """Wrapper for module-level derived-type array getitem."""

        wrapper_name = self._module_helper_wrapper_name(helper)
        helper_symbol = self._module_helper_symbol(helper)

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")
        self.write("PyObject* py_parent;")
        self.write("int index = 0;")
        self.write("static char *kwlist[] = {\"handle\", \"index\", NULL};")
        self.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"Oi\", kwlist, &py_parent, &index)) {"
        )
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(
            "PyObject* parent_sequence = PySequence_Fast(py_parent, \"Handle must be a sequence\");"
        )
        self.write("if (parent_sequence == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(
            "Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);"
        )
        self.write(f"if (parent_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"int parent_handle[{self.handle_size}] = {{0}};")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);")
        self.write("parent_handle[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")

        self.write(f"int handle[{self.handle_size}] = {{0}};")
        self.write(f"{helper_symbol}(parent_handle, &index, handle);")
        self.write("Py_DECREF(parent_sequence);")

        self.write(f"PyObject* result = PyList_New({self.handle_size});")
        self.write("if (result == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PyLong_FromLong((long)handle[i]);")
        self.write("if (item == NULL) {")
        self.indent()
        self.write("Py_DECREF(result);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("PyList_SET_ITEM(result, i, item);")
        self.dedent()
        self.write("}")
        self.write("return result;")
        self.dedent()
        self.write("}")
        self.write("")

    def _write_module_array_setitem_wrapper(self, helper: ModuleHelper) -> None:
        """Wrapper for module-level derived-type array setitem."""

        wrapper_name = self._module_helper_wrapper_name(helper)
        helper_symbol = self._module_helper_symbol(helper)

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")
        self.write("PyObject* py_parent;")
        self.write("int index = 0;")
        self.write("PyObject* py_value;")
        self.write("static char *kwlist[] = {\"handle\", \"index\", \"value\", NULL};")
        self.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"OiO\", kwlist, &py_parent, &index, &py_value)) {"
        )
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(
            "PyObject* parent_sequence = PySequence_Fast(py_parent, \"Handle must be a sequence\");"
        )
        self.write("if (parent_sequence == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(
            "Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);"
        )
        self.write(f"if (parent_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write("PyObject* value_handle_obj = NULL;")
        self.write("PyObject* value_sequence = NULL;")
        self.write("Py_ssize_t value_handle_len = 0;")
        self.write("if (PyObject_HasAttrString(py_value, \"_handle\")) {")
        self.indent()
        self.write("value_handle_obj = PyObject_GetAttrString(py_value, \"_handle\");")
        self.write("if (value_handle_obj == NULL) { return NULL; }")
        self.write(
            "value_sequence = PySequence_Fast(value_handle_obj, \"Failed to access handle sequence\");"
        )
        self.write("if (value_sequence == NULL) { Py_DECREF(value_handle_obj); return NULL; }")
        self.dedent()
        self.write("} else if (PySequence_Check(py_value)) {")
        self.indent()
        self.write(
            "value_sequence = PySequence_Fast(py_value, \"Argument value must be a handle sequence\");"
        )
        self.write("if (value_sequence == NULL) { return NULL; }")
        self.dedent()
        self.write("} else {")
        self.indent()
        self.write(
            "PyErr_SetString(PyExc_TypeError, \"Argument value must be a Fortran derived-type instance\");"
        )
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write("value_handle_len = PySequence_Fast_GET_SIZE(value_sequence);")
        self.write(f"if (value_handle_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"int parent_handle[{self.handle_size}] = {{0}};")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);")
        self.write("parent_handle[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")
        self.write("Py_DECREF(parent_sequence);")

        self.write(f"int* value = (int*)malloc(sizeof(int) * {self.handle_size});")
        self.write("if (value == NULL) {")
        self.indent()
        self.write("PyErr_NoMemory();")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);")
        self.write("if (item == NULL) {")
        self.indent()
        self.write("free(value);")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("value[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("free(value);")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")

        self.write(f"{helper_symbol}(parent_handle, &index, value);")
        self.write("free(value);")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("Py_RETURN_NONE;")
        self.dedent()
        self.write("}")
        self.write("")

    def _write_module_array_len_wrapper(self, helper: ModuleHelper) -> None:
        """Wrapper for module-level derived-type array length."""

        wrapper_name = self._module_helper_wrapper_name(helper)
        helper_symbol = self._module_helper_symbol(helper)

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")
        self.write("PyObject* py_parent;")
        self.write("static char *kwlist[] = {\"handle\", NULL};")
        self.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O\", kwlist, &py_parent)) {"
        )
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(
            "PyObject* parent_sequence = PySequence_Fast(py_parent, \"Handle must be a sequence\");"
        )
        self.write("if (parent_sequence == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(
            "Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);"
        )
        self.write(f"if (parent_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"int parent_handle[{self.handle_size}] = {{0}};")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);")
        self.write("parent_handle[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")

        self.write("int length = 0;")
        self.write(f"{helper_symbol}(parent_handle, &length);")
        self.write("Py_DECREF(parent_sequence);")
        self.write("return PyLong_FromLong((long)length);")
        self.dedent()
        self.write("}")
        self.write("")

    def _write_module_get_derived_wrapper(self, helper: ModuleHelper) -> None:
        """Wrapper for derived-type scalar getters returning handles."""
        wrapper_name = self._module_helper_wrapper_name(helper)
        helper_symbol = self._module_helper_symbol(helper)

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")
        self.write("PyObject* py_handle;")
        self.write("static char *kwlist[] = {\"handle\", NULL};")
        self.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"O\", kwlist, &py_handle)) {"
        )
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write("PyObject* handle_sequence = PySequence_Fast(py_handle, \"Handle must be a sequence\");")
        self.write("if (handle_sequence == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);")
        self.write(f"if (handle_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(handle_sequence);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"int parent_handle[{self.handle_size}] = {{0}};")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);")
        self.write("parent_handle[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("Py_DECREF(handle_sequence);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")
        self.write("Py_DECREF(handle_sequence);")

        self.write(f"int value_handle[{self.handle_size}] = {{0}};")
        self.write(f"{helper_symbol}(parent_handle, value_handle);")

        self.write(f"PyObject* result = PyList_New({self.handle_size});")
        self.write("if (result == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PyLong_FromLong((long)value_handle[i]);")
        self.write("if (item == NULL) {")
        self.indent()
        self.write("Py_DECREF(result);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("PyList_SET_ITEM(result, i, item);")
        self.dedent()
        self.write("}")
        self.write("return result;")
        self.dedent()
        self.write("}")
        self.write("")

    def _write_module_set_derived_wrapper(self, helper: ModuleHelper) -> None:
        """Wrapper for derived-type scalar setters accepting handles."""

        wrapper_name = self._module_helper_wrapper_name(helper)
        helper_symbol = self._module_helper_symbol(helper)

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")
        self.write("PyObject* py_parent;")
        self.write("PyObject* py_value;")
        self.write("static char *kwlist[] = {\"handle\", \"value\", NULL};")
        self.write(
            "if (!PyArg_ParseTupleAndKeywords(args, kwargs, \"OO\", kwlist, &py_parent, &py_value)) {"
        )
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write("PyObject* parent_sequence = PySequence_Fast(py_parent, \"Handle must be a sequence\");")
        self.write("if (parent_sequence == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);")
        self.write(f"if (parent_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write("PyObject* value_sequence = NULL;")
        self.write("PyObject* value_handle_obj = NULL;")
        self.write("if (PyObject_HasAttrString(py_value, \"_handle\")) {")
        self.indent()
        self.write("value_handle_obj = PyObject_GetAttrString(py_value, \"_handle\");")
        self.write("if (value_handle_obj == NULL) { Py_DECREF(parent_sequence); return NULL; }")
        self.write(
            "value_sequence = PySequence_Fast(value_handle_obj, \"Failed to access handle sequence\");"
        )
        self.write("if (value_sequence == NULL) { Py_DECREF(parent_sequence); Py_DECREF(value_handle_obj); return NULL; }")
        self.dedent()
        self.write("} else if (PySequence_Check(py_value)) {")
        self.indent()
        self.write(
            "value_sequence = PySequence_Fast(py_value, \"Argument value must be a handle sequence\");"
        )
        self.write("if (value_sequence == NULL) { Py_DECREF(parent_sequence); return NULL; }")
        self.dedent()
        self.write("} else {")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write(
            "PyErr_SetString(PyExc_TypeError, \"Argument value must be a Fortran derived-type instance\");"
        )
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write("Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(value_sequence);")
        self.write(f"if (handle_len != {self.handle_size}) {{")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("PyErr_SetString(PyExc_ValueError, \"Unexpected handle length\");")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"int parent_handle[{self.handle_size}] = {{0}};")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);")
        self.write("parent_handle[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("Py_DECREF(parent_sequence);")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")
        self.write("Py_DECREF(parent_sequence);")

        self.write(f"int handle[{self.handle_size}] = {{0}};")
        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);")
        self.write("if (item == NULL) {")
        self.indent()
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("handle[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")

        self.write(f"{helper_symbol}(parent_handle, handle);")
        self.write("Py_DECREF(value_sequence);")
        self.write("if (value_handle_obj) Py_DECREF(value_handle_obj);")
        self.write("Py_RETURN_NONE;")
        self.dedent()
        self.write("}")
        self.write("")

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
            elif self._is_derived_type(arg):
                params.append(f"int* {arg.name}")
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

    def _write_module_helper_declaration(self, helper: ModuleHelper) -> None:
        """Write extern declaration for module-level helper routines."""

        symbol = self._module_helper_symbol(helper)
        if helper.kind in {"get", "set", "get_derived", "set_derived"}:
            c_type = c_type_from_fortran(helper.element.type, self.kind_map)
            is_char = helper.element.type.strip().lower().startswith("character")
            if helper.kind in {"get_derived", "set_derived"}:
                self.write(f"extern void {symbol}(int* value);")
            elif helper.is_type_member:
                if is_char:
                    self.write(
                        f"extern void {symbol}(int* handle, char* value, int value_len);"
                    )
                else:
                    self.write(f"extern void {symbol}(int* handle, {c_type}* value);")
            else:
                if is_char:
                    self.write(f"extern void {symbol}(char* value, int value_len);")
                else:
                    self.write(f"extern void {symbol}({c_type}* value);")
        elif helper.kind == "array":
            self.write(
                f"extern void {symbol}(int* dummy_this, int* nd, int* dtype, int* dshape, long long* handle);"
            )
        elif helper.kind == "array_getitem":
            self.write(
                f"extern void {symbol}(int* dummy_this, int* index, int* handle);"
            )
        elif helper.kind == "array_setitem":
            self.write(
                f"extern void {symbol}(int* dummy_this, int* index, int* handle);"
            )
        elif helper.kind == "array_len":
            self.write(
                f"extern void {symbol}(int* dummy_this, int* length);"
            )

    def _write_wrapper_function(self, proc: ft.Procedure, mod_name: str) -> None:
        """Write Python C API wrapper function for a procedure."""

        if 'constructor' in proc.attributes:
            self._write_constructor_wrapper(proc, mod_name)
            return

        if 'destructor' in proc.attributes:
            self._write_destructor_wrapper(proc, mod_name)
            return

        wrapper_name = self._wrapper_name(mod_name, proc)

        prev_value_map = getattr(self, "_value_map", None)
        prev_hidden = getattr(self, "_hidden_names", set())
        prev_hidden_lower = getattr(self, "_hidden_names_lower", set())
        prev_proc = getattr(self, "_current_proc", None)
        self._value_map = self._build_value_map(proc)
        self._hidden_names = {arg.name for arg in proc.arguments if self._is_hidden_argument(arg)}
        self._hidden_names_lower = {name.lower() for name in self._hidden_names}
        self._current_proc = proc

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

        if self._procedure_error_args(proc):
            self._write_auto_raise_guard(proc)

        # Build return value
        self._write_return_value(proc)

        self.dedent()
        self.write("}")
        self.write("")

        self._value_map = prev_value_map
        self._hidden_names = prev_hidden
        self._hidden_names_lower = prev_hidden_lower
        self._current_proc = prev_proc

    def _write_constructor_wrapper(self, proc: ft.Procedure, mod_name: str) -> None:
        """Specialised wrapper for derived-type constructors."""

        wrapper_name = self._wrapper_name(mod_name, proc)
        helper_symbol = self._helper_symbol(proc)

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")
        self.write(
            "if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {"
        )
        self.indent()
        self.write(
            "PyErr_SetString(PyExc_TypeError, \"Constructor does not accept arguments\");"
        )
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("")

        self.write(f"int handle[{self.handle_size}] = {{0}};")
        self.write(f"{helper_symbol}(handle);")

        self.write(f"PyObject* py_handle = PyList_New({self.handle_size});")
        self.write("if (py_handle == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
        self.indent()
        self.write("PyObject* item = PyLong_FromLong((long)handle[i]);")
        self.write("if (item == NULL) {")
        self.indent()
        self.write("Py_DECREF(py_handle);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write("PyList_SET_ITEM(py_handle, i, item);")
        self.dedent()
        self.write("}")

        self.write("return py_handle;")
        self.dedent()
        self.write("}")
        self.write("")

    def _write_destructor_wrapper(self, proc: ft.Procedure, mod_name: str) -> None:
        """Specialised wrapper for derived-type destructors."""

        wrapper_name = self._wrapper_name(mod_name, proc)
        helper_symbol = self._helper_symbol(proc)
        arg = proc.arguments[0]

        self.write(
            f"static PyObject* {wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
        )
        self.write("{")
        self.indent()
        self.write("(void)self;")

        self._write_arg_parsing(proc)
        self._write_arg_preparation(proc)

        self.write(f"/* Call f90wrap helper */")
        self.write(f"{helper_symbol}({arg.name});")

        # Cleanup for derived handle
        self.write(f"if ({arg.name}_sequence) {{")
        self.indent()
        self.write(f"Py_DECREF({arg.name}_sequence);")
        self.dedent()
        self.write("}")
        self.write(f"if ({arg.name}_handle_obj) {{")
        self.indent()
        self.write(f"Py_DECREF({arg.name}_handle_obj);")
        self.dedent()
        self.write("}")
        self.write(f"free({arg.name});")

        self.write("Py_RETURN_NONE;")
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

        format_parts: List[str] = []
        parse_vars: List[str] = []
        kw_names: List[str] = []
        optional_started = False

        for arg in proc.arguments:
            if self._is_hidden_argument(arg):
                continue

            intent = self._arg_intent(arg)
            optional = self._is_optional(arg)
            should_parse = self._should_parse_argument(arg)

            if not should_parse:
                if not self._is_array(arg) and not self._is_derived_type(arg) and not arg.type.lower().startswith("character"):
                    c_type = c_type_from_fortran(arg.type, self.kind_map)
                    self.write(f"{c_type} {arg.name}_val = 0;")
                continue

            if optional and not optional_started:
                format_parts.append("|")
                optional_started = True

            if self._is_derived_type(arg):
                format_parts.append("O")
                self.write(f"PyObject* py_{arg.name} = NULL;")
                parse_vars.append(f"&py_{arg.name}")
            elif self._is_array(arg):
                format_parts.append("O")
                self.write(f"PyObject* py_{arg.name} = NULL;")
                parse_vars.append(f"&py_{arg.name}")
            elif arg.type.lower().startswith("character"):
                if optional or intent != "in":
                    format_parts.append("O")
                    self.write(f"PyObject* py_{arg.name} = Py_None;")
                    parse_vars.append(f"&py_{arg.name}")
                else:
                    format_parts.append("s")
                    self.write(f"const char* {arg.name}_str;")
                    parse_vars.append(f"&{arg.name}_str")
            else:
                c_type = c_type_from_fortran(arg.type, self.kind_map)
                if optional:
                    format_parts.append("O")
                    self.write(f"PyObject* py_{arg.name} = Py_None;")
                    self.write(f"{c_type} {arg.name}_val = 0;")
                    parse_vars.append(f"&py_{arg.name}")
                else:
                    fmt = parse_arg_format(arg.type)
                    format_parts.append(fmt)
                    self.write(f"{c_type} {arg.name}_val;")
                    parse_vars.append(f"&{arg.name}_val")

            kw_names.append(f'"{arg.name}"')

        if parse_vars:
            format_str = "".join(format_parts) if format_parts else ""
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
            intent = self._arg_intent(arg)
            optional = self._is_optional(arg)
            parsed = self._should_parse_argument(arg)

            if self._is_array(arg):
                self._declare_array_storage(arg)
                if parsed:
                    if optional:
                        self.write(
                            f"if (py_{arg.name} == NULL || py_{arg.name} == Py_None) {{"
                        )
                        self.indent()
                        self.write(
                            f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} cannot be None");'
                        )
                        self.write("return NULL;")
                        self.dedent()
                        self.write("}")
                    self._write_array_preparation(arg)
                else:
                    self._prepare_output_array(arg)
            elif arg.type.lower().startswith("character"):
                self._prepare_character_argument(arg, intent, optional)
            elif self._is_derived_type(arg):
                if self._should_parse_argument(arg):
                    if optional:
                        self.write(f"if (py_{arg.name} == Py_None) {{")
                        self.indent()
                        self.write(
                            f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} cannot be None");'
                        )
                        self.write("return NULL;")
                        self.dedent()
                        self.write("}")
                    self._write_derived_preparation(arg)
                else:
                    self.write(f"int {arg.name}[{self.handle_size}] = {{0}};")
            else:
                self._prepare_scalar_argument(arg, intent, optional)

    def _prepare_scalar_argument(self, arg: ft.Argument, intent: str, optional: bool) -> None:
        """Prepare scalar argument values."""

        c_type = c_type_from_fortran(arg.type, self.kind_map)

        if not self._should_parse_argument(arg):
            return

        if optional:
            self.write(f"if (py_{arg.name} == Py_None) {{")
            self.indent()
            self.write(f"{arg.name}_val = 0;")
            self.dedent()
            self.write("} else {")
            self.indent()
            fmt = parse_arg_format(arg.type)
            if fmt in {"i", "l", "h", "I"}:
                self.write(f"{arg.name}_val = ({c_type})PyLong_AsLong(py_{arg.name});")
                self.write("if (PyErr_Occurred()) {")
                self.indent()
                self.write("return NULL;")
                self.dedent()
                self.write("}")
            elif fmt in {"d", "f"}:
                self.write(f"{arg.name}_val = ({c_type})PyFloat_AsDouble(py_{arg.name});")
                self.write("if (PyErr_Occurred()) {")
                self.indent()
                self.write("return NULL;")
                self.dedent()
                self.write("}")
            else:
                self.write(
                    f'PyErr_SetString(PyExc_TypeError, "Unsupported optional argument {arg.name}");'
                )
                self.write("return NULL;")
            self.dedent()
            self.write("}")

    def _prepare_character_argument(self, arg: ft.Argument, intent: str, optional: bool) -> None:
        """Allocate and populate character buffers."""

        type_spec = arg.type
        default_len = self._character_length_expr(type_spec) or "1024"

        if self._should_parse_argument(arg):
            if optional or intent != "in":
                self.write(f"int {arg.name}_len = 0;")
                self.write(f"char* {arg.name} = NULL;")
                self.write(f"if (py_{arg.name} == Py_None) {{")
                self.indent()
                self.write(f"{arg.name}_len = {default_len};")
                self.write(f"if ({arg.name}_len <= 0) {{")
                self.indent()
                self.write(
                    f'PyErr_SetString(PyExc_ValueError, "Character length for {arg.name} must be positive");'
                )
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                self.write(f"{arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
                self.write(f"if ({arg.name} == NULL) {{")
                self.indent()
                self.write("PyErr_NoMemory();")
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                self.write(f"memset({arg.name}, ' ', {arg.name}_len);")
                self.write(f"{arg.name}[{arg.name}_len] = '\\0';")
                self.dedent()
                self.write("} else {")
                self.indent()
                self.write(f"PyObject* {arg.name}_bytes = NULL;")
                self.write(f"if (PyBytes_Check(py_{arg.name})) {{")
                self.indent()
                self.write(f"{arg.name}_bytes = py_{arg.name};")
                self.write(f"Py_INCREF({arg.name}_bytes);")
                self.dedent()
                self.write(f"}} else if (PyUnicode_Check(py_{arg.name})) {{")
                self.indent()
                self.write(f"{arg.name}_bytes = PyUnicode_AsUTF8String(py_{arg.name});")
                self.write(f"if ({arg.name}_bytes == NULL) {{")
                self.indent()
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                self.dedent()
                self.write("} else {")
                self.indent()
                self.write(
                    f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be str or bytes");'
                )
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                self.write(f"{arg.name}_len = (int)PyBytes_GET_SIZE({arg.name}_bytes);")
                self.write(f"{arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
                self.write(f"if ({arg.name} == NULL) {{")
                self.indent()
                self.write(f"Py_DECREF({arg.name}_bytes);")
                self.write("PyErr_NoMemory();")
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                self.write(
                    f"memcpy({arg.name}, PyBytes_AS_STRING({arg.name}_bytes), (size_t){arg.name}_len);"
                )
                self.write(f"{arg.name}[{arg.name}_len] = '\\0';")
                self.write(f"Py_DECREF({arg.name}_bytes);")
                self.dedent()
                self.write("}")
            else:
                self.write(f"int {arg.name}_len = (int)strlen({arg.name}_str);")
                self.write(f"char* {arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
                self.write(f"if ({arg.name} == NULL) {{")
                self.indent()
                self.write("PyErr_NoMemory();")
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                self.write(
                    f"memcpy({arg.name}, {arg.name}_str, (size_t){arg.name}_len + 1);"
                )
        else:
            self.write(f"int {arg.name}_len = {default_len};")
            self.write(f"if ({arg.name}_len <= 0) {{")
            self.indent()
            self.write(
                f'PyErr_SetString(PyExc_ValueError, "Character length for {arg.name} must be positive");'
            )
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write(f"char* {arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
            self.write(f"if ({arg.name} == NULL) {{")
            self.indent()
            self.write("PyErr_NoMemory();")
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            self.write(f"memset({arg.name}, ' ', {arg.name}_len);")
            self.write(f"{arg.name}[{arg.name}_len] = '\\0';")

    def _declare_array_storage(self, arg: ft.Argument) -> None:
        """Declare local variables needed for array arguments."""

        c_type = c_type_from_fortran(arg.type, self.kind_map)
        self.write(f"PyArrayObject* {arg.name}_arr = NULL;")
        if self._is_output_argument(arg):
            self.write(f"PyObject* py_{arg.name}_arr = NULL;")
        self.write(f"{c_type}* {arg.name} = NULL;")

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
        self.write(f"{arg.name}_arr = (PyArrayObject*)PyArray_FROM_OTF(")
        self.write(
            f"    py_{arg.name}, {numpy_type}, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);")
        self.write(f"if ({arg.name}_arr == NULL) {{")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"{arg.name} = ({c_type}*)PyArray_DATA({arg.name}_arr);")

        # Get dimensions if needed
        dims = self._extract_dimensions(arg)
        if dims:
            for i in range(len(dims)):
                self.write(f"int n{i}_{arg.name} = (int)PyArray_DIM({arg.name}_arr, {i});")
            for i, dim in enumerate(dims):
                dim_name = dim.strip()
                if dim_name and dim_name.startswith("f90wrap_"):
                    self.write(f"{dim_name}_val = n{i}_{arg.name};")

        if self._is_output_argument(arg):
            self.write(f"py_{arg.name}_arr = (PyObject*){arg.name}_arr;")

        self.write("")

    def _prepare_output_array(self, arg: ft.Argument) -> None:
        """Allocate NumPy array for output-only arguments."""

        proc = getattr(self, "_current_proc", None)
        trans_dims = self._extract_dimensions(arg)
        orig_dims = self._original_dimensions(proc, arg.name)
        if not trans_dims and not orig_dims:
            trans_dims = ["1"]

        count = max(len(trans_dims), len(orig_dims or []))
        if count == 0:
            count = 1

        dim_vars = []
        for index in range(count):
            trans_token = trans_dims[index] if index < len(trans_dims) else None
            source_expr = None
            if orig_dims and index < len(orig_dims):
                source_expr = orig_dims[index]
            if not source_expr:
                source_expr = trans_token or "1"

            expr = self._dimension_c_expression(source_expr)
            size_var = f"{arg.name}_dim_{index}"
            self.write(f"npy_intp {size_var} = (npy_intp)({expr});")
            self.write(f"if ({size_var} <= 0) {{")
            self.indent()
            self.write(
                f'PyErr_SetString(PyExc_ValueError, "Dimension for {arg.name} must be positive");'
            )
            self.write("return NULL;")
            self.dedent()
            self.write("}")
            if trans_token:
                token = trans_token.strip()
                hidden_lower = getattr(self, "_hidden_names_lower", set())
                value_map = getattr(self, "_value_map", {})
                token_lower = token.lower()
                if token_lower in hidden_lower:
                    replacement = value_map.get(token) or value_map.get(token_lower)
                    if replacement:
                        self.write(f"{replacement} = (int){size_var};")
            dim_vars.append(size_var)

        dims_array = f"{arg.name}_dims"
        self.write(f"npy_intp {dims_array}[{len(dim_vars)}] = {{{', '.join(dim_vars)}}};")
        numpy_type = numpy_type_from_fortran(arg.type, self.kind_map)
        self.write(
            f"py_{arg.name}_arr = PyArray_SimpleNew({len(dim_vars)}, {dims_array}, {numpy_type});"
        )
        self.write(f"if (py_{arg.name}_arr == NULL) {{")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(f"{arg.name}_arr = (PyArrayObject*)py_{arg.name}_arr;")
        c_type = c_type_from_fortran(arg.type, self.kind_map)
        self.write(f"{arg.name} = ({c_type}*)PyArray_DATA({arg.name}_arr);")
        self.write("")

    def _write_derived_preparation(self, arg: ft.Argument) -> None:
        """Extract derived-type handle from Python object."""

        name = arg.name
        self.write(f"PyObject* {name}_handle_obj = NULL;")
        self.write(f"PyObject* {name}_sequence = NULL;")
        self.write(f"Py_ssize_t {name}_handle_len = 0;")

        self.write(f"if (PyObject_HasAttrString(py_{name}, \"_handle\")) {{")
        self.indent()
        self.write(f"{name}_handle_obj = PyObject_GetAttrString(py_{name}, \"_handle\");")
        self.write(f"if ({name}_handle_obj == NULL) {{")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(
            f"{name}_sequence = PySequence_Fast({name}_handle_obj, \"Failed to access handle sequence\");"
        )
        self.write(f"if ({name}_sequence == NULL) {{")
        self.indent()
        self.write(f"Py_DECREF({name}_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write(f"}} else if (PySequence_Check(py_{name})) {{")
        self.indent()
        self.write(
            f"{name}_sequence = PySequence_Fast(py_{name}, \"Argument {name} must be a handle sequence\");"
        )
        self.write(f"if ({name}_sequence == NULL) {{")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("} else {")
        self.indent()
        self.write(
            f'PyErr_SetString(PyExc_TypeError, "Argument {name} must be a Fortran derived-type instance");'
        )
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(
            f"{name}_handle_len = PySequence_Fast_GET_SIZE({name}_sequence);")
        self.write(f"if ({name}_handle_len <= 0) {{")
        self.indent()
        self.write(
            f'PyErr_SetString(PyExc_ValueError, "Argument {name} has no active handle");'
        )
        self.write(f"Py_DECREF({name}_sequence);")
        self.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"int* {name} = (int*)malloc(sizeof(int) * {name}_handle_len);")
        self.write(f"if ({name} == NULL) {{")
        self.indent()
        self.write("PyErr_NoMemory();")
        self.write(f"Py_DECREF({name}_sequence);")
        self.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"for (Py_ssize_t i = 0; i < {name}_handle_len; ++i) {{")
        self.indent()
        self.write(
            f"PyObject* item = PySequence_Fast_GET_ITEM({name}_sequence, i);")
        self.write("if (item == NULL) {")
        self.indent()
        self.write(f"free({name});")
        self.write(f"Py_DECREF({name}_sequence);")
        self.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.write(f"{name}[i] = (int)PyLong_AsLong(item);")
        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self.write(f"free({name});")
        self.write(f"Py_DECREF({name}_sequence);")
        self.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        self.dedent()
        self.write("}")
        self.write(f"(void){name}_handle_len;  /* suppress unused warnings when unchanged */")
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
            elif self._is_derived_type(arg):
                call_args.append(arg.name)
            elif self._is_array(arg):
                call_args.append(arg.name)
            elif arg.type.lower().startswith("character"):
                call_args.append(arg.name)
                call_args.append(f"{arg.name}_len")
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

        output_args = [
            arg for arg in proc.arguments if self._is_output_argument(arg)
        ]

        if isinstance(proc, ft.Function):
            ret_type = proc.ret_val.type.lower()
            if self._is_array(proc.ret_val):
                self._write_array_return(proc.ret_val, "result")
            elif ret_type.startswith("logical"):
                self.write("return PyBool_FromLong(result);")
            else:
                fmt = build_arg_format(proc.ret_val.type)
                self.write(f'return Py_BuildValue("{fmt}", result);')

            # Clean up non-output buffers for functions
            for arg in proc.arguments:
                if arg.type.lower().startswith("character") and not self._is_output_argument(arg):
                    self.write(f"free({arg.name});")
                elif self._is_array(arg) and self._should_parse_argument(arg):
                    self.write(f"Py_DECREF({arg.name}_arr);")
                elif self._is_derived_type(arg):
                    self.write(f"if ({arg.name}_sequence) {{")
                    self.indent()
                    self.write(f"Py_DECREF({arg.name}_sequence);")
                    self.dedent()
                    self.write("}")
                    self.write(f"if ({arg.name}_handle_obj) {{")
                    self.indent()
                    self.write(f"Py_DECREF({arg.name}_handle_obj);")
                    self.dedent()
                    self.write("}")
                    self.write(f"free({arg.name});")
            return

        result_objects: List[str] = []

        for arg in output_args:
            if self._is_array(arg):
                result_objects.append(f"py_{arg.name}_arr")
                continue

            if arg.type.lower().startswith("character"):
                self.write(f"int {arg.name}_trim = {arg.name}_len;")
                self.write(f"while ({arg.name}_trim > 0 && {arg.name}[{arg.name}_trim - 1] == ' ') {{")
                self.indent()
                self.write(f"--{arg.name}_trim;")
                self.dedent()
                self.write("}")
                self.write(
                    f"PyObject* py_{arg.name}_obj = PyBytes_FromStringAndSize({arg.name}, {arg.name}_trim);"
                )
                self.write(f"free({arg.name});")
                self.write(f"if (py_{arg.name}_obj == NULL) {{")
                self.indent()
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                result_objects.append(f"py_{arg.name}_obj")
            elif self._is_derived_type(arg):
                parsed = self._should_parse_argument(arg)
                self.write(f"PyObject* py_{arg.name}_obj = PyList_New({self.handle_size});")
                self.write(f"if (py_{arg.name}_obj == NULL) {{")
                self.indent()
                if parsed:
                    self.write(f"free({arg.name});")
                    self.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
                    self.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                self.write(f"for (int i = 0; i < {self.handle_size}; ++i) {{")
                self.indent()
                self.write(f"PyObject* item = PyLong_FromLong((long){arg.name}[i]);")
                self.write("if (item == NULL) {")
                self.indent()
                self.write(f"Py_DECREF(py_{arg.name}_obj);")
                if parsed:
                    self.write(f"free({arg.name});")
                    self.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
                    self.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                self.write(f"PyList_SET_ITEM(py_{arg.name}_obj, i, item);")
                self.dedent()
                self.write("}")
                if parsed:
                    self.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
                    self.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    self.write(f"free({arg.name});")
                result_objects.append(f"py_{arg.name}_obj")
            else:
                fmt = build_arg_format(arg.type)
                self.write(
                    f"PyObject* py_{arg.name}_obj = Py_BuildValue(\"{fmt}\", {arg.name}_val);"
                )
                self.write(f"if (py_{arg.name}_obj == NULL) {{")
                self.indent()
                self.write("return NULL;")
                self.dedent()
                self.write("}")
                result_objects.append(f"py_{arg.name}_obj")

        # Clean up non-output buffers
        for arg in proc.arguments:
            if arg.type.lower().startswith("character") and not self._is_array(arg) and not self._is_output_argument(arg):
                self.write(f"free({arg.name});")
            elif self._is_array(arg):
                if not self._is_output_argument(arg):
                    self.write(f"Py_DECREF({arg.name}_arr);")
            elif self._is_derived_type(arg) and not self._is_output_argument(arg):
                self.write(f"if ({arg.name}_sequence) {{")
                self.indent()
                self.write(f"Py_DECREF({arg.name}_sequence);")
                self.dedent()
                self.write("}")
                self.write(f"if ({arg.name}_handle_obj) {{")
                self.indent()
                self.write(f"Py_DECREF({arg.name}_handle_obj);")
                self.dedent()
                self.write("}")
                self.write(f"free({arg.name});")

        if not result_objects:
            self.write("Py_RETURN_NONE;")
            return

        if len(result_objects) == 1:
            self.write(f"return {result_objects[0]};")
            return

        self.write(f"PyObject* result_tuple = PyTuple_New({len(result_objects)});")
        self.write("if (result_tuple == NULL) {")
        self.indent()
        for name in result_objects:
            self.write(f"Py_DECREF({name});")
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        for index, name in enumerate(result_objects):
            self.write(f"PyTuple_SET_ITEM(result_tuple, {index}, {name});")
        self.write("return result_tuple;")

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

    def _write_method_table(
        self,
        procedures: List[ft.Procedure],
        module_helpers: List[ModuleHelper],
        mod_name: str,
    ) -> None:
        """Write the module method table."""

        self.write(f"/* Method table for {mod_name} module */")
        self.write(f"static PyMethodDef {mod_name}_methods[] = {{")
        self.indent()

        for proc in procedures:
            wrapper_name = self._wrapper_name(mod_name, proc)
            method_name = self._helper_name(proc)
            docstring = proc.doc[0] if proc.doc else f"Wrapper for {proc.name}"
            # Escape any quotes and newlines in docstring
            docstring = docstring.replace('"', '\\"').replace('\n', '\\n')
            self.write(
                f'{{"{method_name}", (PyCFunction){wrapper_name}, '
                f"METH_VARARGS | METH_KEYWORDS, \"{docstring}\"}},"
            )

        for helper in module_helpers:
            wrapper_name = self._module_helper_wrapper_name(helper)
            method_name = self._module_helper_name(helper)
            if helper.kind == "array":
                docstring = f"Array helper for {helper.name}"
            else:
                docstring = f"Module helper for {helper.name}"
            docstring = docstring.replace('"', '\\"').replace('\n', '\\n')
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

        py_mod_name = mod_name
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
        self.write(f"PyObject* module = PyModule_Create(&{mod_name}module);")
        self.write("if (module == NULL) {")
        self.indent()
        self.write("return NULL;")
        self.dedent()
        self.write("}")
        if self.callbacks:
            for callback_name in self.callbacks:
                attr = callback_name
                self.write("Py_INCREF(Py_None);")
                self.write(
                    f"if (PyModule_AddObject(module, \"{attr}\", Py_None) < 0) {{"
                )
                self.indent()
                self.write("Py_DECREF(Py_None);")
                self.write("Py_DECREF(module);")
                self.write("return NULL;")
                self.dedent()
                self.write("}")
        self.write("return module;")
        self.dedent()
        self.write("}")

    def _is_array(self, arg: ft.Argument) -> bool:
        """Check if argument is an array."""
        return any("dimension" in attr for attr in arg.attributes)

    def _is_derived_type(self, arg: ft.Argument) -> bool:
        """Return True if argument represents a derived type handle."""

        ftype = arg.type.strip().lower()
        return ftype.startswith("type(") or ftype.startswith("class(")

    def _is_hidden_argument(self, arg: ft.Argument) -> bool:
        """Return True if argument is hidden from the Python API."""

        return any(attr.startswith("intent(hide)") for attr in arg.attributes)

    def _arg_intent(self, arg: ft.Argument) -> str:
        """Return the declared intent for an argument."""

        for attr in arg.attributes:
            if attr.startswith("intent(") and attr.endswith(")"):
                return attr[len("intent(") : -1].strip().lower()
        return "in"

    def _is_optional(self, arg: ft.Argument) -> bool:
        """Return True if the argument is optional."""

        return any(attr.strip().lower() == "optional" for attr in arg.attributes)

    def _should_parse_argument(self, arg: ft.Argument) -> bool:
        """Determine whether the argument should be parsed from Python."""

        if self._is_hidden_argument(arg):
            return False

        if self._is_optional(arg):
            return True

        intent = self._arg_intent(arg)
        return intent != "out"

    def _build_value_map(self, proc: ft.Procedure) -> Dict[str, str]:
        """Create mapping from Fortran argument names to C variable names."""

        mapping: Dict[str, str] = {}
        for arg in proc.arguments:
            if self._is_hidden_argument(arg):
                mapping[arg.name] = f"{arg.name}_val"
                continue

            if self._is_array(arg):
                if self._should_parse_argument(arg):
                    mapping[arg.name] = f"{arg.name}_arr"
                continue

            if arg.type.lower().startswith("character"):
                mapping[arg.name] = arg.name
            else:
                mapping[arg.name] = f"{arg.name}_val"
        return mapping

    def _dimension_c_expression(self, expr: str) -> str:
        """Convert a Fortran dimension expression into C code."""

        import re

        expression = expr.strip()
        if not expression:
            return "0"

        value_map = getattr(self, "_value_map", {})

        def replace_size(match):
            array_name = match.group(1)
            dim_index = match.group(2)
            arr_var = value_map.get(array_name)
            if not arr_var:
                return match.group(0)
            try:
                dim_val = int(dim_index) - 1
                if dim_val < 0:
                    dim_val = 0
                return f"PyArray_DIM({arr_var}, {dim_val})"
            except ValueError:
                dim_expr = self._dimension_c_expression(dim_index)
                return f"PyArray_DIM({arr_var}, (({dim_expr}) - 1))"

        expression = re.sub(r"size\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([0-9]+)\s*\)", replace_size, expression)

        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
        value_map_lower = {name.lower(): value for name, value in value_map.items()}
        for token in tokens:
            replacement = value_map_lower.get(token.lower())
            if replacement:
                escaped = re.escape(token)
                pattern = r"\b%s\b" % escaped
                expression = re.sub(pattern, replacement, expression, flags=re.IGNORECASE)
        return expression

    def _original_dimensions(
        self, proc: Optional[ft.Procedure], name: str
    ) -> Optional[List[str]]:
        """Look up original dimension expressions before wrapper generation."""

        if proc is None or not self.shape_hints:
            return None
        key = (proc.mod_name, getattr(proc, "type_name", None), proc.name, name)
        dims = self.shape_hints.get(key)
        if dims is not None:
            return dims
        return None

    def _is_output_argument(self, arg: ft.Argument) -> bool:
        """Return True if the argument contributes to the Python return value."""

        if self._is_hidden_argument(arg):
            return False

        intent = self._arg_intent(arg)
        return intent in {"out", "inout"}

    def _procedure_error_args(self, proc: ft.Procedure) -> Optional[Tuple[str, str]]:
        """Return error argument names when auto-raise is enabled."""

        if not self.error_num_arg or not self.error_msg_arg:
            return None

        names = {arg.name for arg in proc.arguments}
        if self.error_num_arg in names and self.error_msg_arg in names:
            return (self.error_num_arg, self.error_msg_arg)
        return None

    def _write_error_cleanup(self, proc: ft.Procedure) -> None:
        """Free allocated resources before returning on error."""

        for arg in proc.arguments:
            if arg.type.lower().startswith("character") and not self._is_array(arg):
                self.write(f"free({arg.name});")
            elif self._is_array(arg):
                if self._is_output_argument(arg):
                    self.write(f"Py_XDECREF(py_{arg.name}_arr);")
                else:
                    self.write(f"Py_XDECREF({arg.name}_arr);")
            elif self._is_derived_type(arg) and self._should_parse_argument(arg):
                self.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
                self.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                self.write(f"free({arg.name});")

    def _write_auto_raise_guard(self, proc: ft.Procedure) -> None:
        """Emit error handling guard for auto-raise logic."""

        error_args = self._procedure_error_args(proc)
        if not error_args:
            return

        num_name, msg_name = error_args
        num_var = f"{num_name}_val"
        msg_ptr = msg_name
        msg_len = f"{msg_name}_len"

        self.write("if (PyErr_Occurred()) {")
        self.indent()
        self._write_error_cleanup(proc)
        self.write("return NULL;")
        self.dedent()
        self.write("}")

        self.write(f"if ({num_var} != 0) {{")
        self.indent()
        self.write(f"f90wrap_abort_({msg_ptr}, {msg_len});")
        self._write_error_cleanup(proc)
        self.write("return NULL;")
        self.dedent()
        self.write("}")

    def _extract_dimensions(self, arg: ft.Argument) -> List[str]:
        """Extract array dimensions from argument attributes."""

        for attr in arg.attributes:
            if attr.startswith("dimension("):
                dim_str = attr[len("dimension("):-1]
                return [d.strip() for d in dim_str.split(",")]
        return []
