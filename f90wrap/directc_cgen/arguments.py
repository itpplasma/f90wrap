"""Argument parsing and preparation for Direct-C code generation."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from f90wrap import fortran as ft
from f90wrap.numpy_utils import c_type_from_fortran, numpy_type_from_fortran, parse_arg_format
from .utils import (
    is_hidden_argument,
    is_array,
    is_derived_type,
    should_parse_argument,
    arg_intent,
    is_optional,
    is_output_argument,
    derived_pointer_name,
    character_length_expr,
    extract_dimensions,
    dimension_c_expression,
    original_dimensions,
)

if TYPE_CHECKING:
    from . import DirectCGenerator


def write_arg_parsing(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Generate PyArg_ParseTuple code for procedure arguments."""
    hidden_args = [arg for arg in proc.arguments if is_hidden_argument(arg)]
    for hidden in hidden_args:
        gen.write(f"int {hidden.name}_val = 0;")

    if not proc.arguments:
        return

    format_parts: List[str] = []
    parse_vars: List[str] = []
    kw_names: List[str] = []
    optional_started = False

    for arg in proc.arguments:
        if is_hidden_argument(arg):
            # Hidden arguments (like f90wrap_n0 dimension vars) need to be in kwlist
            # so Python can pass them, but they're optional and use integer format
            if not optional_started:
                format_parts.append("|")
                optional_started = True
            format_parts.append("i")
            parse_vars.append(f"&{arg.name}_val")
            kw_names.append(f'"{arg.name}"')
            continue

        intent = arg_intent(arg)
        optional = is_optional(arg)
        should_parse = should_parse_argument(arg)

        if not should_parse:
            if not is_array(arg) and not is_derived_type(arg) and not arg.type.lower().startswith("character"):
                c_type = c_type_from_fortran(arg.type, gen.kind_map)
                gen.write(f"{c_type} {arg.name}_val = 0;")
            continue

        if optional and not optional_started:
            format_parts.append("|")
            optional_started = True

        if is_derived_type(arg):
            format_parts.append("O")
            gen.write(f"PyObject* py_{arg.name} = NULL;")
            parse_vars.append(f"&py_{arg.name}")
        elif is_array(arg):
            format_parts.append("O")
            gen.write(f"PyObject* py_{arg.name} = NULL;")
            parse_vars.append(f"&py_{arg.name}")
        elif arg.type.lower().startswith("character"):
            if should_parse_argument(arg):
                format_parts.append("O")
                if optional or intent != "in":
                    gen.write(f"PyObject* py_{arg.name} = Py_None;")
                else:
                    gen.write(f"PyObject* py_{arg.name} = NULL;")
                parse_vars.append(f"&py_{arg.name}")
            else:
                format_parts.append("O")
                if optional:
                    gen.write(f"PyObject* py_{arg.name} = Py_None;")
                else:
                    gen.write(f"PyObject* py_{arg.name} = NULL;")
                parse_vars.append(f"&py_{arg.name}")
        else:
            c_type = c_type_from_fortran(arg.type, gen.kind_map)
            format_parts.append("O")
            if optional:
                gen.write(f"PyObject* py_{arg.name} = Py_None;")
            else:
                gen.write(f"PyObject* py_{arg.name} = NULL;")
            parse_vars.append(f"&py_{arg.name}")
            gen.write(f"{c_type} {arg.name}_val = 0;")
            gen.write(f"PyArrayObject* {arg.name}_scalar_arr = NULL;")
            gen.write(f"int {arg.name}_scalar_copyback = 0;")
            gen.write(f"int {arg.name}_scalar_is_array = 0;")

        kw_names.append(f'"{arg.name}"')

    if parse_vars:
        format_str = "".join(format_parts) if format_parts else ""
        kwlist = ", ".join(kw_names) if kw_names else ""
        gen.write(f"static char *kwlist[] = {{{kwlist}{', ' if kwlist else ''}NULL}};")
        gen.write("")
        gen.write(
            f'if (!PyArg_ParseTupleAndKeywords(args, kwargs, "{format_str}", kwlist, '
            f"{', '.join(parse_vars)})) {{"
        )
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write("")


def write_arg_preparation(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Prepare arguments for helper function call."""
    for arg in proc.arguments:
        intent = arg_intent(arg)
        optional = is_optional(arg)
        parsed = should_parse_argument(arg)

        if is_array(arg):
            declare_array_storage(gen, arg)
            if parsed:
                if optional:
                    # Allow None for optional arrays
                    gen.write(f"if (py_{arg.name} != NULL && py_{arg.name} != Py_None) {{")
                    gen.indent()
                write_array_preparation(gen, arg)
                if optional:
                    gen.dedent()
                    gen.write("}")
            else:
                prepare_output_array(gen, arg)
        elif arg.type.lower().startswith("character"):
            prepare_character_argument(gen, arg, intent, optional)
        elif is_derived_type(arg):
            if should_parse_argument(arg):
                if optional:
                    ptr_name = derived_pointer_name(arg.name)
                    # Declare variables that cleanup code expects
                    gen.write(f"PyObject* {arg.name}_handle_obj = NULL;")
                    gen.write(f"PyObject* {arg.name}_sequence = NULL;")
                    gen.write(f"Py_ssize_t {arg.name}_handle_len = 0;")
                    gen.write(f"int* {ptr_name} = NULL;")

                    gen.write(f"if (py_{arg.name} != Py_None) {{")
                    gen.indent()
                    # Extract handle without declaring variables
                    gen.write(f"if (PyObject_HasAttrString(py_{arg.name}, \"_handle\")) {{")
                    gen.indent()
                    gen.write(f"{arg.name}_handle_obj = PyObject_GetAttrString(py_{arg.name}, \"_handle\");")
                    gen.write(f"if ({arg.name}_handle_obj == NULL) {{")
                    gen.indent()
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.write(
                        f"{arg.name}_sequence = PySequence_Fast({arg.name}_handle_obj, \"Failed to access handle sequence\");"
                    )
                    gen.write(f"if ({arg.name}_sequence == NULL) {{")
                    gen.indent()
                    gen.write(f"Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.dedent()
                    gen.write(f"}} else if (PySequence_Check(py_{arg.name})) {{")
                    gen.indent()
                    gen.write(
                        f"{arg.name}_sequence = PySequence_Fast(py_{arg.name}, \"Argument {arg.name} must be a handle sequence\");"
                    )
                    gen.write(f"if ({arg.name}_sequence == NULL) {{")
                    gen.indent()
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.dedent()
                    gen.write("} else {")
                    gen.indent()
                    gen.write(
                        f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a Fortran derived-type instance");'
                    )
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")

                    gen.write(
                        f"{arg.name}_handle_len = PySequence_Fast_GET_SIZE({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_len != {gen.handle_size}) {{")
                    gen.indent()
                    gen.write(
                        f'PyErr_SetString(PyExc_ValueError, "Argument {arg.name} has an invalid handle length");'
                    )
                    gen.write(f"Py_DECREF({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")

                    gen.write(f"{ptr_name} = (int*)malloc(sizeof(int) * {arg.name}_handle_len);")
                    gen.write(f"if ({ptr_name} == NULL) {{")
                    gen.indent()
                    gen.write("PyErr_NoMemory();")
                    gen.write(f"Py_DECREF({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")

                    gen.write(f"for (Py_ssize_t i = 0; i < {arg.name}_handle_len; ++i) {{")
                    gen.indent()
                    gen.write(
                        f"PyObject* item = PySequence_Fast_GET_ITEM({arg.name}_sequence, i);")
                    gen.write("if (item == NULL) {")
                    gen.indent()
                    gen.write(f"free({ptr_name});")
                    gen.write(f"Py_DECREF({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.write(f"{ptr_name}[i] = (int)PyLong_AsLong(item);")
                    gen.write("if (PyErr_Occurred()) {")
                    gen.indent()
                    gen.write(f"free({ptr_name});")
                    gen.write(f"Py_DECREF({arg.name}_sequence);")
                    gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
                    gen.write("return NULL;")
                    gen.dedent()
                    gen.write("}")
                    gen.dedent()
                    gen.write("}")
                    gen.write(f"(void){arg.name}_handle_len;  /* suppress unused warnings when unchanged */")

                    gen.dedent()
                    gen.write("}")
                else:
                    write_derived_preparation(gen, arg)
            else:
                gen.write(f"int {arg.name}[{gen.handle_size}] = {{0}};")
        else:
            prepare_scalar_argument(gen, arg, intent, optional)


def _write_scalar_array_handling(gen: DirectCGenerator, arg: ft.Argument, c_type: str, numpy_type: str) -> None:
    """Helper to write array handling for scalar arguments."""
    gen.write(f"if (PyArray_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write(
        f"{arg.name}_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(\n"
        f"    py_{arg.name}, {numpy_type}, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);")
    gen.write(f"if ({arg.name}_scalar_arr == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"if (PyArray_SIZE({arg.name}_scalar_arr) != 1) {{")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_ValueError, "Argument {arg.name} must have exactly one element");'
    )
    gen.write(f"Py_DECREF({arg.name}_scalar_arr);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{arg.name}_scalar_is_array = 1;")
    gen.write(f"{arg.name} = ({c_type}*)PyArray_DATA({arg.name}_scalar_arr);")
    gen.write(f"{arg.name}_val = {arg.name}[0];")
    gen.write(
        f"if (PyArray_DATA({arg.name}_scalar_arr) != PyArray_DATA((PyArrayObject*)py_{arg.name}) || "
        f"PyArray_TYPE({arg.name}_scalar_arr) != PyArray_TYPE((PyArrayObject*)py_{arg.name})) {{"
    )
    gen.indent()
    gen.write(f"{arg.name}_scalar_copyback = 1;")
    gen.dedent()
    gen.write("}")
    gen.dedent()


def _write_scalar_number_handling(gen: DirectCGenerator, arg: ft.Argument, c_type: str) -> None:
    """Helper to write number handling for scalar arguments."""
    gen.write(f"}} else if (PyNumber_Check(py_{arg.name})) {{")
    gen.indent()
    fmt = parse_arg_format(arg.type)
    if fmt in {"i", "l", "h", "I"}:
        gen.write(f"{arg.name}_val = ({c_type})PyLong_AsLong(py_{arg.name});")
    elif fmt in {"k", "K"}:
        gen.write(f"{arg.name}_val = ({c_type})PyLong_AsUnsignedLong(py_{arg.name});")
    elif fmt in {"L", "q"}:
        gen.write(f"{arg.name}_val = ({c_type})PyLong_AsLongLong(py_{arg.name});")
    elif fmt in {"Q"}:
        gen.write(f"{arg.name}_val = ({c_type})PyLong_AsUnsignedLongLong(py_{arg.name});")
    elif fmt in {"d", "f"}:
        gen.write(f"{arg.name}_val = ({c_type})PyFloat_AsDouble(py_{arg.name});")
    else:
        gen.write(
            f'PyErr_SetString(PyExc_TypeError, "Unsupported argument {arg.name}");'
        )
        gen.write("return NULL;")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("} else {")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a scalar number or NumPy array");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")


def prepare_scalar_argument(gen: DirectCGenerator, arg: ft.Argument, intent: str, optional: bool) -> None:
    """Prepare scalar argument values."""
    c_type = c_type_from_fortran(arg.type, gen.kind_map)
    numpy_type = numpy_type_from_fortran(arg.type, gen.kind_map)

    if not should_parse_argument(arg):
        return

    gen.write(f"{c_type}* {arg.name} = &{arg.name}_val;")

    if optional:
        gen.write(f"if (py_{arg.name} == Py_None) {{")
        gen.indent()
        gen.write(f"{arg.name}_val = 0;")
        gen.dedent()
        gen.write("} else {")
        gen.indent()

    _write_scalar_array_handling(gen, arg, c_type, numpy_type)
    _write_scalar_number_handling(gen, arg, c_type)

    if optional:
        gen.dedent()
        gen.write("}")


def _prepare_character_none_case(gen: DirectCGenerator, arg: ft.Argument, intent: str, optional: bool, default_len: str) -> None:
    """Helper to handle None case for character arguments."""
    gen.write(f"if (py_{arg.name} == Py_None) {{")
    gen.indent()
    if optional or intent != "in":
        gen.write(f"{arg.name}_len = {default_len};")
        gen.write(f"if ({arg.name}_len <= 0) {{")
        gen.indent()
        gen.write(
            f'PyErr_SetString(PyExc_ValueError, "Character length for {arg.name} must be positive");'
        )
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"{arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
        gen.write(f"if ({arg.name} == NULL) {{")
        gen.indent()
        gen.write("PyErr_NoMemory();")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"memset({arg.name}, ' ', {arg.name}_len);")
        gen.write(f"{arg.name}[{arg.name}_len] = '\\0';")
    else:
        gen.write(
            f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} cannot be None");'
        )
        gen.write("return NULL;")
    gen.dedent()


def _prepare_character_string_case(gen: DirectCGenerator, arg: ft.Argument, is_output: bool) -> None:
    """Helper to handle string/bytes case for character arguments."""
    gen.write("} else {")
    gen.indent()
    gen.write(f"PyObject* {arg.name}_bytes = NULL;")
    gen.write(f"if (PyArray_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write("/* Handle numpy array - extract buffer for in-place modification */")
    gen.write(f"PyArrayObject* {arg.name}_arr = (PyArrayObject*)py_{arg.name};")
    gen.write(f"if (PyArray_TYPE({arg.name}_arr) != NPY_STRING) {{")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a string array");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{arg.name}_len = (int)PyArray_ITEMSIZE({arg.name}_arr);")
    gen.write(f"{arg.name} = (char*)PyArray_DATA({arg.name}_arr);")
    if is_output:
        gen.write(f"{arg.name}_is_array = 1;")
    gen.dedent()
    gen.write(f"}} else if (PyBytes_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write(f"{arg.name}_bytes = py_{arg.name};")
    gen.write(f"Py_INCREF({arg.name}_bytes);")
    gen.dedent()
    gen.write(f"}} else if (PyUnicode_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write(f"{arg.name}_bytes = PyUnicode_AsUTF8String(py_{arg.name});")
    gen.write(f"if ({arg.name}_bytes == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("} else {")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be str, bytes, or numpy array");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"if ({arg.name}_bytes != NULL) {{")
    gen.indent()
    gen.write(f"{arg.name}_len = (int)PyBytes_GET_SIZE({arg.name}_bytes);")
    gen.write(f"{arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
    gen.write(f"if ({arg.name} == NULL) {{")
    gen.indent()
    gen.write(f"Py_DECREF({arg.name}_bytes);")
    gen.write("PyErr_NoMemory();")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(
        f"memcpy({arg.name}, PyBytes_AS_STRING({arg.name}_bytes), (size_t){arg.name}_len);"
    )
    gen.write(f"{arg.name}[{arg.name}_len] = '\\0';")
    gen.write(f"Py_DECREF({arg.name}_bytes);")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")


def prepare_character_argument(gen: DirectCGenerator, arg: ft.Argument, intent: str, optional: bool) -> None:
    """Allocate and populate character buffers."""
    type_spec = arg.type
    default_len = character_length_expr(type_spec) or "1024"

    if should_parse_argument(arg):
        gen.write(f"int {arg.name}_len = 0;")
        gen.write(f"char* {arg.name} = NULL;")
        # Track if buffer is from numpy array (don't free it)
        gen.write(f"int {arg.name}_is_array = 0;")
        is_output = is_output_argument(arg)
        _prepare_character_none_case(gen, arg, intent, optional, default_len)
        _prepare_character_string_case(gen, arg, True)  # Always set flag if numpy array
    else:
        gen.write(f"int {arg.name}_len = {default_len};")
        gen.write(f"if ({arg.name}_len <= 0) {{")
        gen.indent()
        gen.write(
            f'PyErr_SetString(PyExc_ValueError, "Character length for {arg.name} must be positive");'
        )
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"char* {arg.name} = (char*)malloc((size_t){arg.name}_len + 1);")
        gen.write(f"if ({arg.name} == NULL) {{")
        gen.indent()
        gen.write("PyErr_NoMemory();")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.write(f"memset({arg.name}, ' ', {arg.name}_len);")
        gen.write(f"{arg.name}[{arg.name}_len] = '\\0';")


def declare_array_storage(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Declare local variables needed for array arguments."""
    c_type = c_type_from_fortran(arg.type, gen.kind_map)
    gen.write(f"PyArrayObject* {arg.name}_arr = NULL;")
    if is_output_argument(arg):
        gen.write(f"PyObject* py_{arg.name}_arr = NULL;")
        if should_parse_argument(arg):
            gen.write(f"int {arg.name}_needs_copyback = 0;")
    gen.write(f"{c_type}* {arg.name} = NULL;")
    # For character arrays, declare element length variable
    if arg.type.lower().startswith("character"):
        gen.write(f"int {arg.name}_elem_len = 0;")


def write_array_preparation(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Extract array data from NumPy array."""
    numpy_type = numpy_type_from_fortran(arg.type, gen.kind_map)
    c_type = c_type_from_fortran(arg.type, gen.kind_map)

    gen.write(f"/* Extract {arg.name} array data */")
    gen.write(f"if (!PyArray_Check(py_{arg.name})) {{")
    gen.indent()
    gen.write(f'PyErr_SetString(PyExc_TypeError, "Argument {arg.name} must be a NumPy array");')
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    # Convert to Fortran-contiguous if needed
    gen.write(f"{arg.name}_arr = (PyArrayObject*)PyArray_FROM_OTF(")
    gen.write(
        f"    py_{arg.name}, {numpy_type}, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);")
    gen.write(f"if ({arg.name}_arr == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"{arg.name} = ({c_type}*)PyArray_DATA({arg.name}_arr);")

    # For character arrays, get element length for Fortran calling convention
    if arg.type.lower().startswith("character"):
        len_var = f"{arg.name}_elem_len"
        gen.write(f"{len_var} = (int)PyArray_ITEMSIZE({arg.name}_arr);")

    # Get dimensions if needed
    dims = extract_dimensions(arg)
    if dims:
        for i in range(len(dims)):
            gen.write(f"int n{i}_{arg.name} = (int)PyArray_DIM({arg.name}_arr, {i});")
        for i, dim in enumerate(dims):
            dim_name = dim.strip()
            if dim_name and dim_name.startswith("f90wrap_"):
                gen.write(f"{dim_name}_val = n{i}_{arg.name};")

    if is_output_argument(arg):
        gen.write(f"Py_INCREF(py_{arg.name});")
        gen.write(f"py_{arg.name}_arr = py_{arg.name};")
        if should_parse_argument(arg):
            gen.write(
                f"if (PyArray_DATA({arg.name}_arr) != PyArray_DATA((PyArrayObject*)py_{arg.name}) || "
                f"PyArray_TYPE({arg.name}_arr) != PyArray_TYPE((PyArrayObject*)py_{arg.name})) {{"
            )
            gen.indent()
            gen.write(f"{arg.name}_needs_copyback = 1;")
            gen.dedent()
            gen.write("}")

    gen.write("")


def prepare_output_array(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Allocate NumPy array for output-only arguments."""
    proc = getattr(gen, "_current_proc", None)
    value_map = getattr(gen, "_value_map", {})
    trans_dims = extract_dimensions(arg)
    orig_dims = original_dimensions(proc, arg.name, gen.shape_hints)
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

        expr = dimension_c_expression(source_expr, value_map)
        size_var = f"{arg.name}_dim_{index}"
        gen.write(f"npy_intp {size_var} = (npy_intp)({expr});")
        gen.write(f"if ({size_var} <= 0) {{")
        gen.indent()
        gen.write(
            f'PyErr_SetString(PyExc_ValueError, "Dimension for {arg.name} must be positive");'
        )
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        if trans_token:
            token = trans_token.strip()
            hidden_lower = getattr(gen, "_hidden_names_lower", set())
            token_lower = token.lower()
            if token_lower in hidden_lower:
                replacement = value_map.get(token) or value_map.get(token_lower)
                if replacement:
                    gen.write(f"{replacement} = (int){size_var};")
        dim_vars.append(size_var)

    dims_array = f"{arg.name}_dims"
    gen.write(f"npy_intp {dims_array}[{len(dim_vars)}] = {{{', '.join(dim_vars)}}};")
    numpy_type = numpy_type_from_fortran(arg.type, gen.kind_map)
    gen.write(
        f"py_{arg.name}_arr = PyArray_SimpleNew({len(dim_vars)}, {dims_array}, {numpy_type});"
    )
    gen.write(f"if (py_{arg.name}_arr == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{arg.name}_arr = (PyArrayObject*)py_{arg.name}_arr;")
    c_type = c_type_from_fortran(arg.type, gen.kind_map)
    gen.write(f"{arg.name} = ({c_type}*)PyArray_DATA({arg.name}_arr);")
    gen.write("")


def write_derived_preparation(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Extract derived-type handle from Python object."""
    name = arg.name
    ptr_name = derived_pointer_name(name)
    gen.write(f"PyObject* {name}_handle_obj = NULL;")
    gen.write(f"PyObject* {name}_sequence = NULL;")
    gen.write(f"Py_ssize_t {name}_handle_len = 0;")

    gen.write(f"if (PyObject_HasAttrString(py_{name}, \"_handle\")) {{")
    gen.indent()
    gen.write(f"{name}_handle_obj = PyObject_GetAttrString(py_{name}, \"_handle\");")
    gen.write(f"if ({name}_handle_obj == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(
        f"{name}_sequence = PySequence_Fast({name}_handle_obj, \"Failed to access handle sequence\");"
    )
    gen.write(f"if ({name}_sequence == NULL) {{")
    gen.indent()
    gen.write(f"Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write(f"}} else if (PySequence_Check(py_{name})) {{")
    gen.indent()
    gen.write(
        f"{name}_sequence = PySequence_Fast(py_{name}, \"Argument {name} must be a handle sequence\");"
    )
    gen.write(f"if ({name}_sequence == NULL) {{")
    gen.indent()
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("} else {")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_TypeError, "Argument {name} must be a Fortran derived-type instance");'
    )
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(
        f"{name}_handle_len = PySequence_Fast_GET_SIZE({name}_sequence);")
    gen.write(f"if ({name}_handle_len != {gen.handle_size}) {{")
    gen.indent()
    gen.write(
        f'PyErr_SetString(PyExc_ValueError, "Argument {name} has an invalid handle length");'
    )
    gen.write(f"Py_DECREF({name}_sequence);")
    gen.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"int* {ptr_name} = (int*)malloc(sizeof(int) * {name}_handle_len);")
    gen.write(f"if ({ptr_name} == NULL) {{")
    gen.indent()
    gen.write("PyErr_NoMemory();")
    gen.write(f"Py_DECREF({name}_sequence);")
    gen.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"for (Py_ssize_t i = 0; i < {name}_handle_len; ++i) {{")
    gen.indent()
    gen.write(
        f"PyObject* item = PySequence_Fast_GET_ITEM({name}_sequence, i);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write(f"free({ptr_name});")
    gen.write(f"Py_DECREF({name}_sequence);")
    gen.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"{ptr_name}[i] = (int)PyLong_AsLong(item);")
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    gen.write(f"free({ptr_name});")
    gen.write(f"Py_DECREF({name}_sequence);")
    gen.write(f"if ({name}_handle_obj) Py_DECREF({name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.dedent()
    gen.write("}")
    gen.write(f"(void){name}_handle_len;  /* suppress unused warnings when unchanged */")
    gen.write("")