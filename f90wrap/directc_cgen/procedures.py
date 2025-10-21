"""Procedure wrappers and return value handling for Direct-C code generation."""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from f90wrap import fortran as ft
from f90wrap.numpy_utils import build_arg_format, c_type_from_fortran, numpy_type_from_fortran
from .utils import (
    is_output_argument,
    is_array,
    is_derived_type,
    should_parse_argument,
    is_hidden_argument,
    derived_pointer_name,
    build_value_map,
    helper_symbol,
    helper_param_list,
    wrapper_name,
    extract_dimensions,
)
from .arguments import write_arg_parsing, write_arg_preparation

if TYPE_CHECKING:
    from . import DirectCGenerator


def write_wrapper_function(gen: DirectCGenerator, proc: ft.Procedure, mod_name: str) -> None:
    """Write Python C API wrapper function for a procedure."""
    proc_attributes = getattr(proc, "attributes", []) or []
    if 'destructor' in proc_attributes:
        write_destructor_wrapper(gen, proc, mod_name)
        return

    func_wrapper_name = wrapper_name(mod_name, proc)

    prev_value_map = getattr(gen, "_value_map", None)
    prev_hidden = getattr(gen, "_hidden_names", set())
    prev_hidden_lower = getattr(gen, "_hidden_names_lower", set())
    prev_proc = getattr(gen, "_current_proc", None)
    gen._value_map = build_value_map(proc)
    gen._hidden_names = {arg.name for arg in proc.arguments if is_hidden_argument(arg)}
    gen._hidden_names_lower = {name.lower() for name in gen._hidden_names}
    gen._current_proc = proc

    gen.write(
        f"static PyObject* {func_wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()

    # Parse Python arguments
    write_arg_parsing(gen, proc)

    # Prepare arguments for helper call
    write_arg_preparation(gen, proc)

    # Call the helper function
    write_helper_call(gen, proc)

    if procedure_error_args(gen, proc):
        write_auto_raise_guard(gen, proc)

    # Build return value
    write_return_value(gen, proc)

    gen.dedent()
    gen.write("}")
    gen.write("")

    gen._value_map = prev_value_map
    gen._hidden_names = prev_hidden
    gen._hidden_names_lower = prev_hidden_lower
    gen._current_proc = prev_proc


def write_destructor_wrapper(gen: DirectCGenerator, proc: ft.Procedure, mod_name: str) -> None:
    """Specialised wrapper for derived-type destructors."""
    func_wrapper_name = wrapper_name(mod_name, proc)
    helper_sym = helper_symbol(proc, gen.prefix)
    arg = proc.arguments[0]

    gen.write(
        f"static PyObject* {func_wrapper_name}(PyObject* self, PyObject* args, PyObject* kwargs)"
    )
    gen.write("{")
    gen.indent()
    gen.write("(void)self;")

    write_arg_parsing(gen, proc)
    write_arg_preparation(gen, proc)

    gen.write(f"/* Call f90wrap helper */")
    ptr_name = derived_pointer_name(arg.name)
    gen.write(f"{helper_sym}({ptr_name});")

    # Cleanup for derived handle
    gen.write(f"if ({arg.name}_sequence) {{")
    gen.indent()
    gen.write(f"Py_DECREF({arg.name}_sequence);")
    gen.dedent()
    gen.write("}")
    gen.write(f"if ({arg.name}_handle_obj) {{")
    gen.indent()
    gen.write(f"Py_DECREF({arg.name}_handle_obj);")
    gen.dedent()
    gen.write("}")
    gen.write(f"free({ptr_name});")

    gen.write("Py_RETURN_NONE;")
    gen.dedent()
    gen.write("}")
    gen.write("")


def write_helper_call(gen: DirectCGenerator, proc: ft.Procedure, helper_sym: Optional[str] = None) -> None:
    """Generate the call to the f90wrap helper function."""
    call_args = []
    helper_sym = helper_sym or helper_symbol(proc, gen.prefix)

    # Add result parameter for functions
    if isinstance(proc, ft.Function):
        c_type = c_type_from_fortran(proc.ret_val.type, gen.kind_map)
        if is_array(proc.ret_val):
            gen.write(f"{c_type}* result;")
        else:
            gen.write(f"{c_type} result;")
        call_args.append("&result")

    # Add regular arguments
    for arg in proc.arguments:
        parsed = should_parse_argument(arg)
        if is_hidden_argument(arg):
            call_args.append(f"&{arg.name}_val")
        elif is_derived_type(arg):
            ptr_name = derived_pointer_name(arg.name)
            call_args.append(ptr_name)
        elif is_array(arg):
            call_args.append(arg.name)
        elif arg.type.lower().startswith("character"):
            call_args.append(arg.name)
            call_args.append(f"{arg.name}_len")
        else:
            if parsed:
                call_args.append(arg.name)
            else:
                call_args.append(f"&{arg.name}_val")

    gen.write(f"/* Call f90wrap helper */")
    if call_args:
        gen.write(f"{helper_sym}({', '.join(call_args)});")
    else:
        gen.write(f"{helper_sym}();")

    # Check if Fortran code raised an exception via f90wrap_abort
    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    write_error_cleanup(gen, proc)
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("")


def _handle_scalar_copyback(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Handle scalar array copyback for input/output arguments."""
    for arg in proc.arguments:
        if not should_parse_argument(arg):
            continue
        if (
            not is_array(arg)
            and not is_derived_type(arg)
            and not arg.type.lower().startswith("character")
        ):
            gen.write(f"if ({arg.name}_scalar_is_array) {{")
            gen.indent()
            gen.write(f"if ({arg.name}_scalar_copyback) {{")
            gen.indent()
            gen.write(
                f"if (PyArray_CopyInto((PyArrayObject*)py_{arg.name}, {arg.name}_scalar_arr) < 0) {{"
            )
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_scalar_arr);")
            gen.write("return NULL;")
            gen.dedent()
            gen.write("}")
            gen.dedent()
            gen.write("}")
            gen.write(f"Py_DECREF({arg.name}_scalar_arr);")
            gen.dedent()
            gen.write("}")


def _handle_array_copyback(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Handle array copyback for input/output arguments."""
    for arg in proc.arguments:
        if not is_array(arg) or not should_parse_argument(arg):
            continue
        if is_output_argument(arg):
            gen.write(f"if ({arg.name}_needs_copyback) {{")
            gen.indent()
            gen.write(
                f"if (PyArray_CopyInto((PyArrayObject*)py_{arg.name}, {arg.name}_arr) < 0) {{"
            )
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_arr);")
            gen.write(f"Py_DECREF(py_{arg.name}_arr);")
            gen.write("return NULL;")
            gen.dedent()
            gen.write("}")
            gen.dedent()
            gen.write("}")
        gen.write(f"Py_DECREF({arg.name}_arr);")


def _handle_function_return(gen: DirectCGenerator, proc: ft.Function) -> None:
    """Handle return value for function procedures."""
    ret_type = proc.ret_val.type.lower()
    if is_array(proc.ret_val):
        write_array_return(gen, proc.ret_val, "result")
    elif ret_type.startswith("logical"):
        gen.write("return PyBool_FromLong(result);")
    else:
        fmt = build_arg_format(proc.ret_val.type)
        gen.write(f'return Py_BuildValue("{fmt}", result);')

    # Clean up non-output buffers for functions
    value_map = getattr(gen, "_value_map", {})
    for arg in proc.arguments:
        if arg.type.lower().startswith("character") and not is_output_argument(arg):
            cleanup_var = value_map.get(arg.name, arg.name)
            gen.write(f"free({cleanup_var});")
        elif is_array(arg) and should_parse_argument(arg):
            gen.write(f"Py_DECREF({arg.name}_arr);")
        elif is_derived_type(arg):
            ptr_name = derived_pointer_name(arg.name)
            gen.write(f"if ({arg.name}_sequence) {{")
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_sequence);")
            gen.dedent()
            gen.write("}")
            gen.write(f"if ({arg.name}_handle_obj) {{")
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_handle_obj);")
            gen.dedent()
            gen.write("}")
            gen.write(f"free({ptr_name});")


def _prepare_output_objects(gen: DirectCGenerator, output_args: List[ft.Argument]) -> List[str]:
    """Prepare Python objects for output arguments."""
    result_objects: List[str] = []

    for arg in output_args:
        if is_array(arg):
            result_objects.append(f"py_{arg.name}_arr")
            continue

        if arg.type.lower().startswith("character"):
            _prepare_character_output(gen, arg)
            # For numpy arrays, py_<arg>_obj will be Py_None, which is fine in tuple
            result_objects.append(f"py_{arg.name}_obj")
        elif is_derived_type(arg):
            _prepare_derived_output(gen, arg)
            result_objects.append(f"py_{arg.name}_obj")
        else:
            fmt = build_arg_format(arg.type)
            gen.write(
                f"PyObject* py_{arg.name}_obj = Py_BuildValue(\"{fmt}\", {arg.name}_val);"
            )
            gen.write(f"if (py_{arg.name}_obj == NULL) {{")
            gen.indent()
            gen.write("return NULL;")
            gen.dedent()
            gen.write("}")
            result_objects.append(f"py_{arg.name}_obj")

    return result_objects


def _prepare_character_output(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Prepare character output object."""
    parsed = should_parse_argument(arg)

    if parsed:
        # Check if buffer is from numpy array
        gen.write(f"PyObject* py_{arg.name}_obj = NULL;")
        gen.write(f"if ({arg.name}_is_array) {{")
        gen.indent()
        gen.write("/* Numpy array was modified in place, no return object or free needed */")
        gen.dedent()
        gen.write("} else {")
        gen.indent()

    gen.write(f"int {arg.name}_trim = {arg.name}_len;")
    gen.write(f"while ({arg.name}_trim > 0 && {arg.name}[{arg.name}_trim - 1] == ' ') {{")
    gen.indent()
    gen.write(f"--{arg.name}_trim;")
    gen.dedent()
    gen.write("}")

    if parsed:
        gen.write(
            f"py_{arg.name}_obj = PyBytes_FromStringAndSize({arg.name}, {arg.name}_trim);"
        )
        gen.write(f"free({arg.name});")
        gen.write(f"if (py_{arg.name}_obj == NULL) {{")
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.dedent()
        gen.write("}")
    else:
        gen.write(
            f"PyObject* py_{arg.name}_obj = PyBytes_FromStringAndSize({arg.name}, {arg.name}_trim);"
        )
        gen.write(f"free({arg.name});")
        gen.write(f"if (py_{arg.name}_obj == NULL) {{")
        gen.indent()
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")


def _prepare_derived_output(gen: DirectCGenerator, arg: ft.Argument) -> None:
    """Prepare derived type output object."""
    parsed = should_parse_argument(arg)
    ptr_name = derived_pointer_name(arg.name)
    gen.write(f"PyObject* py_{arg.name}_obj = PyList_New({gen.handle_size});")
    gen.write(f"if (py_{arg.name}_obj == NULL) {{")
    gen.indent()
    if parsed:
        gen.write(f"free({ptr_name});")
        gen.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
        gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"for (int i = 0; i < {gen.handle_size}; ++i) {{")
    gen.indent()
    gen.write(f"PyObject* item = PyLong_FromLong((long){ptr_name}[i]);")
    gen.write("if (item == NULL) {")
    gen.indent()
    gen.write(f"Py_DECREF(py_{arg.name}_obj);")
    if parsed:
        gen.write(f"free({ptr_name});")
        gen.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
        gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write(f"PyList_SET_ITEM(py_{arg.name}_obj, i, item);")
    gen.dedent()
    gen.write("}")
    if parsed:
        gen.write(f"if (PyObject_HasAttrString(py_{arg.name}, \"_handle\")) {{")
        gen.indent()
        gen.write(f"Py_INCREF(py_{arg.name}_obj);")
        gen.write(f"if (PyObject_SetAttrString(py_{arg.name}, \"_handle\", py_{arg.name}_obj) < 0) {{")
        gen.indent()
        gen.write(f"Py_DECREF(py_{arg.name}_obj);")
        gen.write("return NULL;")
        gen.dedent()
        gen.write("}")
        gen.dedent()
        gen.write("}")
        gen.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
        gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
        gen.write(f"free({ptr_name});")


def _cleanup_non_output_buffers(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Clean up non-output buffers."""
    value_map = getattr(gen, "_value_map", {})
    for arg in proc.arguments:
        if arg.type.lower().startswith("character") and not is_array(arg) and not is_output_argument(arg):
            cleanup_var = value_map.get(arg.name, arg.name)
            # Only free if not from numpy array
            if should_parse_argument(arg):
                gen.write(f"if (!{arg.name}_is_array) free({cleanup_var});")
            else:
                gen.write(f"free({cleanup_var});")
        elif is_derived_type(arg) and not is_output_argument(arg):
            ptr_name = derived_pointer_name(arg.name)
            gen.write(f"if ({arg.name}_sequence) {{")
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_sequence);")
            gen.dedent()
            gen.write("}")
            gen.write(f"if ({arg.name}_handle_obj) {{")
            gen.indent()
            gen.write(f"Py_DECREF({arg.name}_handle_obj);")
            gen.dedent()
            gen.write("}")
            gen.write(f"free({ptr_name});")


def write_return_value(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Build and return the Python return value."""
    output_args = [
        arg for arg in proc.arguments if is_output_argument(arg)
    ]

    _handle_scalar_copyback(gen, proc)
    _handle_array_copyback(gen, proc)

    if isinstance(proc, ft.Function):
        _handle_function_return(gen, proc)
        return

    result_objects = _prepare_output_objects(gen, output_args)
    _cleanup_non_output_buffers(gen, proc)

    if not result_objects:
        gen.write("Py_RETURN_NONE;")
        return

    # Filter out NULL objects at runtime (numpy arrays modified in place)
    gen.write("/* Build result tuple, filtering out NULL objects */")
    gen.write("int result_count = 0;")
    for name in result_objects:
        gen.write(f"if ({name} != NULL) result_count++;")

    gen.write("if (result_count == 0) {")
    gen.indent()
    gen.write("Py_RETURN_NONE;")
    gen.dedent()
    gen.write("}")

    gen.write("if (result_count == 1) {")
    gen.indent()
    for name in result_objects:
        gen.write(f"if ({name} != NULL) return {name};")
    gen.dedent()
    gen.write("}")

    gen.write("PyObject* result_tuple = PyTuple_New(result_count);")
    gen.write("if (result_tuple == NULL) {")
    gen.indent()
    for name in result_objects:
        gen.write(f"if ({name} != NULL) Py_DECREF({name});")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("int tuple_index = 0;")
    for name in result_objects:
        gen.write(f"if ({name} != NULL) {{")
        gen.indent()
        gen.write(f"PyTuple_SET_ITEM(result_tuple, tuple_index++, {name});")
        gen.dedent()
        gen.write("}")
    gen.write("return result_tuple;")


def write_array_return(gen: DirectCGenerator, ret_val: ft.Argument, var_name: str) -> None:
    """Create NumPy array from returned Fortran array with error handling."""
    numpy_type = numpy_type_from_fortran(ret_val.type, gen.kind_map)
    dims = extract_dimensions(ret_val)
    ndim = len(dims) if dims else 1

    gen.write("/* Create NumPy array from result */")
    gen.write(f"npy_intp result_dims[{ndim}];")
    for i, dim in enumerate(dims or [1]):
        gen.write(f"result_dims[{i}] = {dim};")

    gen.write(f"PyObject* result_arr = PyArray_New(&PyArray_Type, {ndim}, result_dims,")
    gen.write(f"    {numpy_type}, NULL, (void*){var_name},")
    gen.write(f"    0, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_OWNDATA, NULL);")
    gen.write("if (result_arr == NULL) {")
    gen.indent()
    gen.write("/* Free owned data buffer on failure to avoid leaks */")
    gen.write(f"free({var_name});")
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")
    gen.write("return result_arr;")


def procedure_error_args(gen: DirectCGenerator, proc: ft.Procedure) -> Optional[Tuple[str, str]]:
    """Return error argument names when auto-raise is enabled."""
    if not gen.error_num_arg or not gen.error_msg_arg:
        return None

    names = {arg.name for arg in proc.arguments}
    if gen.error_num_arg in names and gen.error_msg_arg in names:
        return (gen.error_num_arg, gen.error_msg_arg)
    return None


def write_error_cleanup(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Free allocated resources before returning on error."""
    for arg in proc.arguments:
        if arg.type.lower().startswith("character") and not is_array(arg):
            # Only free if not from numpy array
            if should_parse_argument(arg):
                gen.write(f"if (!{arg.name}_is_array) free({arg.name});")
            else:
                gen.write(f"free({arg.name});")
        elif is_array(arg):
            if is_output_argument(arg):
                gen.write(f"Py_XDECREF(py_{arg.name}_arr);")
            else:
                gen.write(f"Py_XDECREF({arg.name}_arr);")
        elif is_derived_type(arg) and should_parse_argument(arg):
            ptr_name = derived_pointer_name(arg.name)
            gen.write(f"if ({arg.name}_sequence) Py_DECREF({arg.name}_sequence);")
            gen.write(f"if ({arg.name}_handle_obj) Py_DECREF({arg.name}_handle_obj);")
            gen.write(f"free({ptr_name});")


def write_auto_raise_guard(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Emit error handling guard for auto-raise logic."""
    error_args = procedure_error_args(gen, proc)
    if not error_args:
        return

    num_name, msg_name = error_args
    num_var = f"{num_name}_val"
    msg_ptr = msg_name
    msg_len = f"{msg_name}_len"

    gen.write("if (PyErr_Occurred()) {")
    gen.indent()
    write_error_cleanup(gen, proc)
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")

    gen.write(f"if ({num_var} != 0) {{")
    gen.indent()
    gen.write(f"f90wrap_abort_({msg_ptr}, {msg_len});")
    write_error_cleanup(gen, proc)
    gen.write("return NULL;")
    gen.dedent()
    gen.write("}")


def write_helper_declaration(gen: DirectCGenerator, proc: ft.Procedure) -> None:
    """Write extern declaration for f90wrap helper functions."""
    helper_sym = helper_symbol(proc, gen.prefix)
    params = helper_param_list(proc, gen.kind_map)

    if params:
        param_str = ", ".join(params)
        gen.write(f"extern void {helper_sym}({param_str});")
    else:
        gen.write(f"extern void {helper_sym}(void);")


def write_alias_helper_declaration(
    gen: DirectCGenerator,
    alias_name: str,
    binding: ft.Binding,
    proc: ft.Procedure,
) -> None:
    """Write extern declaration for alias binding helpers."""
    from .utils import helper_param_list

    helper_sym = f"F90WRAP_F_SYMBOL({alias_name})"
    params = helper_param_list(proc, gen.kind_map)

    if params:
        param_str = ", ".join(params)
        gen.write(f"extern void {helper_sym}({param_str});")
    else:
        gen.write(f"extern void {helper_sym}(void);")