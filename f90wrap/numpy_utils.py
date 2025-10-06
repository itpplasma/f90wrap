"""NumPy C API utilities for Direct-C code generation."""

from __future__ import annotations

from typing import Dict


def numpy_type_from_fortran(ftype: str, kind_map: Dict[str, Dict[str, str]]) -> str:
    """Convert Fortran type to NumPy dtype enum constant."""

    ftype_lower = ftype.strip().lower()
    base, _, kind_str = ftype_lower.partition("(")
    base = base.strip()
    if base.startswith("character"):
        base = "character"

    if kind_str:
        kind_str = kind_str.rstrip(")").strip()

    # Map basic types
    if base == "integer":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "int":
                return "NPY_INT32"
            elif c_type == "long_long":
                return "NPY_INT64"
        return "NPY_INT"

    elif base == "real":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "float":
                return "NPY_FLOAT32"
            elif c_type == "double":
                return "NPY_FLOAT64"
        return "NPY_DOUBLE"

    elif base == "logical":
        return "NPY_BOOL"

    elif base == "complex":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "float_complex":
                return "NPY_COMPLEX64"
            elif c_type == "double_complex":
                return "NPY_COMPLEX128"
        return "NPY_CDOUBLE"

    elif base == "character":
        return "NPY_STRING"

    return "NPY_OBJECT"  # fallback for unknown types


def c_type_from_fortran(ftype: str, kind_map: Dict[str, Dict[str, str]]) -> str:
    """Convert Fortran type to C type string."""

    ftype_lower = ftype.strip().lower()
    base, _, kind_str = ftype_lower.partition("(")
    base = base.strip()

    if kind_str:
        kind_str = kind_str.rstrip(")").strip()

    # Map basic types
    if base == "integer":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "int":
                return "int"
            elif c_type == "long_long":
                return "long long"
        return "int"

    elif base == "real":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "float":
                return "float"
            elif c_type == "double":
                return "double"
        return "double"

    elif base == "logical":
        return "int"  # Fortran logical maps to int in C

    elif base == "complex":
        if kind_str and base in kind_map and kind_str in kind_map[base]:
            c_type = kind_map[base][kind_str]
            if c_type == "float_complex":
                return "float _Complex"
            elif c_type == "double_complex":
                return "double _Complex"
        return "double _Complex"

    elif base == "character":
        return "char"

    return "void"  # fallback


def parse_arg_format(arg_type: str) -> str:
    """Get Python argument format character for PyArg_ParseTuple."""

    ftype_lower = arg_type.strip().lower()
    base = ftype_lower.partition("(")[0].strip()
    if base.startswith("character"):
        base = "character"

    if base == "integer":
        return "i"
    elif base == "real":
        return "d"
    elif base == "logical":
        return "p"  # boolean
    elif base == "complex":
        return "D"  # complex number
    elif base == "character":
        return "s"
    else:
        return "O"  # generic object


def build_arg_format(arg_type: str) -> str:
    """Get Python build format character for Py_BuildValue."""

    ftype_lower = arg_type.strip().lower()
    base = ftype_lower.partition("(")[0].strip()
    if base.startswith("character"):
        base = "character"

    if base == "integer":
        return "i"
    elif base == "real":
        return "d"
    elif base == "logical":
        return "O"  # Use PyBool_FromLong
    elif base == "complex":
        return "D"
    elif base == "character":
        return "s"
    else:
        return "O"  # generic object
