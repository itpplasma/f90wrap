"""Utilities for direct-C code generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from f90wrap import fortran as ft


INTRINSIC_TYPES = {"integer", "real", "logical", "complex"}


@dataclass(frozen=True)
class ProcedureKey:
    """Identifier for a procedure within a module/type scope."""

    module: str | None
    type_name: str | None
    name: str


@dataclass
class InteropInfo:
    """Interop classification for a procedure."""

    requires_helper: bool


def bind_c_symbol(prefix: str, key: ProcedureKey) -> str:
    """Return the name of the BIND(C) shim for a procedure."""

    parts = [prefix]
    if key.module:
        parts.append(f"{key.module}__")
    if key.type_name:
        parts.append(f"{key.type_name}__")
    parts.append(key.name)
    parts.append("_c")
    return ''.join(parts)


def _argument_is_iso_c(arg: ft.Argument, kind_map: Dict[str, Dict[str, str]]) -> bool:
    """Best-effort test for ISO C compatibility."""

    if any(attr.startswith("intent(out)") for attr in arg.attributes) and arg.type.startswith("character"):
        return False

    if any(attr.startswith("value") for attr in arg.attributes):
        # value arguments are fine when the type is scalar
        pass

    if any(attr.startswith("optional") for attr in arg.attributes):
        return False

    if any(attr.startswith("pointer") or attr.startswith("allocatable")
           for attr in arg.attributes):
        return False

    # Dimension attributes are allowed only for explicit-shape arrays
    dims = [attr for attr in arg.attributes if attr.startswith("dimension")]
    if dims:
        dim_expr = dims[0][len("dimension("):-1]
        if any("*" in part for part in dim_expr.split(",")):
            return False

    ftype = arg.type.strip().lower()
    if ftype.startswith("type(") or ftype.startswith("class("):
        return False

    base, _, kind = ftype.partition("(")
    base = base.strip()
    if base not in INTRINSIC_TYPES:
        return False

    if kind:
        kind = kind.rstrip(") ")
        # Map via kind_map when possible
        if base in kind_map and kind in kind_map[base]:
            c_type = kind_map[base][kind]
            if c_type not in {"int", "float", "double", "long_long"}:
                return False
        else:
            # Unknown kind
            return False

    return True


def _procedure_requires_helper(proc: ft.Procedure, kind_map: Dict[str, Dict[str, str]]) -> bool:
    """Determine if procedure needs a classic f90wrap helper for direct-C mode."""

    # If procedure has any attributes (e.g. recursive), keep helper
    if proc.attributes:
        return True

    for arg in proc.arguments:
        if not _argument_is_iso_c(arg, kind_map):
            return True

    if isinstance(proc, ft.Function):
        if not _argument_is_iso_c(proc.ret_val, kind_map):
            return True

    return False


def analyse_interop(tree: ft.Root, kind_map: Dict[str, Dict[str, str]]) -> Dict[ProcedureKey, InteropInfo]:
    """Analyse the transformed tree and flag which procedures need helpers."""

    classification: Dict[ProcedureKey, InteropInfo] = {}

    def record(procs: Iterable[ft.Procedure]):
        for proc in procs:
            key = ProcedureKey(proc.mod_name, getattr(proc, 'type_name', None), proc.name)
            classification[key] = InteropInfo(
                requires_helper=_procedure_requires_helper(proc, kind_map)
            )

    for module in tree.modules:
        record(module.procedures)
    record(getattr(tree, 'procedures', []))

    return classification
