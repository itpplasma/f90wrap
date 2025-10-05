"""
Python wrapper for issue258_derived_type_attributes_direct - Direct C mode

This module re-exports types and functions from the C extension.
In direct-C mode, the C extension provides native Python types.
"""
from _issue258_derived_type_attributes_direct import *

__all__ = dir()
