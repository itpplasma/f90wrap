"""
Python wrapper for optional_string_direct - Direct C mode

This module re-exports types and functions from the C extension.
In direct-C mode, the C extension provides native Python types.
"""
from _optional_string_direct import *

__all__ = dir()
