"""
Python wrapper for docstring_direct - Direct C mode

This module re-exports types and functions from the C extension.
In direct-C mode, the C extension provides native Python types.
"""
from _docstring_direct import *

__all__ = dir()
