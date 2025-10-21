"""
Module string_io
Defined at string_io.fpp lines 5-49
"""
from __future__ import print_function, absolute_import, division
import _ExampleStrings_pkg
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

def func_generate_string(n, interface_call=False):
    """
    stringout = func_generate_string(n)
    Defined at string_io.fpp lines 9-16
    
    Parameters
    ----------
    n : int32
    
    Returns
    -------
    stringout : str
    """
    stringout = _ExampleStrings_pkg.f90wrap_string_io__func_generate_string(n=n)
    return stringout

def func_return_string(interface_call=False):
    """
    stringout = func_return_string()
    Defined at string_io.fpp lines 18-20
    
    Returns
    -------
    stringout : str
    """
    stringout = _ExampleStrings_pkg.f90wrap_string_io__func_return_string()
    return stringout

def generate_string(n, interface_call=False):
    """
    stringout = generate_string(n)
    Defined at string_io.fpp lines 22-30
    
    Parameters
    ----------
    n : int32
    
    Returns
    -------
    stringout : str
    """
    stringout = _ExampleStrings_pkg.f90wrap_string_io__generate_string(n=n)
    return stringout

def return_string(interface_call=False):
    """
    stringout = return_string()
    Defined at string_io.fpp lines 32-34
    
    Returns
    -------
    stringout : str
    """
    stringout = _ExampleStrings_pkg.f90wrap_string_io__return_string()
    return stringout

def set_global_string(n, newstring, interface_call=False):
    """
    set_global_string(n, newstring)
    Defined at string_io.fpp lines 36-39
    
    Parameters
    ----------
    n : int32
    newstring : str
    """
    _ExampleStrings_pkg.f90wrap_string_io__set_global_string(n=n, \
        newstring=newstring)

def inout_string(n, stringinout, interface_call=False):
    """
    inout_string(n, stringinout)
    Defined at string_io.fpp lines 41-48
    
    Parameters
    ----------
    n : int32
    stringinout : str
    """
    _ExampleStrings_pkg.f90wrap_string_io__inout_string(n=n, \
        stringinout=stringinout)

def get_global_string():
    """
    Element global_string ftype=character(512) pytype=str
    Defined at string_io.fpp line 7
    """
    return _ExampleStrings_pkg.f90wrap_string_io__get__global_string()

def set_global_string_(global_string):
    _ExampleStrings_pkg.f90wrap_string_io__set__global_string(global_string)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "string_io".')

for func in _dt_array_initialisers:
    func()
