"""
Module library
Defined at library.fpp lines 5-46
"""
from __future__ import print_function, absolute_import, division
import _ExampleArray_pkg
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

def do_array_stuff(n, x, y, br, co, interface_call=False):
    """
    do_array_stuff(n, x, y, br, co)
    Defined at library.fpp lines 13-25
    
    Parameters
    ----------
    n : int32
    x : float array
    y : float array
    br : float array
    co : float array
    """
    _ExampleArray_pkg.f90wrap_library__do_array_stuff(n=n, x=x, y=y, br=br, co=co)

def only_manipulate(n, array, interface_call=False):
    """
    only_manipulate(n, array)
    Defined at library.fpp lines 27-35
    
    Parameters
    ----------
    n : int32
    array : float array
    """
    _ExampleArray_pkg.f90wrap_library__only_manipulate(n=n, array=array)

def return_array(m, n, output, interface_call=False):
    """
    return_array(m, n, output)
    Defined at library.fpp lines 37-45
    
    Parameters
    ----------
    m : int32
    n : int32
    output : int array
    """
    _ExampleArray_pkg.f90wrap_library__return_array(m=m, n=n, output=output)

def get_ia():
    """
    Element ia ftype=integer(4) pytype=int
    Defined at library.fpp line 10
    """
    return _ExampleArray_pkg.f90wrap_library__get__ia()

def set_ia(ia):
    _ExampleArray_pkg.f90wrap_library__set__ia(ia)

def get_array_iarray():
    """
    Element iarray ftype=integer(4) pytype=int
    Defined at library.fpp line 11
    """
    global iarray
    array_ndim, array_type, array_shape, array_handle = \
        _ExampleArray_pkg.f90wrap_library__array__iarray(f90wrap.runtime.empty_handle)
    array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
    if array_hash in _arrays:
        iarray = _arrays[array_hash]
    else:
        try:
            iarray = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _ExampleArray_pkg.f90wrap_library__array__iarray)
        except TypeError:
            iarray = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
        _arrays[array_handle] = iarray
    return iarray

def set_array_iarray(iarray):
    globals()['iarray'][...] = iarray


_array_initialisers = [get_array_iarray]
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "library".')

for func in _dt_array_initialisers:
    func()
