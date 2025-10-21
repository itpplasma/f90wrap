"""
Module gaussian
Defined at ./Source/BasicDefs/aa1_modules.fpp lines 35-44
"""
from __future__ import print_function, absolute_import, division
import _mockdtpkg
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

def get_ng():
    """
    Element ng ftype=integer         pytype=int
    Defined at ./Source/BasicDefs/aa1_modules.fpp line 41
    """
    return _mockdtpkg.f90wrap_gaussian__get__ng()

def set_ng(ng):
    _mockdtpkg.f90wrap_gaussian__set__ng(ng)

def get_ngpsi():
    """
    Element ngpsi ftype=integer         pytype=int
    Defined at ./Source/BasicDefs/aa1_modules.fpp line 41
    """
    return _mockdtpkg.f90wrap_gaussian__get__ngpsi()

def set_ngpsi(ngpsi):
    _mockdtpkg.f90wrap_gaussian__set__ngpsi(ngpsi)

def get_array_ecinv():
    """
    Element ecinv ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa1_modules.fpp line 43
    """
    global ecinv
    array_ndim, array_type, array_shape, array_handle = \
        _mockdtpkg.f90wrap_gaussian__array__ecinv(f90wrap.runtime.empty_handle)
    array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
    if array_hash in _arrays:
        ecinv = _arrays[array_hash]
    else:
        try:
            ecinv = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mockdtpkg.f90wrap_gaussian__array__ecinv)
        except TypeError:
            ecinv = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
        _arrays[array_handle] = ecinv
    return ecinv

def set_array_ecinv(ecinv):
    globals()['ecinv'][...] = ecinv

def get_array_xg():
    """
    Element xg ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa1_modules.fpp line 43
    """
    global xg
    array_ndim, array_type, array_shape, array_handle = \
        _mockdtpkg.f90wrap_gaussian__array__xg(f90wrap.runtime.empty_handle)
    array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
    if array_hash in _arrays:
        xg = _arrays[array_hash]
    else:
        try:
            xg = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mockdtpkg.f90wrap_gaussian__array__xg)
        except TypeError:
            xg = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
        _arrays[array_handle] = xg
    return xg

def set_array_xg(xg):
    globals()['xg'][...] = xg

def get_array_fcinv():
    """
    Element fcinv ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa1_modules.fpp line 43
    """
    global fcinv
    array_ndim, array_type, array_shape, array_handle = \
        _mockdtpkg.f90wrap_gaussian__array__fcinv(f90wrap.runtime.empty_handle)
    array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
    if array_hash in _arrays:
        fcinv = _arrays[array_hash]
    else:
        try:
            fcinv = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mockdtpkg.f90wrap_gaussian__array__fcinv)
        except TypeError:
            fcinv = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
        _arrays[array_handle] = fcinv
    return fcinv

def set_array_fcinv(fcinv):
    globals()['fcinv'][...] = fcinv

def get_array_wg():
    """
    Element wg ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa1_modules.fpp line 43
    """
    global wg
    array_ndim, array_type, array_shape, array_handle = \
        _mockdtpkg.f90wrap_gaussian__array__wg(f90wrap.runtime.empty_handle)
    array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
    if array_hash in _arrays:
        wg = _arrays[array_hash]
    else:
        try:
            wg = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mockdtpkg.f90wrap_gaussian__array__wg)
        except TypeError:
            wg = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
        _arrays[array_handle] = wg
    return wg

def set_array_wg(wg):
    globals()['wg'][...] = wg

def get_array_xgpsi():
    """
    Element xgpsi ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa1_modules.fpp line 45
    """
    global xgpsi
    array_ndim, array_type, array_shape, array_handle = \
        _mockdtpkg.f90wrap_gaussian__array__xgpsi(f90wrap.runtime.empty_handle)
    array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
    if array_hash in _arrays:
        xgpsi = _arrays[array_hash]
    else:
        try:
            xgpsi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mockdtpkg.f90wrap_gaussian__array__xgpsi)
        except TypeError:
            xgpsi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
        _arrays[array_handle] = xgpsi
    return xgpsi

def set_array_xgpsi(xgpsi):
    globals()['xgpsi'][...] = xgpsi

def get_array_wgpsi():
    """
    Element wgpsi ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa1_modules.fpp line 45
    """
    global wgpsi
    array_ndim, array_type, array_shape, array_handle = \
        _mockdtpkg.f90wrap_gaussian__array__wgpsi(f90wrap.runtime.empty_handle)
    array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
    if array_hash in _arrays:
        wgpsi = _arrays[array_hash]
    else:
        try:
            wgpsi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                    f90wrap.runtime.empty_handle,
                                    _mockdtpkg.f90wrap_gaussian__array__wgpsi)
        except TypeError:
            wgpsi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
        _arrays[array_handle] = wgpsi
    return wgpsi

def set_array_wgpsi(wgpsi):
    globals()['wgpsi'][...] = wgpsi


_array_initialisers = [get_array_ecinv, get_array_xg, get_array_fcinv, \
    get_array_wg, get_array_xgpsi, get_array_wgpsi]
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "gaussian".')

for func in _dt_array_initialisers:
    func()
