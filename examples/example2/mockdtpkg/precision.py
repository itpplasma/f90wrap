"""
Module precision
Defined at ./Source/BasicDefs/aa0_typelist.fpp lines 9-23
"""
from __future__ import print_function, absolute_import, division
import _mockdtpkg
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_mockdtpkg = _SafeDirectCExecutor(_mockdtpkg, module_import_name='_mockdtpkg')

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

def get_rdp():
    """
    Element rdp ftype=integer pytype=int
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 19
    """
    return _mockdtpkg.f90wrap_precision__get__rdp()

rdp = get_rdp()

def get_zero():
    """
    Element zero ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__zero()

def set_zero(zero):
    _mockdtpkg.f90wrap_precision__set__zero(zero)

def get_one():
    """
    Element one ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__one()

def set_one(one):
    _mockdtpkg.f90wrap_precision__set__one(one)

def get_half():
    """
    Element half ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__half()

def set_half(half):
    _mockdtpkg.f90wrap_precision__set__half(half)

def get_two():
    """
    Element two ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__two()

def set_two(two):
    _mockdtpkg.f90wrap_precision__set__two(two)

def get_three():
    """
    Element three ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__three()

def set_three(three):
    _mockdtpkg.f90wrap_precision__set__three(three)

def get_four():
    """
    Element four ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__four()

def set_four(four):
    _mockdtpkg.f90wrap_precision__set__four(four)

def get_six():
    """
    Element six ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__six()

def set_six(six):
    _mockdtpkg.f90wrap_precision__set__six(six)

def get_eight():
    """
    Element eight ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__eight()

def set_eight(eight):
    _mockdtpkg.f90wrap_precision__set__eight(eight)

def get_pi():
    """
    Element pi ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__pi()

def set_pi(pi):
    _mockdtpkg.f90wrap_precision__set__pi(pi)

def get_twopi():
    """
    Element twopi ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__twopi()

def set_twopi(twopi):
    _mockdtpkg.f90wrap_precision__set__twopi(twopi)

def get_d2r():
    """
    Element d2r ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__d2r()

def set_d2r(d2r):
    _mockdtpkg.f90wrap_precision__set__d2r(d2r)

def get_r2d():
    """
    Element r2d ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__r2d()

def set_r2d(r2d):
    _mockdtpkg.f90wrap_precision__set__r2d(r2d)

def get_xk2fps():
    """
    Element xk2fps ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__xk2fps()

def set_xk2fps(xk2fps):
    _mockdtpkg.f90wrap_precision__set__xk2fps(xk2fps)

def get_lb2n():
    """
    Element lb2n ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__lb2n()

def set_lb2n(lb2n):
    _mockdtpkg.f90wrap_precision__set__lb2n(lb2n)

def get_ftlb2nm():
    """
    Element ftlb2nm ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__ftlb2nm()

def set_ftlb2nm(ftlb2nm):
    _mockdtpkg.f90wrap_precision__set__ftlb2nm(ftlb2nm)

def get_one80():
    """
    Element one80 ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__one80()

def set_one80(one80):
    _mockdtpkg.f90wrap_precision__set__one80(one80)

def get_ft2m():
    """
    Element ft2m ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__ft2m()

def set_ft2m(ft2m):
    _mockdtpkg.f90wrap_precision__set__ft2m(ft2m)

def get_gsi():
    """
    Element gsi ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__gsi()

def set_gsi(gsi):
    _mockdtpkg.f90wrap_precision__set__gsi(gsi)

def get_gfps():
    """
    Element gfps ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__gfps()

def set_gfps(gfps):
    _mockdtpkg.f90wrap_precision__set__gfps(gfps)

def get_three60():
    """
    Element three60 ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__three60()

def set_three60(three60):
    _mockdtpkg.f90wrap_precision__set__three60(three60)

def get_in2ft():
    """
    Element in2ft ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__in2ft()

def set_in2ft(in2ft):
    _mockdtpkg.f90wrap_precision__set__in2ft(in2ft)

def get_five():
    """
    Element five ftype=real(kind=rdp) pytype=float
    Defined at ./Source/BasicDefs/aa0_typelist.fpp line 24
    """
    return _mockdtpkg.f90wrap_precision__get__five()

def set_five(five):
    _mockdtpkg.f90wrap_precision__set__five(five)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "precision".')

for func in _dt_array_initialisers:
    func()
