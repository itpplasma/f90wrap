"""
Module parameters
Defined at parameters.fpp lines 5-11
"""
from __future__ import print_function, absolute_import, division
import _ExampleDerivedTypes_pkg
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_ExampleDerivedTypes_pkg = _SafeDirectCExecutor(_ExampleDerivedTypes_pkg, \
    module_import_name='_ExampleDerivedTypes_pkg')

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

def get_idp():
    """
    Element idp ftype=integer pytype=int
    Defined at parameters.fpp line 9
    """
    return _ExampleDerivedTypes_pkg.f90wrap_parameters__get__idp()

idp = get_idp()

def get_isp():
    """
    Element isp ftype=integer pytype=int
    Defined at parameters.fpp line 10
    """
    return _ExampleDerivedTypes_pkg.f90wrap_parameters__get__isp()

isp = get_isp()


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "parameters".')

for func in _dt_array_initialisers:
    func()
