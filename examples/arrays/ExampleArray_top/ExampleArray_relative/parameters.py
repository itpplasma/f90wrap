"""
Module parameters
Defined at parameters.fpp lines 5-14
"""
from __future__ import print_function, absolute_import, division
from .. import _ExampleArray_relative
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_ExampleArray_relative = _SafeDirectCExecutor(_ExampleArray_relative, \
    module_import_name='_ExampleArray_relative')

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

def get_idp():
    """
    Element idp ftype=integer pytype=int
    Defined at parameters.fpp line 10
    """
    return _ExampleArray_relative.f90wrap_parameters__get__idp()

idp = get_idp()

def get_isp():
    """
    Element isp ftype=integer pytype=int
    Defined at parameters.fpp line 11
    """
    return _ExampleArray_relative.f90wrap_parameters__get__isp()

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
