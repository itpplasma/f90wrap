"""
Module constant_parameters
Defined at ./Source/BasicDefs/aa1_modules.fpp lines 14-30
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


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module \
        "constant_parameters".')

for func in _dt_array_initialisers:
    func()
