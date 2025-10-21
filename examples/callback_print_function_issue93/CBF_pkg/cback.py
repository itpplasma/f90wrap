"""
Module cback
Defined at cback.fpp lines 5-26
"""
from __future__ import print_function, absolute_import, division
import _CBF_pkg
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

def write_message(msg, interface_call=False):
    """
    write_message(msg)
    Defined at cback.fpp lines 9-13
    
    Parameters
    ----------
    msg : str
    """
    _CBF_pkg.f90wrap_cback__write_message(msg=msg)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "cback".')

for func in _dt_array_initialisers:
    func()
