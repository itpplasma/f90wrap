"""
Module caller
Defined at caller.fpp lines 5-22
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

def test_write_msg(interface_call=False):
    """
    test_write_msg()
    Defined at caller.fpp lines 10-12
    
    """
    _CBF_pkg.f90wrap_caller__test_write_msg()

def test_write_msg_2(interface_call=False):
    """
    test_write_msg_2()
    Defined at caller.fpp lines 14-16
    
    """
    _CBF_pkg.f90wrap_caller__test_write_msg_2()


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "caller".')

for func in _dt_array_initialisers:
    func()
