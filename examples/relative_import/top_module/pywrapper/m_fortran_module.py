"""
Module m_fortran_module
Defined at main.fpp lines 5-24
"""
from __future__ import print_function, absolute_import, division
from .. import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

def a_subroutine(self, interface_call=False):
    """
    a_subroutine(self)
    Defined at main.fpp lines 13-15
    
    Parameters
    ----------
    input : T_Base_Type
    """
    _pywrapper.f90wrap_m_fortran_module__a_subroutine(input=self._handle)

def b_subroutine(self, interface_call=False):
    """
    b_subroutine(self)
    Defined at main.fpp lines 17-20
    
    Parameters
    ----------
    input : T_Inheritance
    """
    _pywrapper.f90wrap_m_fortran_module__b_subroutine(input=self._handle)

def c_subroutine(self, interface_call=False):
    """
    c_subroutine(self)
    Defined at main.fpp lines 22-24
    
    Parameters
    ----------
    input : T_Composition
    """
    _pywrapper.f90wrap_m_fortran_module__c_subroutine(input=self._handle)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module \
        "m_fortran_module".')

for func in _dt_array_initialisers:
    func()
