from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

class M_Intent_Out(f90wrap.runtime.FortranModule):
    """
    Module m_intent_out
    Defined at main.fpp lines 5-20
    """
    @staticmethod
    def interpolation(n1, n2, a1, a2, output, interface_call=False):
        """
        interpolation(n1, n2, a1, a2, output)
        Defined at main.fpp lines 9-19
        
        Parameters
        ----------
        n1 : int32
        n2 : int32
        a1 : float array
        a2 : float array
        output : float array
        """
        _pywrapper.f90wrap_m_intent_out__interpolation(n1=n1, n2=n2, a1=a1, a2=a2, \
            output=output)
    
    _dt_array_initialisers = []
    

m_intent_out = M_Intent_Out()

