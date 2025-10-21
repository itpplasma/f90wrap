from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

class M_Test(f90wrap.runtime.FortranModule):
    """
    Module m_test
    Defined at main.fpp lines 7-24
    """
    @staticmethod
    def test_real(in_real, interface_call=False):
        """
        out_int = test_real(in_real)
        Defined at main.fpp lines 11-14
        
        Parameters
        ----------
        in_real : float32
        
        Returns
        -------
        out_int : int32
        """
        out_int = _pywrapper.f90wrap_m_test__test_real(in_real=in_real)
        return out_int
    
    @staticmethod
    def test_real4(in_real, interface_call=False):
        """
        out_int = test_real4(in_real)
        Defined at main.fpp lines 16-19
        
        Parameters
        ----------
        in_real : float32
        
        Returns
        -------
        out_int : int32
        """
        out_int = _pywrapper.f90wrap_m_test__test_real4(in_real=in_real)
        return out_int
    
    @staticmethod
    def test_real8(in_real, interface_call=False):
        """
        out_int = test_real8(in_real)
        Defined at main.fpp lines 21-24
        
        Parameters
        ----------
        in_real : float64
        
        Returns
        -------
        out_int : int32
        """
        out_int = _pywrapper.f90wrap_m_test__test_real8(in_real=in_real)
        return out_int
    
    _dt_array_initialisers = []
    

m_test = M_Test()

