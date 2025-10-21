from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

class M_Error(f90wrap.runtime.FortranModule):
    """
    Module m_error
    Defined at main.fpp lines 5-55
    """
    @staticmethod
    def str_input(keyword=None, interface_call=False):
        """
        str_input([keyword])
        Defined at main.fpp lines 13-15
        
        Parameters
        ----------
        keyword : str
        """
        _pywrapper.f90wrap_m_error__str_input(keyword=keyword)
    
    @staticmethod
    def auto_raise(interface_call=False):
        """
        ierr, errmsg = auto_raise()
        Defined at main.fpp lines 17-23
        
        Returns
        -------
        ierr : int32
        errmsg : str
        """
        _pywrapper.f90wrap_m_error__auto_raise()
        return
    
    @staticmethod
    def auto_raise_optional(interface_call=False):
        """
        auto_raise_optional([ierr, errmsg])
        Defined at main.fpp lines 25-31
        
        Parameters
        ----------
        ierr : int32
        errmsg : str
        """
        _pywrapper.f90wrap_m_error__auto_raise_optional()
    
    @staticmethod
    def auto_no_raise(interface_call=False):
        """
        ierr, errmsg = auto_no_raise()
        Defined at main.fpp lines 33-39
        
        Returns
        -------
        ierr : int32
        errmsg : str
        """
        _pywrapper.f90wrap_m_error__auto_no_raise()
        return
    
    @staticmethod
    def auto_no_raise_optional(interface_call=False):
        """
        auto_no_raise_optional([ierr, errmsg])
        Defined at main.fpp lines 41-47
        
        Parameters
        ----------
        ierr : int32
        errmsg : str
        """
        _pywrapper.f90wrap_m_error__auto_no_raise_optional()
    
    @staticmethod
    def no_error_var(interface_call=False):
        """
        a_num, a_string = no_error_var()
        Defined at main.fpp lines 49-55
        
        Returns
        -------
        a_num : int32
        a_string : str
        """
        a_num, a_string = _pywrapper.f90wrap_m_error__no_error_var()
        return a_num, a_string
    
    _dt_array_initialisers = []
    

m_error = M_Error()

