from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings

class M_Test(f90wrap.runtime.FortranModule):
    """
    Module m_test
    Defined at main.fpp lines 7-21
    """
    @staticmethod
    def not_to_be_ignored(interface_call=False):
        """
        out_int = not_to_be_ignored()
        Defined at main.fpp lines 19-21
        
        Returns
        -------
        out_int : int32
        """
        out_int = _pywrapper.f90wrap_m_test__not_to_be_ignored()
        return out_int
    
    _dt_array_initialisers = []
    

m_test = M_Test()

