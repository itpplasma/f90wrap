from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings

class M_Out_Test(f90wrap.runtime.FortranModule):
    """
    Module m_out_test
    Defined at main.fpp lines 5-53
    """
    @staticmethod
    def out_scalar_int1(interface_call=False):
        """
        output = out_scalar_int1()
        Defined at main.fpp lines 14-16
        
        Returns
        -------
        output : int8
        """
        output = _pywrapper.f90wrap_m_out_test__out_scalar_int1()
        return output
    
    @staticmethod
    def out_scalar_int2(interface_call=False):
        """
        output = out_scalar_int2()
        Defined at main.fpp lines 18-20
        
        Returns
        -------
        output : int16
        """
        output = _pywrapper.f90wrap_m_out_test__out_scalar_int2()
        return output
    
    @staticmethod
    def out_scalar_int4(interface_call=False):
        """
        output = out_scalar_int4()
        Defined at main.fpp lines 22-24
        
        Returns
        -------
        output : int32
        """
        output = _pywrapper.f90wrap_m_out_test__out_scalar_int4()
        return output
    
    @staticmethod
    def out_scalar_int8(interface_call=False):
        """
        output = out_scalar_int8()
        Defined at main.fpp lines 26-28
        
        Returns
        -------
        output : int64
        """
        output = _pywrapper.f90wrap_m_out_test__out_scalar_int8()
        return output
    
    @staticmethod
    def out_scalar_real4(interface_call=False):
        """
        output = out_scalar_real4()
        Defined at main.fpp lines 30-32
        
        Returns
        -------
        output : float32
        """
        output = _pywrapper.f90wrap_m_out_test__out_scalar_real4()
        return output
    
    @staticmethod
    def out_scalar_real8(interface_call=False):
        """
        output = out_scalar_real8()
        Defined at main.fpp lines 34-36
        
        Returns
        -------
        output : float64
        """
        output = _pywrapper.f90wrap_m_out_test__out_scalar_real8()
        return output
    
    @staticmethod
    def out_array_int4(interface_call=False):
        """
        output = out_array_int4()
        Defined at main.fpp lines 38-40
        
        Returns
        -------
        output : int array
        """
        output = _pywrapper.f90wrap_m_out_test__out_array_int4()
        return output
    
    @staticmethod
    def out_array_int8(interface_call=False):
        """
        output = out_array_int8()
        Defined at main.fpp lines 42-44
        
        Returns
        -------
        output : int array
        """
        output = _pywrapper.f90wrap_m_out_test__out_array_int8()
        return output
    
    @staticmethod
    def out_array_real4(interface_call=False):
        """
        output = out_array_real4()
        Defined at main.fpp lines 46-48
        
        Returns
        -------
        output : float array
        """
        output = _pywrapper.f90wrap_m_out_test__out_array_real4()
        return output
    
    @staticmethod
    def out_array_real8(interface_call=False):
        """
        output = out_array_real8()
        Defined at main.fpp lines 50-52
        
        Returns
        -------
        output : float array
        """
        output = _pywrapper.f90wrap_m_out_test__out_array_real8()
        return output
    
    _dt_array_initialisers = []
    

m_out_test = M_Out_Test()

