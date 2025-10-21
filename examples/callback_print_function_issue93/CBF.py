from __future__ import print_function, absolute_import, division
import _CBF
import f90wrap.runtime
import logging
import numpy
import warnings

class Caller(f90wrap.runtime.FortranModule):
    """
    Module caller
    Defined at caller.fpp lines 5-22
    """
    @staticmethod
    def test_write_msg(interface_call=False):
        """
        test_write_msg()
        Defined at caller.fpp lines 10-12
        
        """
        _CBF.f90wrap_caller__test_write_msg()
    
    @staticmethod
    def test_write_msg_2(interface_call=False):
        """
        test_write_msg_2()
        Defined at caller.fpp lines 14-16
        
        """
        _CBF.f90wrap_caller__test_write_msg_2()
    
    _dt_array_initialisers = []
    

caller = Caller()

class Cback(f90wrap.runtime.FortranModule):
    """
    Module cback
    Defined at cback.fpp lines 5-26
    """
    @staticmethod
    def write_message(msg, interface_call=False):
        """
        write_message(msg)
        Defined at cback.fpp lines 9-13
        
        Parameters
        ----------
        msg : str
        """
        _CBF.f90wrap_cback__write_message(msg=msg)
    
    _dt_array_initialisers = []
    

cback = Cback()

