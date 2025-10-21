from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy
import warnings

class Itestit(f90wrap.runtime.FortranModule):
    """
    Module itestit
    Defined at main.fpp lines 5-23
    """
    @staticmethod
    def testit1(x, interface_call=False):
        """
        testit1(x)
        Defined at main.fpp lines 13-18
        
        Parameters
        ----------
        x : float array
        """
        _itest.f90wrap_itestit__testit1(x=x)
    
    @staticmethod
    def testit2(x, interface_call=False):
        """
        testit2(x)
        Defined at main.fpp lines 20-23
        
        Parameters
        ----------
        x : float array
        """
        _itest.f90wrap_itestit__testit2(x=x)
    
    _dt_array_initialisers = []
    

itestit = Itestit()

