from __future__ import print_function, absolute_import, division
import _subroutine_mod
import f90wrap.runtime
import logging
import numpy
import warnings

class Subroutine_Mod(f90wrap.runtime.FortranModule):
    """
    Module subroutine_mod
    Defined at subroutine_mod.fpp lines 5-40
    """
    @staticmethod
    def routine_with_simple_args(a, b, interface_call=False):
        """
        c, d = routine_with_simple_args(a, b)
        Defined at subroutine_mod.fpp lines 8-13
        
        Parameters
        ----------
        a : int32
        b : int32
        
        Returns
        -------
        c : int32
        d : int32
        """
        c, d = _subroutine_mod.f90wrap_subroutine_mod__routine_with_simple_args(a=a, \
            b=b)
        return c, d
    
    @staticmethod
    def routine_with_multiline_args(a, b, interface_call=False):
        """
        c, d = routine_with_multiline_args(a, b)
        Defined at subroutine_mod.fpp lines 15-23
        
        Parameters
        ----------
        a : int32
        b : int32
        
        Returns
        -------
        c : int32
        d : int32
        """
        c, d = _subroutine_mod.f90wrap_subroutine_mod__routine_with_multiline_args(a=a, \
            b=b)
        return c, d
    
    @staticmethod
    def routine_with_commented_args(a, b, interface_call=False):
        """
        c, d = routine_with_commented_args(a, b)
        Defined at subroutine_mod.fpp lines 25-34
        
        Parameters
        ----------
        a : int32
        b : int32
        
        Returns
        -------
        c : int32
        d : int32
        """
        c, d = _subroutine_mod.f90wrap_subroutine_mod__routine_with_commented_args(a=a, \
            b=b)
        return c, d
    
    @staticmethod
    def routine_with_more_commented_args(a, b, interface_call=False):
        """
        c, d = routine_with_more_commented_args(a, b)
        Defined at subroutine_mod.fpp lines 36-40
        
        Parameters
        ----------
        a : int32
        b : int32
        
        Returns
        -------
        c : int32
        d : int32
        """
        c, d = \
            _subroutine_mod.f90wrap_subroutine_mod__routine_with_more_commented_args(a=a, \
            b=b)
        return c, d
    
    _dt_array_initialisers = []
    

subroutine_mod = Subroutine_Mod()

