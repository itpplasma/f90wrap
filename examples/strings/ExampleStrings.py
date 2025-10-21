from __future__ import print_function, absolute_import, division
import _ExampleStrings
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_ExampleStrings = _SafeDirectCExecutor(_ExampleStrings, \
    module_import_name='_ExampleStrings')

class String_Io(f90wrap.runtime.FortranModule):
    """
    Module string_io
    Defined at string_io.fpp lines 5-49
    """
    @staticmethod
    def func_generate_string(n, interface_call=False):
        """
        stringout = func_generate_string(n)
        Defined at string_io.fpp lines 9-16
        
        Parameters
        ----------
        n : int32
        
        Returns
        -------
        stringout : str
        """
        stringout = _ExampleStrings.f90wrap_string_io__func_generate_string(n=n)
        return stringout
    
    @staticmethod
    def func_return_string(interface_call=False):
        """
        stringout = func_return_string()
        Defined at string_io.fpp lines 18-20
        
        Returns
        -------
        stringout : str
        """
        stringout = _ExampleStrings.f90wrap_string_io__func_return_string()
        return stringout
    
    @staticmethod
    def generate_string(n, interface_call=False):
        """
        stringout = generate_string(n)
        Defined at string_io.fpp lines 22-30
        
        Parameters
        ----------
        n : int32
        
        Returns
        -------
        stringout : str
        """
        stringout = _ExampleStrings.f90wrap_string_io__generate_string(n=n)
        return stringout
    
    @staticmethod
    def return_string(interface_call=False):
        """
        stringout = return_string()
        Defined at string_io.fpp lines 32-34
        
        Returns
        -------
        stringout : str
        """
        stringout = _ExampleStrings.f90wrap_string_io__return_string()
        return stringout
    
    @staticmethod
    def set_global_string(n, newstring, interface_call=False):
        """
        set_global_string(n, newstring)
        Defined at string_io.fpp lines 36-39
        
        Parameters
        ----------
        n : int32
        newstring : str
        """
        _ExampleStrings.f90wrap_string_io__set_global_string(n=n, newstring=newstring)
    
    @staticmethod
    def inout_string(n, stringinout, interface_call=False):
        """
        inout_string(n, stringinout)
        Defined at string_io.fpp lines 41-48
        
        Parameters
        ----------
        n : int32
        stringinout : str
        """
        _ExampleStrings.f90wrap_string_io__inout_string(n=n, stringinout=stringinout)
    
    @property
    def global_string(self):
        """
        Element global_string ftype=character(512) pytype=str
        Defined at string_io.fpp line 7
        """
        return _ExampleStrings.f90wrap_string_io__get__global_string()
    
    @global_string.setter
    def global_string(self, global_string):
        _ExampleStrings.f90wrap_string_io__set__global_string(global_string)
    
    def get_global_string(self):
        return self.global_string
    
    def set_global_string_value(self, value):
        self.global_string = value
    
    def __str__(self):
        ret = ['<string_io>{\n']
        ret.append('    global_string : ')
        ret.append(repr(self.global_string))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

string_io = String_Io()

