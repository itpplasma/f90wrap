from __future__ import print_function, absolute_import, division
import _ExampleArray
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_ExampleArray = _SafeDirectCExecutor(_ExampleArray, \
    module_import_name='_ExampleArray')

class Library(f90wrap.runtime.FortranModule):
    """
    Module library
    Defined at library.fpp lines 5-46
    """
    @staticmethod
    def do_array_stuff(n, x, y, br, co, interface_call=False):
        """
        do_array_stuff(n, x, y, br, co)
        Defined at library.fpp lines 12-25
        
        Parameters
        ----------
        n : int32
        x : float array
        y : float array
        br : float array
        co : float array
        """
        _ExampleArray.f90wrap_library__do_array_stuff(n=n, x=x, y=y, br=br, co=co)
    
    @staticmethod
    def only_manipulate(n, array, interface_call=False):
        """
        only_manipulate(n, array)
        Defined at library.fpp lines 27-35
        
        Parameters
        ----------
        n : int32
        array : float array
        """
        _ExampleArray.f90wrap_library__only_manipulate(n=n, array=array)
    
    @staticmethod
    def return_array(m, n, output, interface_call=False):
        """
        return_array(m, n, output)
        Defined at library.fpp lines 37-45
        
        Parameters
        ----------
        m : int32
        n : int32
        output : int array
        """
        _ExampleArray.f90wrap_library__return_array(m=m, n=n, output=output)
    
    _dt_array_initialisers = []
    

library = Library()

class Parameters(f90wrap.runtime.FortranModule):
    """
    Module parameters
    Defined at parameters.fpp lines 5-11
    """
    @property
    def idp(self):
        """
        Element idp ftype=integer pytype=int
        Defined at parameters.fpp line 9
        """
        return _ExampleArray.f90wrap_parameters__get__idp()
    
    def get_idp(self):
        return self.idp
    
    @property
    def isp(self):
        """
        Element isp ftype=integer pytype=int
        Defined at parameters.fpp line 10
        """
        return _ExampleArray.f90wrap_parameters__get__isp()
    
    def get_isp(self):
        return self.isp
    
    def __str__(self):
        ret = ['<parameters>{\n']
        ret.append('    idp : ')
        ret.append(repr(self.idp))
        ret.append(',\n    isp : ')
        ret.append(repr(self.isp))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

parameters = Parameters()

