from __future__ import print_function, absolute_import, division
import _ExampleArray
import f90wrap.runtime
import logging
import numpy
import warnings

class Library(f90wrap.runtime.FortranModule):
    """
    Module library
    Defined at library.fpp lines 5-46
    """
    @staticmethod
    def do_array_stuff(n, x, y, br, co, interface_call=False):
        """
        do_array_stuff(n, x, y, br, co)
        Defined at library.fpp lines 13-25
        
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
    
    @property
    def ia(self):
        """
        Element ia ftype=integer(4) pytype=int
        Defined at library.fpp line 10
        """
        return _ExampleArray.f90wrap_library__get__ia()
    
    @ia.setter
    def ia(self, ia):
        _ExampleArray.f90wrap_library__set__ia(ia)
    
    def get_ia(self):
        return self.ia
    
    def set_ia(self, value):
        self.ia = value
    
    @property
    def iarray(self):
        """
        Element iarray ftype=integer(4) pytype=int
        Defined at library.fpp line 11
        """
        array_ndim, array_type, array_shape, array_handle = \
            _ExampleArray.f90wrap_library__array__iarray(f90wrap.runtime.empty_handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            iarray = self._arrays[array_hash]
        else:
            try:
                iarray = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        f90wrap.runtime.empty_handle,
                                        _ExampleArray.f90wrap_library__array__iarray)
            except TypeError:
                iarray = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = iarray
        return iarray
    
    @iarray.setter
    def iarray(self, iarray):
        self.iarray[...] = iarray
    
    def set_array_iarray(self, value):
        self.iarray[...] = value
    
    def get_array_iarray(self):
        return self.iarray
    
    def __str__(self):
        ret = ['<library>{\n']
        ret.append('    ia : ')
        ret.append(repr(self.ia))
        ret.append(',\n    iarray : ')
        ret.append(repr(self.iarray))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

library = Library()

class Parameters(f90wrap.runtime.FortranModule):
    """
    Module parameters
    Defined at parameters.fpp lines 5-14
    """
    @property
    def idp(self):
        """
        Element idp ftype=integer pytype=int
        Defined at parameters.fpp line 10
        """
        return _ExampleArray.f90wrap_parameters__get__idp()
    
    def get_idp(self):
        return self.idp
    
    @property
    def isp(self):
        """
        Element isp ftype=integer pytype=int
        Defined at parameters.fpp line 11
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

