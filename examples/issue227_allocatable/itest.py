from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy
import warnings

class Alloc_Output(f90wrap.runtime.FortranModule):
    """
    Module alloc_output
    Defined at alloc_output.fpp lines 5-35
    """
    @f90wrap.runtime.register_class("itest.alloc_output_type")
    class alloc_output_type(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=alloc_output_type)
        Defined at alloc_output.fpp lines 7-8
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for alloc_output_type
            
            self = Alloc_Output_Type()
            Defined at alloc_output.fpp lines 7-8
            
            Returns
            -------
            this : Alloc_Output_Type
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _itest.f90wrap_alloc_output__alloc_output_type_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for alloc_output_type
            
            Destructor for class Alloc_Output_Type
            Defined at alloc_output.fpp lines 7-8
            
            Parameters
            ----------
            this : Alloc_Output_Type
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _itest.f90wrap_alloc_output__alloc_output_type_finalise(this=self._handle)
        
        @property
        def a(self):
            """
            Element a ftype=real  pytype=float
            Defined at alloc_output.fpp line 8
            """
            return _itest.f90wrap_alloc_output__alloc_output_type__get__a(self._handle)
        
        @a.setter
        def a(self, a):
            _itest.f90wrap_alloc_output__alloc_output_type__set__a(self._handle, a)
        
        def __str__(self):
            ret = ['<alloc_output_type>{\n']
            ret.append('    a : ')
            ret.append(repr(self.a))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def alloc_output_type_func(val, interface_call=False):
        """
        out = alloc_output_type_func(val)
        Defined at alloc_output.fpp lines 12-16
        
        Parameters
        ----------
        val : float32
        
        Returns
        -------
        out : Alloc_Output_Type
        """
        out = _itest.f90wrap_alloc_output__alloc_output_type_func(val=val)
        out = f90wrap.runtime.lookup_class("itest.alloc_output_type").from_handle(out, \
            alloc=True)
        return out
    
    @staticmethod
    def noalloc_output_subroutine(val, out, interface_call=False):
        """
        noalloc_output_subroutine(val, out)
        Defined at alloc_output.fpp lines 32-35
        
        Parameters
        ----------
        val : float32
        out : Alloc_Output_Type
        """
        _itest.f90wrap_alloc_output__noalloc_output_subroutine(val=val, out=out._handle)
    
    _dt_array_initialisers = []
    
    

alloc_output = Alloc_Output()

