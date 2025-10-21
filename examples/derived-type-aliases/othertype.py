from __future__ import print_function, absolute_import, division
import _othertype
import f90wrap.runtime
import logging
import numpy
import warnings

class Othertype_Mod(f90wrap.runtime.FortranModule):
    """
    Module othertype_mod
    Defined at othertype_mod.fpp lines 5-19
    """
    @f90wrap.runtime.register_class("othertype.othertype")
    class othertype(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=othertype)
        Defined at othertype_mod.fpp lines 7-8
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for othertype
            
            self = Othertype()
            Defined at othertype_mod.fpp lines 7-8
            
            Returns
            -------
            this : Othertype
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _othertype.f90wrap_othertype_mod__othertype_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for othertype
            
            Destructor for class Othertype
            Defined at othertype_mod.fpp lines 7-8
            
            Parameters
            ----------
            this : Othertype
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _othertype.f90wrap_othertype_mod__othertype_finalise(this=self._handle)
        
        @property
        def a(self):
            """
            Element a ftype=integer  pytype=int
            Defined at othertype_mod.fpp line 8
            """
            return _othertype.f90wrap_othertype_mod__othertype__get__a(self._handle)
        
        @a.setter
        def a(self, a):
            _othertype.f90wrap_othertype_mod__othertype__set__a(self._handle, a)
        
        def __str__(self):
            ret = ['<othertype>{\n']
            ret.append('    a : ')
            ret.append(repr(self.a))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def constructor(interface_call=False):
        """
        obj = constructor()
        Defined at othertype_mod.fpp lines 11-13
        
        Returns
        -------
        obj : Othertype
        """
        obj = _othertype.f90wrap_othertype_mod__constructor()
        obj = f90wrap.runtime.lookup_class("othertype.othertype").from_handle(obj, \
            alloc=True)
        return obj
    
    @staticmethod
    def plus_b(self, b, interface_call=False):
        """
        c = plus_b(self, b)
        Defined at othertype_mod.fpp lines 15-19
        
        Parameters
        ----------
        obj : Othertype
        b : int32
        
        Returns
        -------
        c : int32
        """
        c = _othertype.f90wrap_othertype_mod__plus_b(obj=self._handle, b=b)
        return c
    
    _dt_array_initialisers = []
    
    

othertype_mod = Othertype_Mod()

