from __future__ import print_function, absolute_import, division
import _mytype
import f90wrap.runtime
import logging
import numpy
import warnings

class Mytype_Mod(f90wrap.runtime.FortranModule):
    """
    Module mytype_mod
    Defined at mytype_mod.fpp lines 5-19
    """
    @f90wrap.runtime.register_class("mytype.mytype")
    class mytype(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=mytype)
        Defined at mytype_mod.fpp lines 7-8
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for mytype
            
            self = Mytype()
            Defined at mytype_mod.fpp lines 7-8
            
            Returns
            -------
            this : Mytype
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _mytype.f90wrap_mytype_mod__mytype_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for mytype
            
            Destructor for class Mytype
            Defined at mytype_mod.fpp lines 7-8
            
            Parameters
            ----------
            this : Mytype
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _mytype.f90wrap_mytype_mod__mytype_finalise(this=self._handle)
        
        @property
        def a(self):
            """
            Element a ftype=integer  pytype=int
            Defined at mytype_mod.fpp line 8
            """
            return _mytype.f90wrap_mytype_mod__mytype__get__a(self._handle)
        
        @a.setter
        def a(self, a):
            _mytype.f90wrap_mytype_mod__mytype__set__a(self._handle, a)
        
        def __str__(self):
            ret = ['<mytype>{\n']
            ret.append('    a : ')
            ret.append(repr(self.a))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def constructor(interface_call=False):
        """
        obj = constructor()
        Defined at mytype_mod.fpp lines 11-13
        
        Returns
        -------
        obj : Mytype
        """
        obj = _mytype.f90wrap_mytype_mod__constructor()
        obj = f90wrap.runtime.lookup_class("mytype.mytype").from_handle(obj, alloc=True)
        return obj
    
    @staticmethod
    def plus_b(self, b, interface_call=False):
        """
        c = plus_b(self, b)
        Defined at mytype_mod.fpp lines 15-19
        
        Parameters
        ----------
        obj : Mytype
        b : int32
        
        Returns
        -------
        c : int32
        """
        c = _mytype.f90wrap_mytype_mod__plus_b(obj=self._handle, b=b)
        return c
    
    _dt_array_initialisers = []
    
    

mytype_mod = Mytype_Mod()

