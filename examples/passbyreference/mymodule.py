from __future__ import print_function, absolute_import, division
import _mymodule
import f90wrap.runtime
import logging
import numpy
import warnings

class Mymodule(f90wrap.runtime.FortranModule):
    """
    Module mymodule
    Defined at mycode.fpp lines 5-18
    """
    @f90wrap.runtime.register_class("mymodule.mytype")
    class mytype(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=mytype)
        Defined at mycode.fpp lines 7-8
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for mytype
            
            self = Mytype()
            Defined at mycode.fpp lines 7-8
            
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
                result = _mymodule.f90wrap_mymodule__mytype_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for mytype
            
            Destructor for class Mytype
            Defined at mycode.fpp lines 7-8
            
            Parameters
            ----------
            this : Mytype
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _mymodule.f90wrap_mymodule__mytype_finalise(this=self._handle)
        
        @property
        def val(self):
            """
            Element val ftype=double precision pytype=unknown
            Defined at mycode.fpp line 8
            """
            return _mymodule.f90wrap_mymodule__mytype__get__val(self._handle)
        
        @val.setter
        def val(self, val):
            _mymodule.f90wrap_mymodule__mytype__set__val(self._handle, val)
        
        def __str__(self):
            ret = ['<mytype>{\n']
            ret.append('    val : ')
            ret.append(repr(self.val))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def mysubroutine(a, b, tt, interface_call=False):
        """
        mysubroutine(a, b, tt)
        Defined at mycode.fpp lines 11-18
        
        Parameters
        ----------
        a : unknown
        b : unknown
        tt : Mytype
        """
        _mymodule.f90wrap_mymodule__mysubroutine(a=a, b=b, tt=tt._handle)
    
    _dt_array_initialisers = []
    
    

mymodule = Mymodule()

