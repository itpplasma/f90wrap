from __future__ import print_function, absolute_import, division
import _keywordr_rename
import f90wrap.runtime
import logging
import numpy
import warnings

class Global_(f90wrap.runtime.FortranModule):
    """
    Module global_
    Defined at rename.fpp lines 5-19
    """
    @f90wrap.runtime.register_class("keywordr_rename.class2")
    class class2(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=class2)
        Defined at rename.fpp lines 7-8
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for class2
            
            self = Class2()
            Defined at rename.fpp lines 7-8
            
            Returns
            -------
            this : Class2
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _keywordr_rename.f90wrap_global__class2_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for class2
            
            Destructor for class Class2
            Defined at rename.fpp lines 7-8
            
            Parameters
            ----------
            this : Class2
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _keywordr_rename.f90wrap_global__class2_finalise(this=self._handle)
        
        @property
        def x(self):
            """
            Element x ftype=integer  pytype=int
            Defined at rename.fpp line 8
            """
            return _keywordr_rename.f90wrap_global__class2__get__x(self._handle)
        
        @x.setter
        def x(self, x):
            _keywordr_rename.f90wrap_global__class2__set__x(self._handle, x)
        
        def __str__(self):
            ret = ['<class2>{\n']
            ret.append('    x : ')
            ret.append(repr(self.x))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def is_(a, interface_call=False):
        """
        is_(a)
        Defined at rename.fpp lines 14-18
        
        Parameters
        ----------
        a : int32
        """
        _keywordr_rename.f90wrap_global__is_(a=a)
    
    @property
    def abc(self):
        """
        Element abc ftype=integer  pytype=int
        Defined at rename.fpp line 10
        """
        return _keywordr_rename.f90wrap_global___get__abc()
    
    @abc.setter
    def abc(self, abc):
        _keywordr_rename.f90wrap_global___set__abc(abc)
    
    def get_abc(self):
        return self.abc
    
    def set_abc(self, value):
        self.abc = value
    
    @property
    def lambda_(self):
        """
        Element lambda_ ftype=integer pytype=int
        Defined at rename.fpp line 11
        """
        return _keywordr_rename.f90wrap_global___get__lambda_()
    
    def get_lambda_(self):
        return self.lambda_
    
    @property
    def with_(self):
        """
        Element with_ ftype=integer  pytype=int
        Defined at rename.fpp line 12
        """
        array_ndim, array_type, array_shape, array_handle = \
            _keywordr_rename.f90wrap_global___array__with_(f90wrap.runtime.empty_handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            with_ = self._arrays[array_hash]
        else:
            try:
                with_ = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        f90wrap.runtime.empty_handle,
                                        _keywordr_rename.f90wrap_global___array__with_)
            except TypeError:
                with_ = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = with_
        return with_
    
    @with_.setter
    def with_(self, with_):
        self.with_[...] = with_
    
    def set_array_with_(self, value):
        self.with_[...] = value
    
    def get_array_with_(self):
        return self.with_
    
    def __str__(self):
        ret = ['<global_>{\n']
        ret.append('    abc : ')
        ret.append(repr(self.abc))
        ret.append(',\n    lambda_ : ')
        ret.append(repr(self.lambda_))
        ret.append(',\n    with_ : ')
        ret.append(repr(self.with_))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    
    

global_ = Global_()

def in_(a, interface_call=False):
    """
    in_ = in_(a)
    Defined at rename.fpp lines 21-24
    
    Parameters
    ----------
    a : int32
    
    Returns
    -------
    in_ : int32
    """
    in_ = _keywordr_rename.f90wrap_in_(a=a)
    return in_

