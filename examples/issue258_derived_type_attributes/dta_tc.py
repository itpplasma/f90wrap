from __future__ import print_function, absolute_import, division
import _dta_tc
import f90wrap.runtime
import logging
import numpy
import warnings

class Dta_Tc(f90wrap.runtime.FortranModule):
    """
    Module dta_tc
    Defined at dta_tc.fpp lines 5-42
    """
    @f90wrap.runtime.register_class("dta_tc.t_inner")
    class t_inner(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_inner)
        Defined at dta_tc.fpp lines 7-10
        """
        def __init__(self, value, handle=None):
            """
            inner = T_Inner(value)
            Defined at dta_tc.fpp lines 23-26
            
            Parameters
            ----------
            value : int32
            
            Returns
            -------
            inner : T_Inner
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _dta_tc.f90wrap_dta_tc__new_inner(value=value)
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_inner
            
            Destructor for class T_Inner
            Defined at dta_tc.fpp lines 7-10
            
            Parameters
            ----------
            this : T_Inner
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _dta_tc.f90wrap_dta_tc__t_inner_finalise(this=self._handle)
        
        def print(self, interface_call=False):
            """
            print(self)
            Defined at dta_tc.fpp lines 35-37
            
            Parameters
            ----------
            inner : T_Inner
            """
            _dta_tc.f90wrap_dta_tc__print__binding__t_inner(inner=self._handle)
        
        @property
        def value(self):
            """
            Element value ftype=integer  pytype=int
            Defined at dta_tc.fpp line 8
            """
            return _dta_tc.f90wrap_dta_tc__t_inner__get__value(self._handle)
        
        @value.setter
        def value(self, value):
            _dta_tc.f90wrap_dta_tc__t_inner__set__value(self._handle, value)
        
        def __str__(self):
            ret = ['<t_inner>{\n']
            ret.append('    value : ')
            ret.append(repr(self.value))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("dta_tc.t_outer")
    class t_outer(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_outer)
        Defined at dta_tc.fpp lines 12-14
        """
        def __init__(self, value, inner, handle=None):
            """
            node = T_Outer(value, inner)
            Defined at dta_tc.fpp lines 28-33
            
            Parameters
            ----------
            value : int32
            inner : T_Inner
            
            Returns
            -------
            node : T_Outer
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _dta_tc.f90wrap_dta_tc__new_outer(value=value, inner=inner._handle)
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_outer
            
            Destructor for class T_Outer
            Defined at dta_tc.fpp lines 12-14
            
            Parameters
            ----------
            this : T_Outer
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _dta_tc.f90wrap_dta_tc__t_outer_finalise(this=self._handle)
        
        @property
        def value(self):
            """
            Element value ftype=integer  pytype=int
            Defined at dta_tc.fpp line 13
            """
            return _dta_tc.f90wrap_dta_tc__t_outer__get__value(self._handle)
        
        @value.setter
        def value(self, value):
            _dta_tc.f90wrap_dta_tc__t_outer__set__value(self._handle, value)
        
        @property
        def inner(self):
            """
            Element inner ftype=type(t_inner) pytype=T_Inner
            Defined at dta_tc.fpp line 14
            """
            inner_handle = _dta_tc.f90wrap_dta_tc__t_outer__get__inner(self._handle)
            if tuple(inner_handle) in self._objs:
                inner = self._objs[tuple(inner_handle)]
            else:
                inner = dta_tc.t_inner.from_handle(inner_handle)
                self._objs[tuple(inner_handle)] = inner
            return inner
        
        @inner.setter
        def inner(self, inner):
            inner = inner._handle
            _dta_tc.f90wrap_dta_tc__t_outer__set__inner(self._handle, inner)
        
        def __str__(self):
            ret = ['<t_outer>{\n']
            ret.append('    value : ')
            ret.append(repr(self.value))
            ret.append(',\n    inner : ')
            ret.append(repr(self.inner))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def get_outer_inner(self, interface_call=False):
        """
        inner = get_outer_inner(self)
        Defined at dta_tc.fpp lines 39-42
        
        Parameters
        ----------
        outer : T_Outer
        
        Returns
        -------
        inner : T_Inner
        """
        inner = _dta_tc.f90wrap_dta_tc__get_outer_inner(outer=self._handle)
        inner = f90wrap.runtime.lookup_class("dta_tc.t_inner").from_handle(inner, \
            alloc=True)
        return inner
    
    _dt_array_initialisers = []
    
    if not hasattr(_dta_tc, "f90wrap_dta_tc__print__binding__t_inner"):
        for _candidate in ["f90wrap_dta_tc__print__binding__t_inner"]:
            if hasattr(_dta_tc, _candidate):
                setattr(_dta_tc, "f90wrap_dta_tc__print__binding__t_inner", getattr(_dta_tc, \
                    _candidate))
                break
    
    @staticmethod
    def print(instance, *args, **kwargs):
        return instance.print(*args, **kwargs)
    

dta_tc = Dta_Tc()

