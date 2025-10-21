from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy
import warnings

def test(interface_call=False):
    """
    test()
    Defined at main.fpp lines 13-18
    
    """
    _itest.f90wrap_test()

class Myclass_Base(f90wrap.runtime.FortranModule):
    """
    Module myclass_base
    Defined at myclass_base.fpp lines 5-16
    """
    @f90wrap.runtime.register_class("itest.myclass_t")
    class myclass_t(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=myclass_t)
        Defined at myclass_base.fpp lines 7-9
        """
        def __init__(self):
            raise(NotImplementedError("This is an abstract class"))
        
        def get_value(self, interface_call=False):
            """
            value = get_value(self)
            Defined at myclass_base.fpp lines 12-15
            
            Parameters
            ----------
            self : Myclass_T
            
            Returns
            -------
            value : float32
            """
            value = \
                _itest.f90wrap_myclass_base__get_value__binding__myclass_t(self=self._handle)
            return value
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    if not hasattr(_itest, "f90wrap_myclass_base__get_value__binding__myclass_t"):
        for _candidate in ["f90wrap_myclass_base__get_value__binding__myclass_t"]:
            if hasattr(_itest, _candidate):
                setattr(_itest, "f90wrap_myclass_base__get_value__binding__myclass_t", \
                    getattr(_itest, _candidate))
                break
    
    @staticmethod
    def get_value(instance, *args, **kwargs):
        return instance.get_value(*args, **kwargs)
    

myclass_base = Myclass_Base()

class Myclass_Factory(f90wrap.runtime.FortranModule):
    """
    Module myclass_factory
    Defined at myclass_factory.fpp lines 5-22
    """
    @staticmethod
    def create_myclass(impl_type, interface_call=False):
        """
        myobject = create_myclass(impl_type)
        Defined at myclass_factory.fpp lines 11-22
        
        Parameters
        ----------
        impl_type : str
        
        Returns
        -------
        myobject : Myclass_T
        """
        myobject = _itest.f90wrap_myclass_factory__create_myclass(impl_type=impl_type)
        myobject = f90wrap.runtime.lookup_class("itest.myclass_t").from_handle(myobject, \
            alloc=True)
        return myobject
    
    _dt_array_initialisers = []
    

myclass_factory = Myclass_Factory()

class Myclass_Impl2(f90wrap.runtime.FortranModule):
    """
    Module myclass_impl2
    Defined at myclass_impl2.fpp lines 5-21
    """
    @f90wrap.runtime.register_class("itest.myclass_impl2_t")
    class myclass_impl2_t(myclass_base.myclass_t):
        """
        Type(name=myclass_impl2_t)
        Defined at myclass_impl2.fpp lines 8-11
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for myclass_impl2_t
            
            self = Myclass_Impl2_T()
            Defined at myclass_impl2.fpp lines 8-11
            
            Returns
            -------
            this : Myclass_Impl2_T
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _itest.f90wrap_myclass_impl2__myclass_impl2_t_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def get_value(self, interface_call=False):
            """
            value = get_value(self)
            Defined at myclass_impl2.fpp lines 14-17
            
            Parameters
            ----------
            self : Myclass_Impl2_T
            
            Returns
            -------
            value : float32
            """
            value = \
                _itest.f90wrap_myclass_impl2__get_value__binding__myclass_impl2_t(self=self._handle)
            return value
        
        def __del__(self):
            """
            Destructor for class Myclass_Impl2_T
            Defined at myclass_impl2.fpp lines 19-21
            
            Parameters
            ----------
            self : Myclass_Impl2_T
            """
            if getattr(self, '_alloc', False):
                _itest.f90wrap_myclass_impl2__myclass_impl2_destroy__binding__mycla358(self=self._handle)
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    if not hasattr(_itest, \
        "f90wrap_myclass_impl2__get_value__binding__myclass_impl2_t"):
        for _candidate in \
            ["f90wrap_myclass_impl2__get_value__binding__myclass_impl2_t"]:
            if hasattr(_itest, _candidate):
                setattr(_itest, "f90wrap_myclass_impl2__get_value__binding__myclass_impl2_t", \
                    getattr(_itest, _candidate))
                break
    
    @staticmethod
    def get_value(instance, *args, **kwargs):
        return instance.get_value(*args, **kwargs)
    

myclass_impl2 = Myclass_Impl2()

class Myclass_Impl(f90wrap.runtime.FortranModule):
    """
    Module myclass_impl
    Defined at myclass_impl.fpp lines 5-21
    """
    @f90wrap.runtime.register_class("itest.myclass_impl_t")
    class myclass_impl_t(myclass_base.myclass_t):
        """
        Type(name=myclass_impl_t)
        Defined at myclass_impl.fpp lines 8-11
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for myclass_impl_t
            
            self = Myclass_Impl_T()
            Defined at myclass_impl.fpp lines 8-11
            
            Returns
            -------
            this : Myclass_Impl_T
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _itest.f90wrap_myclass_impl__myclass_impl_t_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def get_value(self, interface_call=False):
            """
            value = get_value(self)
            Defined at myclass_impl.fpp lines 14-17
            
            Parameters
            ----------
            self : Myclass_Impl_T
            
            Returns
            -------
            value : float32
            """
            value = \
                _itest.f90wrap_myclass_impl__get_value__binding__myclass_impl_t(self=self._handle)
            return value
        
        def __del__(self):
            """
            Destructor for class Myclass_Impl_T
            Defined at myclass_impl.fpp lines 19-21
            
            Parameters
            ----------
            self : Myclass_Impl_T
            """
            if getattr(self, '_alloc', False):
                _itest.f90wrap_myclass_impl__myclass_impl_destroy__binding__myclas021a(self=self._handle)
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    if not hasattr(_itest, \
        "f90wrap_myclass_impl__get_value__binding__myclass_impl_t"):
        for _candidate in ["f90wrap_myclass_impl__get_value__binding__myclass_impl_t"]:
            if hasattr(_itest, _candidate):
                setattr(_itest, "f90wrap_myclass_impl__get_value__binding__myclass_impl_t", \
                    getattr(_itest, _candidate))
                break
    
    @staticmethod
    def get_value(instance, *args, **kwargs):
        return instance.get_value(*args, **kwargs)
    

myclass_impl = Myclass_Impl()

