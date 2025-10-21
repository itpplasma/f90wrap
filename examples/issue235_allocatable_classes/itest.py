from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy
import warnings

class Myclass(f90wrap.runtime.FortranModule):
    """
    Module myclass
    Defined at myclass.fpp lines 5-30
    """
    @f90wrap.runtime.register_class("itest.myclass_t")
    class myclass_t(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=myclass_t)
        Defined at myclass.fpp lines 9-14
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for myclass_t
            
            self = Myclass_T()
            Defined at myclass.fpp lines 9-14
            
            Returns
            -------
            this : Myclass_T
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _itest.f90wrap_myclass__myclass_t_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def get_val(self, interface_call=False):
            """
            val = get_val(self)
            Defined at myclass.fpp lines 17-20
            
            Parameters
            ----------
            self : Myclass_T
            
            Returns
            -------
            val : float32
            """
            val = _itest.f90wrap_myclass__get_val__binding__myclass_t(self=self._handle)
            return val
        
        def set_val(self, val, interface_call=False):
            """
            set_val(self, val)
            Defined at myclass.fpp lines 22-25
            
            Parameters
            ----------
            self : Myclass_T
            val : float32
            """
            _itest.f90wrap_myclass__set_val__binding__myclass_t(self=self._handle, val=val)
        
        def __del__(self):
            """
            Destructor for class Myclass_T
            Defined at myclass.fpp lines 27-30
            
            Parameters
            ----------
            self : Myclass_T
            """
            if getattr(self, '_alloc', False):
                _itest.f90wrap_myclass__myclass_destroy__binding__myclass_t(self=self._handle)
        
        @property
        def val(self):
            """
            Element val ftype=real  pytype=float
            Defined at myclass.fpp line 10
            """
            return _itest.f90wrap_myclass__myclass_t__get__val(self._handle)
        
        @val.setter
        def val(self, val):
            _itest.f90wrap_myclass__myclass_t__set__val(self._handle, val)
        
        def __str__(self):
            ret = ['<myclass_t>{\n']
            ret.append('    val : ')
            ret.append(repr(self.val))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @property
    def create_count(self):
        """
        Element create_count ftype=integer  pytype=int
        Defined at myclass.fpp line 7
        """
        return _itest.f90wrap_myclass__get__create_count()
    
    @create_count.setter
    def create_count(self, create_count):
        _itest.f90wrap_myclass__set__create_count(create_count)
    
    def get_create_count(self):
        return self.create_count
    
    def set_create_count(self, value):
        self.create_count = value
    
    @property
    def destroy_count(self):
        """
        Element destroy_count ftype=integer  pytype=int
        Defined at myclass.fpp line 8
        """
        return _itest.f90wrap_myclass__get__destroy_count()
    
    @destroy_count.setter
    def destroy_count(self, destroy_count):
        _itest.f90wrap_myclass__set__destroy_count(destroy_count)
    
    def get_destroy_count(self):
        return self.destroy_count
    
    def set_destroy_count(self, value):
        self.destroy_count = value
    
    def __str__(self):
        ret = ['<myclass>{\n']
        ret.append('    create_count : ')
        ret.append(repr(self.create_count))
        ret.append(',\n    destroy_count : ')
        ret.append(repr(self.destroy_count))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    
    if not hasattr(_itest, "f90wrap_myclass__get_val__binding__myclass_t"):
        for _candidate in ["f90wrap_myclass__get_val__binding__myclass_t"]:
            if hasattr(_itest, _candidate):
                setattr(_itest, "f90wrap_myclass__get_val__binding__myclass_t", getattr(_itest, \
                    _candidate))
                break
    if not hasattr(_itest, "f90wrap_myclass__set_val__binding__myclass_t"):
        for _candidate in ["f90wrap_myclass__set_val__binding__myclass_t"]:
            if hasattr(_itest, _candidate):
                setattr(_itest, "f90wrap_myclass__set_val__binding__myclass_t", getattr(_itest, \
                    _candidate))
                break
    
    @staticmethod
    def get_val(instance, *args, **kwargs):
        return instance.get_val(*args, **kwargs)
    
    @staticmethod
    def set_val(instance, *args, **kwargs):
        return instance.set_val(*args, **kwargs)
    

myclass = Myclass()

class Myclass_Factory(f90wrap.runtime.FortranModule):
    """
    Module myclass_factory
    Defined at myclass_factory.fpp lines 5-14
    """
    @staticmethod
    def myclass_create(val, interface_call=False):
        """
        myobject = myclass_create(val)
        Defined at myclass_factory.fpp lines 9-14
        
        Parameters
        ----------
        val : float32
        
        Returns
        -------
        myobject : Myclass_T
        """
        myobject = _itest.f90wrap_myclass_factory__myclass_create(val=val)
        myobject = f90wrap.runtime.lookup_class("itest.myclass_t").from_handle(myobject, \
            alloc=True)
        return myobject
    
    _dt_array_initialisers = []
    

myclass_factory = Myclass_Factory()

class Mytype(f90wrap.runtime.FortranModule):
    """
    Module mytype
    Defined at mytype.fpp lines 5-24
    """
    @f90wrap.runtime.register_class("itest.mytype_t")
    class mytype_t(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=mytype_t)
        Defined at mytype.fpp lines 9-12
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for mytype_t
            
            self = Mytype_T()
            Defined at mytype.fpp lines 9-12
            
            Returns
            -------
            this : Mytype_T
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _itest.f90wrap_mytype__mytype_t_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Destructor for class Mytype_T
            Defined at mytype.fpp lines 21-24
            
            Parameters
            ----------
            self : Mytype_T
            """
            if getattr(self, '_alloc', False):
                _itest.f90wrap_mytype__mytype_destroy__binding__mytype_t(self=self._handle)
        
        @property
        def val(self):
            """
            Element val ftype=real  pytype=float
            Defined at mytype.fpp line 10
            """
            return _itest.f90wrap_mytype__mytype_t__get__val(self._handle)
        
        @val.setter
        def val(self, val):
            _itest.f90wrap_mytype__mytype_t__set__val(self._handle, val)
        
        def __str__(self):
            ret = ['<mytype_t>{\n']
            ret.append('    val : ')
            ret.append(repr(self.val))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def mytype_create(val, interface_call=False):
        """
        self = mytype_create(val)
        Defined at mytype.fpp lines 15-19
        
        Parameters
        ----------
        val : float32
        
        Returns
        -------
        self : Mytype_T
        """
        self = _itest.f90wrap_mytype__mytype_create(val=val)
        self = f90wrap.runtime.lookup_class("itest.mytype_t").from_handle(self, \
            alloc=True)
        return self
    
    @property
    def create_count(self):
        """
        Element create_count ftype=integer  pytype=int
        Defined at mytype.fpp line 7
        """
        return _itest.f90wrap_mytype__get__create_count()
    
    @create_count.setter
    def create_count(self, create_count):
        _itest.f90wrap_mytype__set__create_count(create_count)
    
    def get_create_count(self):
        return self.create_count
    
    def set_create_count(self, value):
        self.create_count = value
    
    @property
    def destroy_count(self):
        """
        Element destroy_count ftype=integer  pytype=int
        Defined at mytype.fpp line 8
        """
        return _itest.f90wrap_mytype__get__destroy_count()
    
    @destroy_count.setter
    def destroy_count(self, destroy_count):
        _itest.f90wrap_mytype__set__destroy_count(destroy_count)
    
    def get_destroy_count(self):
        return self.destroy_count
    
    def set_destroy_count(self, value):
        self.destroy_count = value
    
    def __str__(self):
        ret = ['<mytype>{\n']
        ret.append('    create_count : ')
        ret.append(repr(self.create_count))
        ret.append(',\n    destroy_count : ')
        ret.append(repr(self.destroy_count))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    
    

mytype = Mytype()

