from __future__ import print_function, absolute_import, division
import _library
import f90wrap.runtime
import logging
import numpy
import warnings

class Test(f90wrap.runtime.FortranModule):
    """
    Module test
    Defined at library.fpp lines 5-74
    """
    @f90wrap.runtime.register_class("library.atype")
    class atype(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=atype)
        Defined at library.fpp lines 11-19
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for atype
            
            self = Atype()
            Defined at library.fpp lines 11-19
            
            Returns
            -------
            this : Atype
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _library.f90wrap_test__atype_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for atype
            
            Destructor for class Atype
            Defined at library.fpp lines 11-19
            
            Parameters
            ----------
            this : Atype
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _library.f90wrap_test__atype_finalise(this=self._handle)
        
        def p_create(self, n, interface_call=False):
            """
            p_create(self, n)
            Defined at library.fpp lines 36-42
            
            Parameters
            ----------
            self : Atype
            n : int32
            """
            _library.f90wrap_test__p_create__binding__atype(self=self._handle, n=n)
        
        def p_asum(self, interface_call=False):
            """
            asum_class = p_asum(self)
            Defined at library.fpp lines 52-58
            
            Parameters
            ----------
            self : Atype
            
            Returns
            -------
            asum_class : float32
            """
            asum_class = _library.f90wrap_test__p_asum__binding__atype(self=self._handle)
            return asum_class
        
        def p_asum_2(self, interface_call=False):
            """
            asum_class = p_asum_2(self)
            Defined at library.fpp lines 52-58
            
            Parameters
            ----------
            self : Atype
            
            Returns
            -------
            asum_class : float32
            """
            asum_class = _library.f90wrap_test__p_asum_2__binding__atype(self=self._handle)
            return asum_class
        
        def asum_class(self, interface_call=False):
            """
            asum_class = asum_class(self)
            Defined at library.fpp lines 52-58
            
            Parameters
            ----------
            self : Atype
            
            Returns
            -------
            asum_class : float32
            """
            asum_class = \
                _library.f90wrap_test__asum_class__binding__atype(self=self._handle)
            return asum_class
        
        def p_reset(self, value, interface_call=False):
            """
            p_reset(self, value)
            Defined at library.fpp lines 60-66
            
            Parameters
            ----------
            self : Atype
            value : int32
            """
            _library.f90wrap_test__p_reset__binding__atype(self=self._handle, value=value)
        
        def assignment(self, *args, **kwargs):
            """
            Binding(name=assignment(=))
            Defined at library.fpp line 19
            """
            for proc in [self.p_reset]:
                exception=None
                try:
                    return proc(*args, **kwargs, interface_call=True)
                except (TypeError, ValueError, AttributeError, IndexError, \
                    numpy.exceptions.ComplexWarning) as err:
                    exception = "'%s: %s'" % (type(err).__name__, str(err))
                    continue
            
            argTypes=[]
            for arg in args:
                try:
                    argTypes.append("%s: dims '%s', type '%s',"
                    " type code '%s'"
                    %(str(type(arg)),arg.ndim, arg.dtype, arg.dtype.num))
                except AttributeError:
                    argTypes.append(str(type(arg)))
            raise TypeError("Not able to call a version of "
                "assignment compatible with the provided args:"
                "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
        
        @property
        def array(self):
            """
            Element array ftype=integer pytype=int
            Defined at library.fpp line 12
            """
            array_ndim, array_type, array_shape, array_handle = \
                _library.f90wrap_test__atype__array__array(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                array = self._arrays[array_hash]
            else:
                try:
                    array = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _library.f90wrap_test__atype__array__array)
                except TypeError:
                    array = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = array
            return array
        
        @array.setter
        def array(self, array):
            self.array[...] = array
        
        def __str__(self):
            ret = ['<atype>{\n']
            ret.append('    array : ')
            ret.append(repr(self.array))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("library.btype")
    class btype(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=btype)
        Defined at library.fpp lines 21-24
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for btype
            
            self = Btype()
            Defined at library.fpp lines 21-24
            
            Returns
            -------
            this : Btype
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _library.f90wrap_test__btype_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for btype
            
            Destructor for class Btype
            Defined at library.fpp lines 21-24
            
            Parameters
            ----------
            this : Btype
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _library.f90wrap_test__btype_finalise(this=self._handle)
        
        def p_asum(self, interface_call=False):
            """
            bsum_class = p_asum(self)
            Defined at library.fpp lines 68-74
            
            Parameters
            ----------
            self : Btype
            
            Returns
            -------
            bsum_class : float32
            """
            bsum_class = _library.f90wrap_test__p_asum__binding__btype(self=self._handle)
            return bsum_class
        
        @property
        def array(self):
            """
            Element array ftype=integer                      pytype=int
            Defined at library.fpp line 22
            """
            array_ndim, array_type, array_shape, array_handle = \
                _library.f90wrap_test__btype__array__array(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                array = self._arrays[array_hash]
            else:
                try:
                    array = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _library.f90wrap_test__btype__array__array)
                except TypeError:
                    array = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = array
            return array
        
        @array.setter
        def array(self, array):
            self.array[...] = array
        
        def __str__(self):
            ret = ['<btype>{\n']
            ret.append('    array : ')
            ret.append(repr(self.array))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def create(self, n, interface_call=False):
        """
        create(self, n)
        Defined at library.fpp lines 27-34
        
        Parameters
        ----------
        self : Atype
        n : int32
        """
        _library.f90wrap_test__create(self=self._handle, n=n)
    
    @staticmethod
    def asum(self, interface_call=False):
        """
        asum = asum(self)
        Defined at library.fpp lines 44-50
        
        Parameters
        ----------
        self : Atype
        
        Returns
        -------
        asum : float32
        """
        asum = _library.f90wrap_test__asum(self=self._handle)
        return asum
    
    _dt_array_initialisers = []
    
    if not hasattr(_library, "f90wrap_test__p_create__binding__atype"):
        for _candidate in ["f90wrap_test__p_create__binding__atype"]:
            if hasattr(_library, _candidate):
                setattr(_library, "f90wrap_test__p_create__binding__atype", getattr(_library, \
                    _candidate))
                break
    if not hasattr(_library, "f90wrap_test__p_asum__binding__atype"):
        for _candidate in ["f90wrap_test__p_asum__binding__atype"]:
            if hasattr(_library, _candidate):
                setattr(_library, "f90wrap_test__p_asum__binding__atype", getattr(_library, \
                    _candidate))
                break
    if not hasattr(_library, "f90wrap_test__p_asum_2__binding__atype"):
        for _candidate in ["f90wrap_test__p_asum_2__binding__atype"]:
            if hasattr(_library, _candidate):
                setattr(_library, "f90wrap_test__p_asum_2__binding__atype", getattr(_library, \
                    _candidate))
                break
    if not hasattr(_library, "f90wrap_test__asum_class__binding__atype"):
        for _candidate in ["f90wrap_test__asum_class__binding__atype"]:
            if hasattr(_library, _candidate):
                setattr(_library, "f90wrap_test__asum_class__binding__atype", getattr(_library, \
                    _candidate))
                break
    if not hasattr(_library, "f90wrap_test__p_reset__binding__atype"):
        for _candidate in ["f90wrap_test__p_reset__binding__atype"]:
            if hasattr(_library, _candidate):
                setattr(_library, "f90wrap_test__p_reset__binding__atype", getattr(_library, \
                    _candidate))
                break
    if not hasattr(_library, "f90wrap_test__p_asum__binding__btype"):
        for _candidate in ["f90wrap_test__p_asum__binding__btype"]:
            if hasattr(_library, _candidate):
                setattr(_library, "f90wrap_test__p_asum__binding__btype", getattr(_library, \
                    _candidate))
                break
    
    @staticmethod
    def asum_class(instance, *args, **kwargs):
        return instance.asum_class(*args, **kwargs)
    

test = Test()

