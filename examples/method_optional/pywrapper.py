from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings

class M_Array(f90wrap.runtime.FortranModule):
    """
    Module m_array
    Defined at main.fpp lines 5-29
    """
    @f90wrap.runtime.register_class("pywrapper.Array")
    class Array(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=array)
        Defined at main.fpp lines 8-13
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for array
            
            self = Array()
            Defined at main.fpp lines 8-13
            
            Returns
            -------
            this : Array
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_array__array_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for array
            
            Destructor for class Array
            Defined at main.fpp lines 8-13
            
            Parameters
            ----------
            this : Array
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_array__array_finalise(this=self._handle)
        
        def init(self, n, interface_call=False):
            """
            init(self, n)
            Defined at main.fpp lines 16-21
            
            Parameters
            ----------
            this : Array
            n : int32
            """
            _pywrapper.f90wrap_m_array__init__binding__array(this=self._handle, n=n)
        
        def init_optional(self, n, optional_arg=None, interface_call=False):
            """
            init_optional(self, n[, optional_arg])
            Defined at main.fpp lines 23-29
            
            Parameters
            ----------
            this : Array
            n : int32
            optional_arg : Array
            """
            _pywrapper.f90wrap_m_array__init_optional__binding__array(this=self._handle, \
                n=n, optional_arg=None if optional_arg is None else optional_arg._handle)
        
        @property
        def buffer(self):
            """
            Element buffer ftype=real pytype=float
            Defined at main.fpp line 9
            """
            array_ndim, array_type, array_shape, array_handle = \
                _pywrapper.f90wrap_m_array__array__array__buffer(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                buffer = self._arrays[array_hash]
            else:
                try:
                    buffer = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _pywrapper.f90wrap_m_array__array__array__buffer)
                except TypeError:
                    buffer = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = buffer
            return buffer
        
        @buffer.setter
        def buffer(self, buffer):
            self.buffer[...] = buffer
        
        @property
        def array_size(self):
            """
            Element array_size ftype=integer  pytype=int
            Defined at main.fpp line 10
            """
            return _pywrapper.f90wrap_m_array__array__get__array_size(self._handle)
        
        @array_size.setter
        def array_size(self, array_size):
            _pywrapper.f90wrap_m_array__array__set__array_size(self._handle, array_size)
        
        def __str__(self):
            ret = ['<array>{\n']
            ret.append('    buffer : ')
            ret.append(repr(self.buffer))
            ret.append(',\n    array_size : ')
            ret.append(repr(self.array_size))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    if not hasattr(_pywrapper, "f90wrap_m_array__init__binding__array"):
        for _candidate in ["f90wrap_m_array__init__binding__array"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_array__init__binding__array", getattr(_pywrapper, \
                    _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_array__init_optional__binding__array"):
        for _candidate in ["f90wrap_m_array__init_optional__binding__array"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_array__init_optional__binding__array", \
                    getattr(_pywrapper, _candidate))
                break
    
    @staticmethod
    def init(instance, *args, **kwargs):
        return instance.init(*args, **kwargs)
    
    @staticmethod
    def init_optional(instance, *args, **kwargs):
        return instance.init_optional(*args, **kwargs)
    

m_array = M_Array()

