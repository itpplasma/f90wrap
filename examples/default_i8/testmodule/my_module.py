"""
Module my_module
Defined at test.fpp lines 5-24
"""
from __future__ import print_function, absolute_import, division
import _testmodule
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("testmodule.mytype")
class mytype(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=mytype)
    Defined at test.fpp lines 7-9
    """
    def __init__(self, handle=None):
        """
        Automatically generated constructor for mytype
        
        self = Mytype()
        Defined at test.fpp lines 7-9
        
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
            result = _testmodule.f90wrap_my_module__mytype_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
            self._alloc = True
    
    def __del__(self):
        """
        Automatically generated destructor for mytype
        
        Destructor for class Mytype
        Defined at test.fpp lines 7-9
        
        Parameters
        ----------
        this : Mytype
            Object to be destructed
        
        """
        if getattr(self, '_alloc', False):
            _testmodule.f90wrap_my_module__mytype_finalise(this=self._handle)
    
    @property
    def n(self):
        """
        Element n ftype=integer  pytype=int
        Defined at test.fpp line 8
        """
        return _testmodule.f90wrap_my_module__mytype__get__n(self._handle)
    
    @n.setter
    def n(self, n):
        _testmodule.f90wrap_my_module__mytype__set__n(self._handle, n)
    
    @property
    def m(self):
        """
        Element m ftype=integer  pytype=int
        Defined at test.fpp line 8
        """
        return _testmodule.f90wrap_my_module__mytype__get__m(self._handle)
    
    @m.setter
    def m(self, m):
        _testmodule.f90wrap_my_module__mytype__set__m(self._handle, m)
    
    @property
    def y(self):
        """
        Element y ftype=real(8) pytype=float
        Defined at test.fpp line 9
        """
        array_ndim, array_type, array_shape, array_handle = \
            _testmodule.f90wrap_my_module__mytype__array__y(self._handle)
        array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
        if array_hash in self._arrays:
            y = self._arrays[array_hash]
        else:
            try:
                y = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                        self._handle,
                                        _testmodule.f90wrap_my_module__mytype__array__y)
            except TypeError:
                y = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
            self._arrays[array_handle] = y
        return y
    
    @y.setter
    def y(self, y):
        self.y[...] = y
    
    def __str__(self):
        ret = ['<mytype>{\n']
        ret.append('    n : ')
        ret.append(repr(self.n))
        ret.append(',\n    m : ')
        ret.append(repr(self.m))
        ret.append(',\n    y : ')
        ret.append(repr(self.y))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

def allocit(self, n, m, interface_call=False):
    """
    allocit(self, n, m)
    Defined at test.fpp lines 12-24
    
    Parameters
    ----------
    x : Mytype
    n : int64
    m : int64
    """
    _testmodule.f90wrap_my_module__allocit(x=self._handle, n=n, m=m)


_array_initialisers = []
_dt_array_initialisers = []


try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "my_module".')

for func in _dt_array_initialisers:
    func()
