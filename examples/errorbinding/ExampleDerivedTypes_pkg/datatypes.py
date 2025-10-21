"""
Module datatypes
Defined at datatypes.fpp lines 6-40
"""
from __future__ import print_function, absolute_import, division
import _ExampleDerivedTypes_pkg
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("ExampleDerivedTypes_pkg.typewithprocedure")
class typewithprocedure(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=typewithprocedure)
    Defined at datatypes.fpp lines 11-16
    """
    def __init__(self, handle=None):
        """
        Automatically generated constructor for typewithprocedure
        
        self = Typewithprocedure()
        Defined at datatypes.fpp lines 11-16
        
        Returns
        -------
        this : Typewithprocedure
            Object to be constructed
        
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        if handle is not None:
            self._handle = handle
            self._alloc = True
        else:
            result = \
                _ExampleDerivedTypes_pkg.f90wrap_datatypes__typewithprocedure_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
            self._alloc = True
    
    def __del__(self):
        """
        Automatically generated destructor for typewithprocedure
        
        Destructor for class Typewithprocedure
        Defined at datatypes.fpp lines 11-16
        
        Parameters
        ----------
        this : Typewithprocedure
            Object to be destructed
        
        """
        if getattr(self, '_alloc', False):
            _ExampleDerivedTypes_pkg.f90wrap_datatypes__typewithprocedure_finalise(this=self._handle)
    
    def init(self, a, n, interface_call=False):
        """
        init(self, a, n)
        Defined at datatypes.fpp lines 19-24
        
        Parameters
        ----------
        this : Typewithprocedure
        a : float64
        n : int32
        """
        _ExampleDerivedTypes_pkg.f90wrap_datatypes__init__binding__typewithprocedure(this=self._handle, \
            a=a, n=n)
    
    def info(self, lun, interface_call=False):
        """
        info(self, lun)
        Defined at datatypes.fpp lines 26-29
        
        Parameters
        ----------
        this : Typewithprocedure
        lun : int32
        """
        _ExampleDerivedTypes_pkg.f90wrap_datatypes__info__binding__typewithprocedure(this=self._handle, \
            lun=lun)
    
    @property
    def a(self):
        """
        Element a ftype=real(idp) pytype=float
        Defined at datatypes.fpp line 12
        """
        return \
            _ExampleDerivedTypes_pkg.f90wrap_datatypes__typewithprocedure__get__a(self._handle)
    
    @a.setter
    def a(self, a):
        _ExampleDerivedTypes_pkg.f90wrap_datatypes__typewithprocedure__set__a(self._handle, \
            a)
    
    @property
    def n(self):
        """
        Element n ftype=integer(4) pytype=int
        Defined at datatypes.fpp line 13
        """
        return \
            _ExampleDerivedTypes_pkg.f90wrap_datatypes__typewithprocedure__get__n(self._handle)
    
    @n.setter
    def n(self, n):
        _ExampleDerivedTypes_pkg.f90wrap_datatypes__typewithprocedure__set__n(self._handle, \
            n)
    
    def __str__(self):
        ret = ['<typewithprocedure>{\n']
        ret.append('    a : ')
        ret.append(repr(self.a))
        ret.append(',\n    n : ')
        ret.append(repr(self.n))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

def constructor_typewithprocedure(self, a, n, interface_call=False):
    """
    constructor_typewithprocedure(self, a, n)
    Defined at datatypes.fpp lines 31-35
    
    Parameters
    ----------
    this : Typewithprocedure
    a : float64
    n : int32
    """
    _ExampleDerivedTypes_pkg.f90wrap_datatypes__constructor_typewithprocedure(this=self._handle, \
        a=a, n=n)

def info_typewithprocedure(self, lun, interface_call=False):
    """
    info_typewithprocedure(self, lun)
    Defined at datatypes.fpp lines 37-40
    
    Parameters
    ----------
    this : Typewithprocedure
    lun : int32
    """
    _ExampleDerivedTypes_pkg.f90wrap_datatypes__info_typewithprocedure(this=self._handle, \
        lun=lun)


_array_initialisers = []
_dt_array_initialisers = []

if not hasattr(_ExampleDerivedTypes_pkg, \
    "f90wrap_datatypes__init__binding__typewithprocedure"):
    for _candidate in ["f90wrap_datatypes__init__binding__typewithprocedure"]:
        if hasattr(_ExampleDerivedTypes_pkg, _candidate):
            setattr(_ExampleDerivedTypes_pkg, \
                "f90wrap_datatypes__init__binding__typewithprocedure", \
                getattr(_ExampleDerivedTypes_pkg, _candidate))
            break
if not hasattr(_ExampleDerivedTypes_pkg, \
    "f90wrap_datatypes__info__binding__typewithprocedure"):
    for _candidate in ["f90wrap_datatypes__info__binding__typewithprocedure"]:
        if hasattr(_ExampleDerivedTypes_pkg, _candidate):
            setattr(_ExampleDerivedTypes_pkg, \
                "f90wrap_datatypes__info__binding__typewithprocedure", \
                getattr(_ExampleDerivedTypes_pkg, _candidate))
            break

@staticmethod
def info(instance, *args, **kwargs):
    return instance.info(*args, **kwargs)

@staticmethod
def init(instance, *args, **kwargs):
    return instance.init(*args, **kwargs)

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "datatypes".')

for func in _dt_array_initialisers:
    func()
