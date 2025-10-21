"""
Module m_base_type
Defined at base_type.fpp lines 5-10
"""
from __future__ import print_function, absolute_import, division
from .. import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("pywrapper.t_base_type")
class t_base_type(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=t_base_type)
    Defined at base_type.fpp lines 8-9
    """
    def __init__(self, handle=None):
        """
        Automatically generated constructor for t_base_type
        
        self = T_Base_Type()
        Defined at base_type.fpp lines 8-9
        
        Returns
        -------
        this : T_Base_Type
            Object to be constructed
        
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        if handle is not None:
            self._handle = handle
            self._alloc = True
        else:
            result = _pywrapper.f90wrap_m_base_type__t_base_type_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
            self._alloc = True
    
    def __del__(self):
        """
        Automatically generated destructor for t_base_type
        
        Destructor for class T_Base_Type
        Defined at base_type.fpp lines 8-9
        
        Parameters
        ----------
        this : T_Base_Type
            Object to be destructed
        
        """
        if getattr(self, '_alloc', False):
            _pywrapper.f90wrap_m_base_type__t_base_type_finalise(this=self._handle)
    
    @property
    def real_number(self):
        """
        Element real_number ftype=real  pytype=float
        Defined at base_type.fpp line 9
        """
        return \
            _pywrapper.f90wrap_m_base_type__t_base_type__get__real_number(self._handle)
    
    @real_number.setter
    def real_number(self, real_number):
        _pywrapper.f90wrap_m_base_type__t_base_type__set__real_number(self._handle, \
            real_number)
    
    def __str__(self):
        ret = ['<t_base_type>{\n']
        ret.append('    real_number : ')
        ret.append(repr(self.real_number))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []


try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "m_base_type".')

for func in _dt_array_initialisers:
    func()
