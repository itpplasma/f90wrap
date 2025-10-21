"""
Module m_inheritance
Defined at inheritance_type.fpp lines 5-11
"""
from __future__ import print_function, absolute_import, division
from .. import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from . import m_base_type

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("pywrapper.t_inheritance")
class t_inheritance(m_base_type.t_base_type):
    """
    Type(name=t_inheritance)
    Defined at inheritance_type.fpp lines 9-10
    """
    def __init__(self, handle=None):
        """
        Automatically generated constructor for t_inheritance
        
        self = T_Inheritance()
        Defined at inheritance_type.fpp lines 9-10
        
        Returns
        -------
        this : T_Inheritance
            Object to be constructed
        
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        if handle is not None:
            self._handle = handle
            self._alloc = True
        else:
            result = _pywrapper.f90wrap_m_inheritance__t_inheritance_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
            self._alloc = True
    
    def __del__(self):
        """
        Automatically generated destructor for t_inheritance
        
        Destructor for class T_Inheritance
        Defined at inheritance_type.fpp lines 9-10
        
        Parameters
        ----------
        this : T_Inheritance
            Object to be destructed
        
        """
        if getattr(self, '_alloc', False):
            _pywrapper.f90wrap_m_inheritance__t_inheritance_finalise(this=self._handle)
    
    @property
    def integer_number(self):
        """
        Element integer_number ftype=integer  pytype=int
        Defined at inheritance_type.fpp line 10
        """
        return \
            _pywrapper.f90wrap_m_inheritance__t_inheritance__get__integer_number(self._handle)
    
    @integer_number.setter
    def integer_number(self, integer_number):
        _pywrapper.f90wrap_m_inheritance__t_inheritance__set__integer_number(self._handle, \
            integer_number)
    
    def __str__(self):
        ret = ['<t_inheritance>{\n']
        ret.append('    integer_number : ')
        ret.append(repr(self.integer_number))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []


try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module \
        "m_inheritance".')

for func in _dt_array_initialisers:
    func()
