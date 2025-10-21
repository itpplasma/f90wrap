"""
Module m_composition
Defined at composition_type.fpp lines 5-11
"""
from __future__ import print_function, absolute_import, division
from .. import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from .m_base_type import t_base_type
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("pywrapper.t_composition")
class t_composition(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=t_composition)
    Defined at composition_type.fpp lines 9-10
    """
    def __init__(self, handle=None):
        """
        Automatically generated constructor for t_composition
        
        self = T_Composition()
        Defined at composition_type.fpp lines 9-10
        
        Returns
        -------
        this : T_Composition
            Object to be constructed
        
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        if handle is not None:
            self._handle = handle
            self._alloc = True
        else:
            result = _pywrapper.f90wrap_m_composition__t_composition_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
            self._alloc = True
    
    def __del__(self):
        """
        Automatically generated destructor for t_composition
        
        Destructor for class T_Composition
        Defined at composition_type.fpp lines 9-10
        
        Parameters
        ----------
        this : T_Composition
            Object to be destructed
        
        """
        if getattr(self, '_alloc', False):
            _pywrapper.f90wrap_m_composition__t_composition_finalise(this=self._handle)
    
    @property
    def member(self):
        """
        Element member ftype=type(t_base_type) pytype=T_Base_Type
        Defined at composition_type.fpp line 10
        """
        member_handle = \
            _pywrapper.f90wrap_m_composition__t_composition__get__member(self._handle)
        if tuple(member_handle) in self._objs:
            member = self._objs[tuple(member_handle)]
        else:
            member = t_base_type.from_handle(member_handle)
            self._objs[tuple(member_handle)] = member
        return member
    
    @member.setter
    def member(self, member):
        member = member._handle
        _pywrapper.f90wrap_m_composition__t_composition__set__member(self._handle, \
            member)
    
    def __str__(self):
        ret = ['<t_composition>{\n']
        ret.append('    member : ')
        ret.append(repr(self.member))
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
        "m_composition".')

for func in _dt_array_initialisers:
    func()
