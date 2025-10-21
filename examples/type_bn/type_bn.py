from __future__ import print_function, absolute_import, division
import _type_bn
import f90wrap.runtime
import logging
import numpy
import warnings

class Module_Structure(f90wrap.runtime.FortranModule):
    """
    Module module_structure
    Defined at type_bn.fpp lines 5-7
    """
    @f90wrap.runtime.register_class("type_bn.type_face")
    class type_face(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=type_face)
        Defined at type_bn.fpp lines 6-7
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for type_face
            
            self = Type_Face()
            Defined at type_bn.fpp lines 6-7
            
            Returns
            -------
            this : Type_Face
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _type_bn.f90wrap_module_structure__type_face_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for type_face
            
            Destructor for class Type_Face
            Defined at type_bn.fpp lines 6-7
            
            Parameters
            ----------
            this : Type_Face
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _type_bn.f90wrap_module_structure__type_face_finalise(this=self._handle)
        
        @property
        def type_bn(self):
            """
            Element type_bn ftype=integer  pytype=int
            Defined at type_bn.fpp line 7
            """
            return _type_bn.f90wrap_module_structure__type_face__get__type_bn(self._handle)
        
        @type_bn.setter
        def type_bn(self, type_bn):
            _type_bn.f90wrap_module_structure__type_face__set__type_bn(self._handle, \
                type_bn)
        
        def __str__(self):
            ret = ['<type_face>{\n']
            ret.append('    type_bn : ')
            ret.append(repr(self.type_bn))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    

module_structure = Module_Structure()

