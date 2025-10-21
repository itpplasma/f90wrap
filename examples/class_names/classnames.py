from __future__ import print_function, absolute_import, division
import _classnames
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_classnames = _SafeDirectCExecutor(_classnames, \
    module_import_name='_classnames')

class ModuleSnake(f90wrap.runtime.FortranModule):
    """
    Module module_snake_mod
    Defined at test.fpp lines 5-16
    """
    @f90wrap.runtime.register_class("classnames.IAmACamel")
    class IAmACamel(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=ceci_ne_pas_un_chameau)
        Defined at test.fpp lines 6-7
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for ceci_ne_pas_un_chameau
            
            self = Ceci_Ne_Pas_Un_Chameau()
            Defined at test.fpp lines 6-7
            
            Returns
            -------
            this : Ceci_Ne_Pas_Un_Chameau
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = \
                    _classnames.f90wrap_module_snake_mod__ceci_ne_pas_un_chameau_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for ceci_ne_pas_un_chameau
            
            Destructor for class Ceci_Ne_Pas_Un_Chameau
            Defined at test.fpp lines 6-7
            
            Parameters
            ----------
            this : Ceci_Ne_Pas_Un_Chameau
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _classnames.f90wrap_module_snake_mod__ceci_ne_pas_un_chameau_finalise(this=self._handle)
        
        @property
        def y(self):
            """
            Element y ftype=integer  pytype=int
            Defined at test.fpp line 7
            """
            return \
                _classnames.f90wrap_module_snake_mod__ceci_ne_pas_un_chameau__get__y(self._handle)
        
        @y.setter
        def y(self, y):
            _classnames.f90wrap_module_snake_mod__ceci_ne_pas_un_chameau__set__y(self._handle, \
                y)
        
        def __str__(self):
            ret = ['<ceci_ne_pas_un_chameau>{\n']
            ret.append('    y : ')
            ret.append(repr(self.y))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("classnames.ArrayType")
    class ArrayType(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=array_type)
        Defined at test.fpp lines 9-10
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for array_type
            
            self = Array_Type()
            Defined at test.fpp lines 9-10
            
            Returns
            -------
            this : Array_Type
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _classnames.f90wrap_module_snake_mod__array_type_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for array_type
            
            Destructor for class Array_Type
            Defined at test.fpp lines 9-10
            
            Parameters
            ----------
            this : Array_Type
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _classnames.f90wrap_module_snake_mod__array_type_finalise(this=self._handle)
        
        def init_array_x(self):
            self.x = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _classnames.f90wrap_module_snake_mod__array_type__array_getitem__x,
                                                _classnames.f90wrap_module_snake_mod__array_type__array_setitem__x,
                                                _classnames.f90wrap_module_snake_mod__array_type__array_len__x,
                                                """
            Element x ftype=type(ceci_ne_pas_un_chameau) pytype=Ceci_Ne_Pas_Un_Chameau
            Defined at test.fpp line 10
            """, ModuleSnake.IAmACamel)
            return self.x
        
        _dt_array_initialisers = [init_array_x]
        
    
    @staticmethod
    def recup_point(self, interface_call=False):
        """
        recup_point(self)
        Defined at test.fpp lines 14-16
        
        Parameters
        ----------
        x : Array_Type
        """
        _classnames.f90wrap_module_snake_mod__recup_point(x=self._handle)
    
    def init_array_xarr(self):
        self.xarr = f90wrap.runtime.FortranDerivedTypeArray(f90wrap.runtime.empty_type,
                                            _classnames.f90wrap_module_snake_mod__array_getitem__xarr,
                                            _classnames.f90wrap_module_snake_mod__array_setitem__xarr,
                                            _classnames.f90wrap_module_snake_mod__array_len__xarr,
                                            """
        Element xarr ftype=type(array_type) pytype=Array_Type
        Defined at test.fpp line 12
        """, ModuleSnake.ArrayType)
        return self.xarr
    
    _dt_array_initialisers = [init_array_xarr]
    
    

module_snake_mod = ModuleSnake()

