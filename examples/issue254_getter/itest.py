from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy
import warnings

class Kimdispersionequation_Module(f90wrap.runtime.FortranModule):
    """
    Module kimdispersionequation_module
    Defined at KIMDispersionEquation.fpp lines 5-23
    """
    @f90wrap.runtime.register_class("itest.OptionsType")
    class OptionsType(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=optionstype)
        Defined at KIMDispersionEquation.fpp lines 8-10
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for optionstype
            
            self = Optionstype()
            Defined at KIMDispersionEquation.fpp lines 8-10
            
            Returns
            -------
            this : Optionstype
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _itest.f90wrap_kimdispersionequation_module__optionstype_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for optionstype
            
            Destructor for class Optionstype
            Defined at KIMDispersionEquation.fpp lines 8-10
            
            Parameters
            ----------
            this : Optionstype
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _itest.f90wrap_kimdispersionequation_module__optionstype_finalise(this=self._handle)
        
        @property
        def omega(self):
            """
            Element omega ftype=real(8) pytype=float
            Defined at KIMDispersionEquation.fpp line 9
            """
            return \
                _itest.f90wrap_kimdispersionequation_module__optionstype__get__omega(self._handle)
        
        @omega.setter
        def omega(self, omega):
            _itest.f90wrap_kimdispersionequation_module__optionstype__set__omega(self._handle, \
                omega)
        
        def __str__(self):
            ret = ['<optionstype>{\n']
            ret.append('    omega : ')
            ret.append(repr(self.omega))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("itest.KIMDispersionEquation")
    class KIMDispersionEquation(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=kimdispersionequation)
        Defined at KIMDispersionEquation.fpp lines 13-15
        """
        def __init__(self):
            raise(NotImplementedError("This is an abstract class"))
        
        def initialize(self, options, interface_call=False):
            """
            initialize(self, options)
            Defined at KIMDispersionEquation.fpp lines 19-22
            
            Parameters
            ----------
            this : Kimdispersionequation
            options : Optionstype
            """
            _itest.f90wrap_kimdispersionequation_module__initialize__binding__5dd3(this=self._handle, \
                options=options._handle)
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    if not hasattr(_itest, \
        "f90wrap_kimdispersionequation_module__initialize__binding__5dd3"):
        for _candidate in \
            ["f90wrap_kimdispersionequation_module__initialize__binding__5dd3"]:
            if hasattr(_itest, _candidate):
                setattr(_itest, \
                    "f90wrap_kimdispersionequation_module__initialize__binding__5dd3", \
                    getattr(_itest, _candidate))
                break
    
    @staticmethod
    def initialize(instance, *args, **kwargs):
        return instance.initialize(*args, **kwargs)
    

kimdispersionequation_module = Kimdispersionequation_Module()

class Kimdispersion_Horton_Module(f90wrap.runtime.FortranModule):
    """
    Module kimdispersion_horton_module
    Defined at KIMDispersion_Horton.fpp lines 5-17
    """
    @f90wrap.runtime.register_class("itest.KIMDispersion_Horton")
    class KIMDispersion_Horton(kimdispersionequation_module.KIMDispersionEquation):
        """
        Type(name=kimdispersion_horton)
        Defined at KIMDispersion_Horton.fpp lines 8-11
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for kimdispersion_horton
            
            self = Kimdispersion_Horton()
            Defined at KIMDispersion_Horton.fpp lines 8-11
            
            Returns
            -------
            this : Kimdispersion_Horton
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = \
                    _itest.f90wrap_kimdispersion_horton_module__kimdispersion_horton_ib155()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for kimdispersion_horton
            
            Destructor for class Kimdispersion_Horton
            Defined at KIMDispersion_Horton.fpp lines 8-11
            
            Parameters
            ----------
            this : Kimdispersion_Horton
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _itest.f90wrap_kimdispersion_horton_module__kimdispersion_horton_fa9f5(this=self._handle)
        
        def initialize(self, options, interface_call=False):
            """
            initialize(self, options)
            Defined at KIMDispersion_Horton.fpp lines 14-17
            
            Parameters
            ----------
            this : Kimdispersion_Horton
            options : Optionstype
            """
            _itest.f90wrap_kimdispersion_horton_module__initialize__binding__k2119(this=self._handle, \
                options=options._handle)
        
        @property
        def options(self):
            """
            Element options ftype=type(optionstype) pytype=Optionstype
            Defined at KIMDispersion_Horton.fpp line 9
            """
            options_handle = \
                _itest.f90wrap_kimdispersion_horton_module__kimdispersion_horton__get__options(self._handle)
            if tuple(options_handle) in self._objs:
                options = self._objs[tuple(options_handle)]
            else:
                options = kimdispersionequation_module.OptionsType.from_handle(options_handle)
                self._objs[tuple(options_handle)] = options
            return options
        
        @options.setter
        def options(self, options):
            options = options._handle
            _itest.f90wrap_kimdispersion_horton_module__kimdispersion_horton__set__options(self._handle, \
                options)
        
        def __str__(self):
            ret = ['<kimdispersion_horton>{\n']
            ret.append('    options : ')
            ret.append(repr(self.options))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    if not hasattr(_itest, \
        "f90wrap_kimdispersion_horton_module__initialize__binding__k2119"):
        for _candidate in \
            ["f90wrap_kimdispersion_horton_module__initialize__binding__k2119"]:
            if hasattr(_itest, _candidate):
                setattr(_itest, \
                    "f90wrap_kimdispersion_horton_module__initialize__binding__k2119", \
                    getattr(_itest, _candidate))
                break
    
    @staticmethod
    def initialize(instance, *args, **kwargs):
        return instance.initialize(*args, **kwargs)
    

kimdispersion_horton_module = Kimdispersion_Horton_Module()

