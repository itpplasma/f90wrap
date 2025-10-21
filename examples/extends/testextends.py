from __future__ import print_function, absolute_import, division
import _testextends
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_testextends = _SafeDirectCExecutor(_testextends, \
    module_import_name='_testextends')

class Testextends_Mod(f90wrap.runtime.FortranModule):
    """
    Module testextends_mod
    Defined at testextends.fpp lines 5-16
    """
    @f90wrap.runtime.register_class("testextends.Superclass")
    class Superclass(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=superclass)
        Defined at testextends.fpp lines 8-10
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for superclass
            
            self = Superclass()
            Defined at testextends.fpp lines 8-10
            
            Returns
            -------
            this : Superclass
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _testextends.f90wrap_testextends_mod__superclass_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for superclass
            
            Destructor for class Superclass
            Defined at testextends.fpp lines 8-10
            
            Parameters
            ----------
            this : Superclass
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _testextends.f90wrap_testextends_mod__superclass_finalise(this=self._handle)
        
        @property
        def stop_at(self):
            """
            Element stop_at ftype=integer  pytype=int
            Defined at testextends.fpp line 10
            """
            return \
                _testextends.f90wrap_testextends_mod__superclass__get__stop_at(self._handle)
        
        @stop_at.setter
        def stop_at(self, stop_at):
            _testextends.f90wrap_testextends_mod__superclass__set__stop_at(self._handle, \
                stop_at)
        
        def __str__(self):
            ret = ['<superclass>{\n']
            ret.append('    stop_at : ')
            ret.append(repr(self.stop_at))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("testextends.Subclass1")
    class Subclass1(Superclass):
        """
        Type(name=subclass1)
        Defined at testextends.fpp lines 12-13
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for subclass1
            
            self = Subclass1()
            Defined at testextends.fpp lines 12-13
            
            Returns
            -------
            this : Subclass1
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _testextends.f90wrap_testextends_mod__subclass1_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for subclass1
            
            Destructor for class Subclass1
            Defined at testextends.fpp lines 12-13
            
            Parameters
            ----------
            this : Subclass1
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _testextends.f90wrap_testextends_mod__subclass1_finalise(this=self._handle)
        
        @property
        def nl(self):
            """
            Element nl ftype=integer  pytype=int
            Defined at testextends.fpp line 13
            """
            return _testextends.f90wrap_testextends_mod__subclass1__get__nl(self._handle)
        
        @nl.setter
        def nl(self, nl):
            _testextends.f90wrap_testextends_mod__subclass1__set__nl(self._handle, nl)
        
        def __str__(self):
            ret = ['<subclass1>{\n']
            ret.append('    nl : ')
            ret.append(repr(self.nl))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("testextends.Subclass2")
    class Subclass2(Superclass):
        """
        Type(name=subclass2)
        Defined at testextends.fpp lines 15-16
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for subclass2
            
            self = Subclass2()
            Defined at testextends.fpp lines 15-16
            
            Returns
            -------
            this : Subclass2
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _testextends.f90wrap_testextends_mod__subclass2_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for subclass2
            
            Destructor for class Subclass2
            Defined at testextends.fpp lines 15-16
            
            Parameters
            ----------
            this : Subclass2
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _testextends.f90wrap_testextends_mod__subclass2_finalise(this=self._handle)
        
        @property
        def nl(self):
            """
            Element nl ftype=integer  pytype=int
            Defined at testextends.fpp line 16
            """
            return _testextends.f90wrap_testextends_mod__subclass2__get__nl(self._handle)
        
        @nl.setter
        def nl(self, nl):
            _testextends.f90wrap_testextends_mod__subclass2__set__nl(self._handle, nl)
        
        def __str__(self):
            ret = ['<subclass2>{\n']
            ret.append('    nl : ')
            ret.append(repr(self.nl))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    

testextends_mod = Testextends_Mod()

