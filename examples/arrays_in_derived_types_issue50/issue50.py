from __future__ import print_function, absolute_import, division
import _issue50
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_issue50 = _SafeDirectCExecutor(_issue50, module_import_name='_issue50')

class Module_Test(f90wrap.runtime.FortranModule):
    """
    Module module_test
    Defined at test.fpp lines 5-15
    """
    @f90wrap.runtime.register_class("issue50.real_array")
    class real_array(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=real_array)
        Defined at test.fpp lines 6-7
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for real_array
            
            self = Real_Array()
            Defined at test.fpp lines 6-7
            
            Returns
            -------
            this : Real_Array
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _issue50.f90wrap_module_test__real_array_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for real_array
            
            Destructor for class Real_Array
            Defined at test.fpp lines 6-7
            
            Parameters
            ----------
            this : Real_Array
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _issue50.f90wrap_module_test__real_array_finalise(this=self._handle)
        
        @property
        def item(self):
            """
            Element item ftype=real pytype=float
            Defined at test.fpp line 7
            """
            array_ndim, array_type, array_shape, array_handle = \
                _issue50.f90wrap_module_test__real_array__array__item(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                item = self._arrays[array_hash]
            else:
                try:
                    item = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _issue50.f90wrap_module_test__real_array__array__item)
                except TypeError:
                    item = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = item
            return item
        
        @item.setter
        def item(self, item):
            self.item[...] = item
        
        def __str__(self):
            ret = ['<real_array>{\n']
            ret.append('    item : ')
            ret.append(repr(self.item))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def testf(self, interface_call=False):
        """
        testf(self)
        Defined at test.fpp lines 10-15
        
        Parameters
        ----------
        x : Real_Array
        """
        _issue50.f90wrap_module_test__testf(x=self._handle)
    
    _dt_array_initialisers = []
    
    

module_test = Module_Test()

