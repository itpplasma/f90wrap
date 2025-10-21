from __future__ import print_function, absolute_import, division
import _test_python
import f90wrap.runtime
import logging
import numpy
import warnings

class Test_Module(f90wrap.runtime.FortranModule):
    """
    Module test_module
    Defined at functions.fpp lines 5-28
    """
    @f90wrap.runtime.register_class("test_python.test_type2")
    class test_type2(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=test_type2)
        Defined at functions.fpp lines 8-9
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for test_type2
            
            self = Test_Type2()
            Defined at functions.fpp lines 8-9
            
            Returns
            -------
            this : Test_Type2
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _test_python.f90wrap_test_module__test_type2_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for test_type2
            
            Destructor for class Test_Type2
            Defined at functions.fpp lines 8-9
            
            Parameters
            ----------
            this : Test_Type2
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _test_python.f90wrap_test_module__test_type2_finalise(this=self._handle)
        
        @property
        def y(self):
            """
            Element y ftype=real pytype=float
            Defined at functions.fpp line 9
            """
            array_ndim, array_type, array_shape, array_handle = \
                _test_python.f90wrap_test_module__test_type2__array__y(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                y = self._arrays[array_hash]
            else:
                try:
                    y = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _test_python.f90wrap_test_module__test_type2__array__y)
                except TypeError:
                    y = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = y
            return y
        
        @y.setter
        def y(self, y):
            self.y[...] = y
        
        def __str__(self):
            ret = ['<test_type2>{\n']
            ret.append('    y : ')
            ret.append(repr(self.y))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("test_python.Test_Type2_Xn_Array")
    class Test_Type2_Xn_Array(f90wrap.runtime.FortranDerivedType):
        """
        super-type
        Automatically generated to handle derived type arrays as a new derived type
        
        Type(name=test_type2_xn_array)
        Defined at functions.fpp lines 8-9
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for test_type2_xn_array
            
            self = Test_Type2_Xn_Array()
            Defined at functions.fpp lines 8-9
            
            Returns
            -------
            this : Test_Type2_Xn_Array
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _test_python.f90wrap_test_module__test_type2_xn_array_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for test_type2_xn_array
            
            Destructor for class Test_Type2_Xn_Array
            Defined at functions.fpp lines 8-9
            
            Parameters
            ----------
            this : Test_Type2_Xn_Array
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _test_python.f90wrap_test_module__test_type2_xn_array_finalise(this=self._handle)
        
        def init_array_items(self):
            self.items = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _test_python.f90wrap_test_module__test_type2_xn_array__array_getitem__items,
                                                _test_python.f90wrap_test_module__test_type2_xn_array__array_setitem__items,
                                                _test_python.f90wrap_test_module__test_type2_xn_array__array_len__items,
                                                """
            Element items ftype=type(test_type2) pytype=Test_Type2
            Defined at  line 0
            """, Test_Module.test_type2)
            return self.items
        
        _dt_array_initialisers = [init_array_items]
        
    
    @f90wrap.runtime.register_class("test_python.Test_Type2_Xm_Array")
    class Test_Type2_Xm_Array(f90wrap.runtime.FortranDerivedType):
        """
        super-type
        Automatically generated to handle derived type arrays as a new derived type
        
        Type(name=test_type2_xm_array)
        Defined at functions.fpp lines 8-9
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for test_type2_xm_array
            
            self = Test_Type2_Xm_Array()
            Defined at functions.fpp lines 8-9
            
            Returns
            -------
            this : Test_Type2_Xm_Array
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _test_python.f90wrap_test_module__test_type2_xm_array_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for test_type2_xm_array
            
            Destructor for class Test_Type2_Xm_Array
            Defined at functions.fpp lines 8-9
            
            Parameters
            ----------
            this : Test_Type2_Xm_Array
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _test_python.f90wrap_test_module__test_type2_xm_array_finalise(this=self._handle)
        
        def init_array_items(self):
            self.items = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _test_python.f90wrap_test_module__test_type2_xm_array__array_getitem__items,
                                                _test_python.f90wrap_test_module__test_type2_xm_array__array_setitem__items,
                                                _test_python.f90wrap_test_module__test_type2_xm_array__array_len__items,
                                                """
            Element items ftype=type(test_type2) pytype=Test_Type2
            Defined at  line 0
            """, Test_Module.test_type2)
            return self.items
        
        _dt_array_initialisers = [init_array_items]
        
    
    @f90wrap.runtime.register_class("test_python.Test_Type2_X5_Array")
    class Test_Type2_X5_Array(f90wrap.runtime.FortranDerivedType):
        """
        super-type
        Automatically generated to handle derived type arrays as a new derived type
        
        Type(name=test_type2_x5_array)
        Defined at functions.fpp lines 8-9
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for test_type2_x5_array
            
            self = Test_Type2_X5_Array()
            Defined at functions.fpp lines 8-9
            
            Returns
            -------
            this : Test_Type2_X5_Array
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _test_python.f90wrap_test_module__test_type2_x5_array_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for test_type2_x5_array
            
            Destructor for class Test_Type2_X5_Array
            Defined at functions.fpp lines 8-9
            
            Parameters
            ----------
            this : Test_Type2_X5_Array
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _test_python.f90wrap_test_module__test_type2_x5_array_finalise(this=self._handle)
        
        def init_array_items(self):
            self.items = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _test_python.f90wrap_test_module__test_type2_x5_array__array_getitem__items,
                                                _test_python.f90wrap_test_module__test_type2_x5_array__array_setitem__items,
                                                _test_python.f90wrap_test_module__test_type2_x5_array__array_len__items,
                                                """
            Element items ftype=type(test_type2) pytype=Test_Type2
            Defined at  line 0
            """, Test_Module.test_type2)
            return self.items
        
        _dt_array_initialisers = [init_array_items]
        
    
    @staticmethod
    def test_routine4(x1, x2, x3, x4, x5, x6, interface_call=False):
        """
        test_routine4(x1, x2, x3, x4, x5, x6)
        Defined at functions.fpp lines 12-27
        
        Parameters
        ----------
        x1 : float array
        x2 : Test_Type2_Xn_Array
            super-type
        
        x3 : Test_Type2_Xn_Array
            super-type
        
        x4 : Test_Type2_Xm_Array
            super-type
        
        x5 : Test_Type2_X5_Array
            super-type
        
        x6 : float32
        """
        _test_python.f90wrap_test_module__test_routine4(x1=x1, x2=x2._handle, \
            x3=x3._handle, x4=x4._handle, x5=x5._handle, x6=x6)
    
    @property
    def m(self):
        """
        Element m ftype=integer pytype=int
        Defined at functions.fpp line 6
        """
        return _test_python.f90wrap_test_module__get__m()
    
    def get_m(self):
        return self.m
    
    @property
    def n(self):
        """
        Element n ftype=integer pytype=int
        Defined at functions.fpp line 7
        """
        return _test_python.f90wrap_test_module__get__n()
    
    def get_n(self):
        return self.n
    
    def __str__(self):
        ret = ['<test_module>{\n']
        ret.append('    m : ')
        ret.append(repr(self.m))
        ret.append(',\n    n : ')
        ret.append(repr(self.n))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    
    

test_module = Test_Module()

