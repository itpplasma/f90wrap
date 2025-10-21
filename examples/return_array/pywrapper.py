from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

class M_Test(f90wrap.runtime.FortranModule):
    """
    Module m_test
    Defined at main.fpp lines 5-126
    """
    @f90wrap.runtime.register_class("pywrapper.t_array_wrapper")
    class t_array_wrapper(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_array_wrapper)
        Defined at main.fpp lines 8-10
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_array_wrapper
            
            self = T_Array_Wrapper()
            Defined at main.fpp lines 8-10
            
            Returns
            -------
            this : T_Array_Wrapper
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_test__t_array_wrapper_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_array_wrapper
            
            Destructor for class T_Array_Wrapper
            Defined at main.fpp lines 8-10
            
            Parameters
            ----------
            this : T_Array_Wrapper
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_test__t_array_wrapper_finalise(this=self._handle)
        
        @property
        def a_size(self):
            """
            Element a_size ftype=integer  pytype=int
            Defined at main.fpp line 9
            """
            return _pywrapper.f90wrap_m_test__t_array_wrapper__get__a_size(self._handle)
        
        @a_size.setter
        def a_size(self, a_size):
            _pywrapper.f90wrap_m_test__t_array_wrapper__set__a_size(self._handle, a_size)
        
        @property
        def a_data(self):
            """
            Element a_data ftype=real pytype=float
            Defined at main.fpp line 10
            """
            array_ndim, array_type, array_shape, array_handle = \
                _pywrapper.f90wrap_m_test__t_array_wrapper__array__a_data(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                a_data = self._arrays[array_hash]
            else:
                try:
                    a_data = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _pywrapper.f90wrap_m_test__t_array_wrapper__array__a_data)
                except TypeError:
                    a_data = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = a_data
            return a_data
        
        @a_data.setter
        def a_data(self, a_data):
            self.a_data[...] = a_data
        
        def __str__(self):
            ret = ['<t_array_wrapper>{\n']
            ret.append('    a_size : ')
            ret.append(repr(self.a_size))
            ret.append(',\n    a_data : ')
            ret.append(repr(self.a_data))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.t_array_2d_wrapper")
    class t_array_2d_wrapper(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_array_2d_wrapper)
        Defined at main.fpp lines 12-14
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_array_2d_wrapper
            
            self = T_Array_2D_Wrapper()
            Defined at main.fpp lines 12-14
            
            Returns
            -------
            this : T_Array_2D_Wrapper
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_test__t_array_2d_wrapper_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_array_2d_wrapper
            
            Destructor for class T_Array_2D_Wrapper
            Defined at main.fpp lines 12-14
            
            Parameters
            ----------
            this : T_Array_2D_Wrapper
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_test__t_array_2d_wrapper_finalise(this=self._handle)
        
        @property
        def a_size_x(self):
            """
            Element a_size_x ftype=integer  pytype=int
            Defined at main.fpp line 13
            """
            return \
                _pywrapper.f90wrap_m_test__t_array_2d_wrapper__get__a_size_x(self._handle)
        
        @a_size_x.setter
        def a_size_x(self, a_size_x):
            _pywrapper.f90wrap_m_test__t_array_2d_wrapper__set__a_size_x(self._handle, \
                a_size_x)
        
        @property
        def a_size_y(self):
            """
            Element a_size_y ftype=integer  pytype=int
            Defined at main.fpp line 13
            """
            return \
                _pywrapper.f90wrap_m_test__t_array_2d_wrapper__get__a_size_y(self._handle)
        
        @a_size_y.setter
        def a_size_y(self, a_size_y):
            _pywrapper.f90wrap_m_test__t_array_2d_wrapper__set__a_size_y(self._handle, \
                a_size_y)
        
        @property
        def a_data(self):
            """
            Element a_data ftype=real pytype=float
            Defined at main.fpp line 14
            """
            array_ndim, array_type, array_shape, array_handle = \
                _pywrapper.f90wrap_m_test__t_array_2d_wrapper__array__a_data(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                a_data = self._arrays[array_hash]
            else:
                try:
                    a_data = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _pywrapper.f90wrap_m_test__t_array_2d_wrapper__array__a_data)
                except TypeError:
                    a_data = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = a_data
            return a_data
        
        @a_data.setter
        def a_data(self, a_data):
            self.a_data[...] = a_data
        
        def __str__(self):
            ret = ['<t_array_2d_wrapper>{\n']
            ret.append('    a_size_x : ')
            ret.append(repr(self.a_size_x))
            ret.append(',\n    a_size_y : ')
            ret.append(repr(self.a_size_y))
            ret.append(',\n    a_data : ')
            ret.append(repr(self.a_data))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.t_array_double_wrapper")
    class t_array_double_wrapper(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_array_double_wrapper)
        Defined at main.fpp lines 16-17
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_array_double_wrapper
            
            self = T_Array_Double_Wrapper()
            Defined at main.fpp lines 16-17
            
            Returns
            -------
            this : T_Array_Double_Wrapper
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_test__t_array_double_wrapper_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_array_double_wrapper
            
            Destructor for class T_Array_Double_Wrapper
            Defined at main.fpp lines 16-17
            
            Parameters
            ----------
            this : T_Array_Double_Wrapper
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_test__t_array_double_wrapper_finalise(this=self._handle)
        
        @property
        def array_wrapper(self):
            """
            Element array_wrapper ftype=type(t_array_wrapper) pytype=T_Array_Wrapper
            Defined at main.fpp line 17
            """
            array_wrapper_handle = \
                _pywrapper.f90wrap_m_test__t_array_double_wrapper__get__array_wrapper(self._handle)
            if tuple(array_wrapper_handle) in self._objs:
                array_wrapper = self._objs[tuple(array_wrapper_handle)]
            else:
                array_wrapper = m_test.t_array_wrapper.from_handle(array_wrapper_handle)
                self._objs[tuple(array_wrapper_handle)] = array_wrapper
            return array_wrapper
        
        @array_wrapper.setter
        def array_wrapper(self, array_wrapper):
            array_wrapper = array_wrapper._handle
            _pywrapper.f90wrap_m_test__t_array_double_wrapper__set__array_wrapper(self._handle, \
                array_wrapper)
        
        def __str__(self):
            ret = ['<t_array_double_wrapper>{\n']
            ret.append('    array_wrapper : ')
            ret.append(repr(self.array_wrapper))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.t_value")
    class t_value(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_value)
        Defined at main.fpp lines 19-20
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_value
            
            self = T_Value()
            Defined at main.fpp lines 19-20
            
            Returns
            -------
            this : T_Value
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_test__t_value_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_value
            
            Destructor for class T_Value
            Defined at main.fpp lines 19-20
            
            Parameters
            ----------
            this : T_Value
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_test__t_value_finalise(this=self._handle)
        
        @property
        def value(self):
            """
            Element value ftype=real  pytype=float
            Defined at main.fpp line 20
            """
            return _pywrapper.f90wrap_m_test__t_value__get__value(self._handle)
        
        @value.setter
        def value(self, value):
            _pywrapper.f90wrap_m_test__t_value__set__value(self._handle, value)
        
        def __str__(self):
            ret = ['<t_value>{\n']
            ret.append('    value : ')
            ret.append(repr(self.value))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.t_size_2d")
    class t_size_2d(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_size_2d)
        Defined at main.fpp lines 22-23
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_size_2d
            
            self = T_Size_2D()
            Defined at main.fpp lines 22-23
            
            Returns
            -------
            this : T_Size_2D
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_test__t_size_2d_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_size_2d
            
            Destructor for class T_Size_2D
            Defined at main.fpp lines 22-23
            
            Parameters
            ----------
            this : T_Size_2D
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_test__t_size_2d_finalise(this=self._handle)
        
        @property
        def x(self):
            """
            Element x ftype=integer  pytype=int
            Defined at main.fpp line 23
            """
            return _pywrapper.f90wrap_m_test__t_size_2d__get__x(self._handle)
        
        @x.setter
        def x(self, x):
            _pywrapper.f90wrap_m_test__t_size_2d__set__x(self._handle, x)
        
        @property
        def y(self):
            """
            Element y ftype=integer  pytype=int
            Defined at main.fpp line 23
            """
            return _pywrapper.f90wrap_m_test__t_size_2d__get__y(self._handle)
        
        @y.setter
        def y(self, y):
            _pywrapper.f90wrap_m_test__t_size_2d__set__y(self._handle, y)
        
        def __str__(self):
            ret = ['<t_size_2d>{\n']
            ret.append('    x : ')
            ret.append(repr(self.x))
            ret.append(',\n    y : ')
            ret.append(repr(self.y))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def array_init(self, in_size, interface_call=False):
        """
        array_init(self, in_size)
        Defined at main.fpp lines 41-46
        
        Parameters
        ----------
        in_array : T_Array_Wrapper
        in_size : int32
        """
        _pywrapper.f90wrap_m_test__array_init(in_array=self._handle, in_size=in_size)
    
    @staticmethod
    def array_2d_init(self, in_size_x, in_size_y, interface_call=False):
        """
        array_2d_init(self, in_size_x, in_size_y)
        Defined at main.fpp lines 48-54
        
        Parameters
        ----------
        in_array : T_Array_2D_Wrapper
        in_size_x : int32
        in_size_y : int32
        """
        _pywrapper.f90wrap_m_test__array_2d_init(in_array=self._handle, \
            in_size_x=in_size_x, in_size_y=in_size_y)
    
    @staticmethod
    def array_wrapper_init(self, in_size, interface_call=False):
        """
        array_wrapper_init(self, in_size)
        Defined at main.fpp lines 56-61
        
        Parameters
        ----------
        in_wrapper : T_Array_Double_Wrapper
        in_size : int32
        """
        _pywrapper.f90wrap_m_test__array_wrapper_init(in_wrapper=self._handle, \
            in_size=in_size)
    
    @staticmethod
    def array_free(self, interface_call=False):
        """
        array_free(self)
        Defined at main.fpp lines 63-66
        
        Parameters
        ----------
        in_array : T_Array_Wrapper
        """
        _pywrapper.f90wrap_m_test__array_free(in_array=self._handle)
    
    @staticmethod
    def return_scalar(self, interface_call=False):
        """
        return_scalar = return_scalar(self)
        Defined at main.fpp lines 68-71
        
        Parameters
        ----------
        in_array : T_Array_Wrapper
        
        Returns
        -------
        return_scalar : float32
        """
        return_scalar = _pywrapper.f90wrap_m_test__return_scalar(in_array=self._handle)
        return return_scalar
    
    @staticmethod
    def return_hard_coded_1d(interface_call=False):
        """
        retval = return_hard_coded_1d()
        Defined at main.fpp lines 73-75
        
        Returns
        -------
        retval : float array
        """
        retval = _pywrapper.f90wrap_m_test__return_hard_coded_1d()
        return retval
    
    @staticmethod
    def return_hard_coded_2d(interface_call=False):
        """
        retval = return_hard_coded_2d()
        Defined at main.fpp lines 77-79
        
        Returns
        -------
        retval : float array
        """
        retval = _pywrapper.f90wrap_m_test__return_hard_coded_2d()
        return retval
    
    @staticmethod
    def return_array_member(self, interface_call=False):
        """
        retval = return_array_member(self)
        Defined at main.fpp lines 81-84
        
        Parameters
        ----------
        in_array : T_Array_Wrapper
        
        Returns
        -------
        retval : float array
        """
        retval = _pywrapper.f90wrap_m_test__return_array_member(in_array=self._handle, \
            f90wrap_n0=self.a_size)
        return retval
    
    @staticmethod
    def return_array_member_2d(self, interface_call=False):
        """
        retval = return_array_member_2d(self)
        Defined at main.fpp lines 86-89
        
        Parameters
        ----------
        in_array : T_Array_2D_Wrapper
        
        Returns
        -------
        retval : float array
        """
        retval = \
            _pywrapper.f90wrap_m_test__return_array_member_2d(in_array=self._handle, \
            f90wrap_n0=self.a_size_x, f90wrap_n1=self.a_size_y)
        return retval
    
    @staticmethod
    def return_array_member_wrapper(self, interface_call=False):
        """
        retval = return_array_member_wrapper(self)
        Defined at main.fpp lines 91-94
        
        Parameters
        ----------
        in_wrapper : T_Array_Double_Wrapper
        
        Returns
        -------
        retval : float array
        """
        retval = \
            _pywrapper.f90wrap_m_test__return_array_member_wrapper(in_wrapper=self._handle, \
            f90wrap_n0=self.array_wrapper.a_size)
        return retval
    
    @staticmethod
    def return_array_input(in_len, interface_call=False):
        """
        retval = return_array_input(in_len)
        Defined at main.fpp lines 96-99
        
        Parameters
        ----------
        in_len : int32
        
        Returns
        -------
        retval : float array
        """
        retval = _pywrapper.f90wrap_m_test__return_array_input(in_len=in_len, \
            f90wrap_n0=in_len)
        return retval
    
    @staticmethod
    def return_array_input_2d(in_len_x, in_len_y, interface_call=False):
        """
        retval = return_array_input_2d(in_len_x, in_len_y)
        Defined at main.fpp lines 101-104
        
        Parameters
        ----------
        in_len_x : int32
        in_len_y : int32
        
        Returns
        -------
        retval : float array
        """
        retval = _pywrapper.f90wrap_m_test__return_array_input_2d(in_len_x=in_len_x, \
            in_len_y=in_len_y, f90wrap_n0=in_len_x, f90wrap_n1=in_len_y)
        return retval
    
    @staticmethod
    def return_array_size(in_array, interface_call=False):
        """
        retval = return_array_size(in_array)
        Defined at main.fpp lines 106-109
        
        Parameters
        ----------
        in_array : float array
        
        Returns
        -------
        retval : float array
        """
        retval = _pywrapper.f90wrap_m_test__return_array_size(in_array=in_array, \
            f90wrap_n1=in_array.shape[0])
        return retval
    
    @staticmethod
    def return_array_size_2d_in(in_array, interface_call=False):
        """
        retval = return_array_size_2d_in(in_array)
        Defined at main.fpp lines 111-114
        
        Parameters
        ----------
        in_array : float array
        
        Returns
        -------
        retval : float array
        """
        retval = _pywrapper.f90wrap_m_test__return_array_size_2d_in(in_array=in_array, \
            f90wrap_n2=in_array.shape[1])
        return retval
    
    @staticmethod
    def return_array_size_2d_out(in_array_1, in_array_2, interface_call=False):
        """
        retval = return_array_size_2d_out(in_array_1, in_array_2)
        Defined at main.fpp lines 116-120
        
        Parameters
        ----------
        in_array_1 : float array
        in_array_2 : float array
        
        Returns
        -------
        retval : float array
        """
        retval = \
            _pywrapper.f90wrap_m_test__return_array_size_2d_out(in_array_1=in_array_1, \
            in_array_2=in_array_2, f90wrap_n4=in_array_1.shape[0], \
            f90wrap_n5=in_array_2.shape[1])
        return retval
    
    @staticmethod
    def return_derived_type_value(self, size_2d, interface_call=False):
        """
        output = return_derived_type_value(self, size_2d)
        Defined at main.fpp lines 122-126
        
        Parameters
        ----------
        this : T_Value
        size_2d : T_Size_2D
        
        Returns
        -------
        output : float array
        """
        output = _pywrapper.f90wrap_m_test__return_derived_type_value(this=self._handle, \
            size_2d=size_2d._handle, f90wrap_n0=size_2d.x, f90wrap_n1=size_2d.y)
        return output
    
    _dt_array_initialisers = []
    
    

m_test = M_Test()

