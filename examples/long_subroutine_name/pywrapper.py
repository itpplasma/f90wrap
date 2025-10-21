from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

class M_Long_Subroutine_Name(f90wrap.runtime.FortranModule):
    """
    Module m_long_subroutine_name
    Defined at main.fpp lines 5-16
    """
    @f90wrap.runtime.register_class("pywrapper.m_long_subroutine_name_type")
    class m_long_subroutine_name_type(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=m_long_subroutine_name_type)
        Defined at main.fpp lines 8-10
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for m_long_subroutine_name_type
            
            self = M_Long_Subroutine_Name_Type()
            Defined at main.fpp lines 8-10
            
            Returns
            -------
            this : M_Long_Subroutine_Name_Type
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = \
                    _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_typefcc3()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for m_long_subroutine_name_type
            
            Destructor for class M_Long_Subroutine_Name_Type
            Defined at main.fpp lines 8-10
            
            Parameters
            ----------
            this : M_Long_Subroutine_Name_Type
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_type6ffd(this=self._handle)
        
        @property
        def m_long_subroutine_name_type_integer(self):
            """
            Element m_long_subroutine_name_type_integer ftype=integer pytype=int
            Defined at main.fpp line 9
            """
            return \
                _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_type2f88(self._handle)
        
        @m_long_subroutine_name_type_integer.setter
        def m_long_subroutine_name_type_integer(self, \
            m_long_subroutine_name_type_integer):
            _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_typebfce(self._handle, \
                m_long_subroutine_name_type_integer)
        
        @property
        def m_long_subroutine_name_type_integer_array(self):
            """
            Element m_long_subroutine_name_type_integer_array ftype=integer pytype=int
            Defined at main.fpp line 10
            """
            array_ndim, array_type, array_shape, array_handle = \
                _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_typeeaf3(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                m_long_subroutine_name_type_integer_array = self._arrays[array_hash]
            else:
                try:
                    m_long_subroutine_name_type_integer_array = \
                        f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_typeeaf3)
                except TypeError:
                    m_long_subroutine_name_type_integer_array = \
                        f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = m_long_subroutine_name_type_integer_array
            return m_long_subroutine_name_type_integer_array
        
        @m_long_subroutine_name_type_integer_array.setter
        def m_long_subroutine_name_type_integer_array(self, \
            m_long_subroutine_name_type_integer_array):
            self.m_long_subroutine_name_type_integer_array[...] = \
                m_long_subroutine_name_type_integer_array
        
        def __str__(self):
            ret = ['<m_long_subroutine_name_type>{\n']
            ret.append('    m_long_subroutine_name_type_integer : ')
            ret.append(repr(self.m_long_subroutine_name_type_integer))
            ret.append(',\n    m_long_subroutine_name_type_integer_array : ')
            ret.append(repr(self.m_long_subroutine_name_type_integer_array))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.m_long_subroutine_name_type_2")
    class m_long_subroutine_name_type_2(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=m_long_subroutine_name_type_2)
        Defined at main.fpp lines 12-13
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for m_long_subroutine_name_type_2
            
            self = M_Long_Subroutine_Name_Type_2()
            Defined at main.fpp lines 12-13
            
            Returns
            -------
            this : M_Long_Subroutine_Name_Type_2
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = \
                    _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_typebe6a()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for m_long_subroutine_name_type_2
            
            Destructor for class M_Long_Subroutine_Name_Type_2
            Defined at main.fpp lines 12-13
            
            Parameters
            ----------
            this : M_Long_Subroutine_Name_Type_2
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_type1fc3(this=self._handle)
        
        def init_array_m_long_subroutine_name_type_2_type_array(self):
            self.m_long_subroutine_name_type_2_type_array = \
                f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_type4adb,
                                                _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_type97d9,
                                                _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_type5005,
                                                """
            Element m_long_subroutine_name_type_2_type_array \
                ftype=type(m_long_subroutine_name_type) pytype=M_Long_Subroutine_Name_Type
            Defined at main.fpp line 13
            """, M_Long_Subroutine_Name.m_long_subroutine_name_type)
            return self.m_long_subroutine_name_type_2_type_array
        
        _dt_array_initialisers = [init_array_m_long_subroutine_name_type_2_type_array]
        
    
    @staticmethod
    def m_long_subroutine_name_subroutine(interface_call=False):
        """
        m_long_subroutine_name_subroutine()
        Defined at main.fpp lines 16-16
        
        """
        _pywrapper.f90wrap_m_long_subroutine_name__m_long_subroutine_name_subra0ea()
    
    @property
    def m_long_subroutine_name_integer(self):
        """
        Element m_long_subroutine_name_integer ftype=integer  pytype=int
        Defined at main.fpp line 7
        """
        return \
            _pywrapper.f90wrap_m_long_subroutine_name__get__m_long_subroutine_namebc01()
    
    @m_long_subroutine_name_integer.setter
    def m_long_subroutine_name_integer(self, m_long_subroutine_name_integer):
        _pywrapper.f90wrap_m_long_subroutine_name__set__m_long_subroutine_name860c(m_long_subroutine_name_integer)
    
    def get_m_long_subroutine_name_integer(self):
        return self.m_long_subroutine_name_integer
    
    def set_m_long_subroutine_name_integer(self, value):
        self.m_long_subroutine_name_integer = value
    
    def __str__(self):
        ret = ['<m_long_subroutine_name>{\n']
        ret.append('    m_long_subroutine_name_integer : ')
        ret.append(repr(self.m_long_subroutine_name_integer))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    
    

m_long_subroutine_name = M_Long_Subroutine_Name()

