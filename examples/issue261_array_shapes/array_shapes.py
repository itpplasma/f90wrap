from __future__ import print_function, absolute_import, division
import _array_shapes
import f90wrap.runtime
import logging
import numpy
import warnings

class Array_Shapes(f90wrap.runtime.FortranModule):
    """
    Module array_shapes
    Defined at array_shapes.fpp lines 5-121
    """
    @f90wrap.runtime.register_class("array_shapes.container")
    class container(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=container)
        Defined at array_shapes.fpp lines 7-9
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for container
            
            self = Container()
            Defined at array_shapes.fpp lines 7-9
            
            Returns
            -------
            this : Container
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _array_shapes.f90wrap_array_shapes__container_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for container
            
            Destructor for class Container
            Defined at array_shapes.fpp lines 7-9
            
            Parameters
            ----------
            this : Container
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _array_shapes.f90wrap_array_shapes__container_finalise(this=self._handle)
        
        @property
        def n_data(self):
            """
            Element n_data ftype=integer  pytype=int
            Defined at array_shapes.fpp line 8
            """
            return _array_shapes.f90wrap_array_shapes__container__get__n_data(self._handle)
        
        @n_data.setter
        def n_data(self, n_data):
            _array_shapes.f90wrap_array_shapes__container__set__n_data(self._handle, n_data)
        
        @property
        def data(self):
            """
            Element data ftype=real pytype=float
            Defined at array_shapes.fpp line 9
            """
            array_ndim, array_type, array_shape, array_handle = \
                _array_shapes.f90wrap_array_shapes__container__array__data(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                data = self._arrays[array_hash]
            else:
                try:
                    data = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _array_shapes.f90wrap_array_shapes__container__array__data)
                except TypeError:
                    data = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = data
            return data
        
        @data.setter
        def data(self, data):
            self.data[...] = data
        
        def __str__(self):
            ret = ['<container>{\n']
            ret.append('    n_data : ')
            ret.append(repr(self.n_data))
            ret.append(',\n    data : ')
            ret.append(repr(self.data))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def one_array_dynamic(x, interface_call=False):
        """
        res = one_array_dynamic(x)
        Defined at array_shapes.fpp lines 12-15
        
        Parameters
        ----------
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__one_array_dynamic(x=x, \
            f90wrap_n1=x.shape[0])
        return res
    
    @staticmethod
    def one_array_fixed(x, interface_call=False):
        """
        res = one_array_fixed(x)
        Defined at array_shapes.fpp lines 17-20
        
        Parameters
        ----------
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__one_array_fixed(x=x)
        return res
    
    @staticmethod
    def one_array_fixed_range(x, interface_call=False):
        """
        res = one_array_fixed_range(x)
        Defined at array_shapes.fpp lines 22-25
        
        Parameters
        ----------
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__one_array_fixed_range(x=x)
        return res
    
    @staticmethod
    def one_array_explicit(x, n, interface_call=False):
        """
        res = one_array_explicit(x, n)
        Defined at array_shapes.fpp lines 27-31
        
        Parameters
        ----------
        x : float array
        n : int32
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__one_array_explicit(x=x, n=n, \
            f90wrap_n1=n)
        return res
    
    @staticmethod
    def one_array_explicit_range(x, n, interface_call=False):
        """
        res = one_array_explicit_range(x, n)
        Defined at array_shapes.fpp lines 33-37
        
        Parameters
        ----------
        x : float array
        n : int32
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__one_array_explicit_range(x=x, n=n, \
            f90wrap_n1=n)
        return res
    
    @staticmethod
    def two_arrays_dynamic(y, x, interface_call=False):
        """
        res = two_arrays_dynamic(y, x)
        Defined at array_shapes.fpp lines 39-43
        
        Parameters
        ----------
        y : float array
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__two_arrays_dynamic(y=y, x=x, \
            f90wrap_n2=x.shape[0])
        return res
    
    @staticmethod
    def two_arrays_fixed(y, x, interface_call=False):
        """
        res = two_arrays_fixed(y, x)
        Defined at array_shapes.fpp lines 45-49
        
        Parameters
        ----------
        y : float array
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__two_arrays_fixed(y=y, x=x)
        return res
    
    @staticmethod
    def two_arrays_mixed(y, x, interface_call=False):
        """
        res = two_arrays_mixed(y, x)
        Defined at array_shapes.fpp lines 51-55
        
        Parameters
        ----------
        y : float array
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__two_arrays_mixed(y=y, x=x, \
            f90wrap_n1=x.shape[0])
        return res
    
    @staticmethod
    def two_arrays_2d_dynamic(y, x, interface_call=False):
        """
        res = two_arrays_2d_dynamic(y, x)
        Defined at array_shapes.fpp lines 58-65
        
        Parameters
        ----------
        y : float array
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__two_arrays_2d_dynamic(y=y, x=x, \
            f90wrap_n2=x.shape[0], f90wrap_n3=y.shape[0])
        return res
    
    @staticmethod
    def two_arrays_2d_fixed(y, x, interface_call=False):
        """
        res = two_arrays_2d_fixed(y, x)
        Defined at array_shapes.fpp lines 67-74
        
        Parameters
        ----------
        y : float array
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__two_arrays_2d_fixed(y=y, x=x)
        return res
    
    @staticmethod
    def two_arrays_2d_fixed_whitespace(y, x, interface_call=False):
        """
        res = two_arrays_2d_fixed_whitespace(y, x)
        Defined at array_shapes.fpp lines 76-83
        
        Parameters
        ----------
        y : float array
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__two_arrays_2d_fixed_whitespace(y=y, \
            x=x)
        return res
    
    @staticmethod
    def two_arrays_2d_mixed(y, x, interface_call=False):
        """
        res = two_arrays_2d_mixed(y, x)
        Defined at array_shapes.fpp lines 85-92
        
        Parameters
        ----------
        y : float array
        x : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__two_arrays_2d_mixed(y=y, x=x, \
            f90wrap_n1=x.shape[0])
        return res
    
    @staticmethod
    def get_container(x, interface_call=False):
        """
        c = get_container(x)
        Defined at array_shapes.fpp lines 95-99
        
        Parameters
        ----------
        x : float array
        
        Returns
        -------
        c : Container
        """
        c = _array_shapes.f90wrap_array_shapes__get_container(x=x)
        c = f90wrap.runtime.lookup_class("array_shapes.container").from_handle(c, \
            alloc=True)
        return c
    
    @staticmethod
    def array_container_dynamic(self, y, interface_call=False):
        """
        res = array_container_dynamic(self, y)
        Defined at array_shapes.fpp lines 101-105
        
        Parameters
        ----------
        c : Container
        y : float array
        
        Returns
        -------
        res : float array
        """
        res = \
            _array_shapes.f90wrap_array_shapes__array_container_dynamic(c=self._handle, \
            y=y, f90wrap_n1=self.n_data)
        return res
    
    @staticmethod
    def array_container_fixed(self, y, interface_call=False):
        """
        res = array_container_fixed(self, y)
        Defined at array_shapes.fpp lines 107-111
        
        Parameters
        ----------
        c : Container
        y : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__array_container_fixed(c=self._handle, \
            y=y, f90wrap_n0=self.n_data)
        return res
    
    @staticmethod
    def array_container_dynamic_2d(n, c, y, interface_call=False):
        """
        res = array_container_dynamic_2d(n, c, y)
        Defined at array_shapes.fpp lines 113-121
        
        Parameters
        ----------
        n : int32
        c : Container
        y : float array
        
        Returns
        -------
        res : float array
        """
        res = _array_shapes.f90wrap_array_shapes__array_container_dynamic_2d(n=n, \
            c=c._handle, y=y, f90wrap_n1=c.n_data, f90wrap_n2=n)
        return res
    
    _dt_array_initialisers = []
    
    

array_shapes = Array_Shapes()

