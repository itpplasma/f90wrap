from __future__ import print_function, absolute_import, division
import _ExampleDerivedTypes
import f90wrap.runtime
import logging
import numpy
import warnings

class Datatypes_Allocatable(f90wrap.runtime.FortranModule):
    """
    Module datatypes_allocatable
    Defined at datatypes.fpp lines 6-27
    """
    @f90wrap.runtime.register_class("ExampleDerivedTypes.alloc_arrays")
    class alloc_arrays(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=alloc_arrays)
        Defined at datatypes.fpp lines 10-14
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for alloc_arrays
            
            self = Alloc_Arrays()
            Defined at datatypes.fpp lines 10-14
            
            Returns
            -------
            this : Alloc_Arrays
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = \
                    _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for alloc_arrays
            
            Destructor for class Alloc_Arrays
            Defined at datatypes.fpp lines 10-14
            
            Parameters
            ----------
            this : Alloc_Arrays
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays_finalise(this=self._handle)
        
        @property
        def chi(self):
            """
            Element chi ftype=real(idp) pytype=float
            Defined at datatypes.fpp line 11
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays__array__chi(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                chi = self._arrays[array_hash]
            else:
                try:
                    chi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays__array__chi)
                except TypeError:
                    chi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = chi
            return chi
        
        @chi.setter
        def chi(self, chi):
            self.chi[...] = chi
        
        @property
        def psi(self):
            """
            Element psi ftype=real(idp) pytype=float
            Defined at datatypes.fpp line 12
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays__array__psi(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                psi = self._arrays[array_hash]
            else:
                try:
                    psi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays__array__psi)
                except TypeError:
                    psi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = psi
            return psi
        
        @psi.setter
        def psi(self, psi):
            self.psi[...] = psi
        
        @property
        def chi_shape(self):
            """
            Element chi_shape ftype=integer(4) pytype=int
            Defined at datatypes.fpp line 13
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays__array__chi_shape(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                chi_shape = self._arrays[array_hash]
            else:
                try:
                    chi_shape = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays__array__chi_shape)
                except TypeError:
                    chi_shape = f90wrap.runtime.direct_c_array(array_type, array_shape, \
                        array_handle)
                self._arrays[array_handle] = chi_shape
            return chi_shape
        
        @chi_shape.setter
        def chi_shape(self, chi_shape):
            self.chi_shape[...] = chi_shape
        
        @property
        def psi_shape(self):
            """
            Element psi_shape ftype=integer(4) pytype=int
            Defined at datatypes.fpp line 14
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays__array__psi_shape(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                psi_shape = self._arrays[array_hash]
            else:
                try:
                    psi_shape = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes_allocatable__alloc_arrays__array__psi_shape)
                except TypeError:
                    psi_shape = f90wrap.runtime.direct_c_array(array_type, array_shape, \
                        array_handle)
                self._arrays[array_handle] = psi_shape
            return psi_shape
        
        @psi_shape.setter
        def psi_shape(self, psi_shape):
            self.psi_shape[...] = psi_shape
        
        def __str__(self):
            ret = ['<alloc_arrays>{\n']
            ret.append('    chi : ')
            ret.append(repr(self.chi))
            ret.append(',\n    psi : ')
            ret.append(repr(self.psi))
            ret.append(',\n    chi_shape : ')
            ret.append(repr(self.chi_shape))
            ret.append(',\n    psi_shape : ')
            ret.append(repr(self.psi_shape))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def init_alloc_arrays(self, m, n, interface_call=False):
        """
        init_alloc_arrays(self, m, n)
        Defined at datatypes.fpp lines 17-21
        
        Parameters
        ----------
        dertype : Alloc_Arrays
        m : int32
        n : int32
        """
        _ExampleDerivedTypes.f90wrap_datatypes_allocatable__init_alloc_arrays(dertype=self._handle, \
            m=m, n=n)
    
    @staticmethod
    def destroy_alloc_arrays(self, interface_call=False):
        """
        destroy_alloc_arrays(self)
        Defined at datatypes.fpp lines 23-26
        
        Parameters
        ----------
        dertype : Alloc_Arrays
        """
        _ExampleDerivedTypes.f90wrap_datatypes_allocatable__destroy_alloc_arrays(dertype=self._handle)
    
    _dt_array_initialisers = []
    
    

datatypes_allocatable = Datatypes_Allocatable()

class Datatypes(f90wrap.runtime.FortranModule):
    """
    Module datatypes
    Defined at datatypes.fpp lines 31-89
    """
    @f90wrap.runtime.register_class("ExampleDerivedTypes.different_types")
    class different_types(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=different_types)
        Defined at datatypes.fpp lines 42-45
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for different_types
            
            self = Different_Types()
            Defined at datatypes.fpp lines 42-45
            
            Returns
            -------
            this : Different_Types
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _ExampleDerivedTypes.f90wrap_datatypes__different_types_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for different_types
            
            Destructor for class Different_Types
            Defined at datatypes.fpp lines 42-45
            
            Parameters
            ----------
            this : Different_Types
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _ExampleDerivedTypes.f90wrap_datatypes__different_types_finalise(this=self._handle)
        
        @property
        def alpha(self):
            """
            Element alpha ftype=logical pytype=bool
            Defined at datatypes.fpp line 43
            """
            return \
                _ExampleDerivedTypes.f90wrap_datatypes__different_types__get__alpha(self._handle)
        
        @alpha.setter
        def alpha(self, alpha):
            _ExampleDerivedTypes.f90wrap_datatypes__different_types__set__alpha(self._handle, \
                alpha)
        
        @property
        def beta(self):
            """
            Element beta ftype=integer(4) pytype=int
            Defined at datatypes.fpp line 44
            """
            return \
                _ExampleDerivedTypes.f90wrap_datatypes__different_types__get__beta(self._handle)
        
        @beta.setter
        def beta(self, beta):
            _ExampleDerivedTypes.f90wrap_datatypes__different_types__set__beta(self._handle, \
                beta)
        
        @property
        def delta(self):
            """
            Element delta ftype=real(idp) pytype=float
            Defined at datatypes.fpp line 45
            """
            return \
                _ExampleDerivedTypes.f90wrap_datatypes__different_types__get__delta(self._handle)
        
        @delta.setter
        def delta(self, delta):
            _ExampleDerivedTypes.f90wrap_datatypes__different_types__set__delta(self._handle, \
                delta)
        
        def __str__(self):
            ret = ['<different_types>{\n']
            ret.append('    alpha : ')
            ret.append(repr(self.alpha))
            ret.append(',\n    beta : ')
            ret.append(repr(self.beta))
            ret.append(',\n    delta : ')
            ret.append(repr(self.delta))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("ExampleDerivedTypes.fixed_shape_arrays")
    class fixed_shape_arrays(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=fixed_shape_arrays)
        Defined at datatypes.fpp lines 48-51
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for fixed_shape_arrays
            
            self = Fixed_Shape_Arrays()
            Defined at datatypes.fpp lines 48-51
            
            Returns
            -------
            this : Fixed_Shape_Arrays
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _ExampleDerivedTypes.f90wrap_datatypes__fixed_shape_arrays_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for fixed_shape_arrays
            
            Destructor for class Fixed_Shape_Arrays
            Defined at datatypes.fpp lines 48-51
            
            Parameters
            ----------
            this : Fixed_Shape_Arrays
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _ExampleDerivedTypes.f90wrap_datatypes__fixed_shape_arrays_finalise(this=self._handle)
        
        @property
        def eta(self):
            """
            Element eta ftype=integer(4) pytype=int
            Defined at datatypes.fpp line 49
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__fixed_shape_arrays__array__eta(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                eta = self._arrays[array_hash]
            else:
                try:
                    eta = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__fixed_shape_arrays__array__eta)
                except TypeError:
                    eta = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = eta
            return eta
        
        @eta.setter
        def eta(self, eta):
            self.eta[...] = eta
        
        @property
        def theta(self):
            """
            Element theta ftype=real(isp) pytype=float
            Defined at datatypes.fpp line 50
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__fixed_shape_arrays__array__theta(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                theta = self._arrays[array_hash]
            else:
                try:
                    theta = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__fixed_shape_arrays__array__theta)
                except TypeError:
                    theta = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = theta
            return theta
        
        @theta.setter
        def theta(self, theta):
            self.theta[...] = theta
        
        @property
        def iota(self):
            """
            Element iota ftype=real(idp) pytype=float
            Defined at datatypes.fpp line 51
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__fixed_shape_arrays__array__iota(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                iota = self._arrays[array_hash]
            else:
                try:
                    iota = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__fixed_shape_arrays__array__iota)
                except TypeError:
                    iota = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = iota
            return iota
        
        @iota.setter
        def iota(self, iota):
            self.iota[...] = iota
        
        def __str__(self):
            ret = ['<fixed_shape_arrays>{\n']
            ret.append('    eta : ')
            ret.append(repr(self.eta))
            ret.append(',\n    theta : ')
            ret.append(repr(self.theta))
            ret.append(',\n    iota : ')
            ret.append(repr(self.iota))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("ExampleDerivedTypes.nested")
    class nested(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=nested)
        Defined at datatypes.fpp lines 53-55
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for nested
            
            self = Nested()
            Defined at datatypes.fpp lines 53-55
            
            Returns
            -------
            this : Nested
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _ExampleDerivedTypes.f90wrap_datatypes__nested_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for nested
            
            Destructor for class Nested
            Defined at datatypes.fpp lines 53-55
            
            Parameters
            ----------
            this : Nested
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _ExampleDerivedTypes.f90wrap_datatypes__nested_finalise(this=self._handle)
        
        @property
        def mu(self):
            """
            Element mu ftype=type(different_types) pytype=Different_Types
            Defined at datatypes.fpp line 54
            """
            mu_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__nested__get__mu(self._handle)
            if tuple(mu_handle) in self._objs:
                mu = self._objs[tuple(mu_handle)]
            else:
                mu = datatypes.different_types.from_handle(mu_handle)
                self._objs[tuple(mu_handle)] = mu
            return mu
        
        @mu.setter
        def mu(self, mu):
            mu = mu._handle
            _ExampleDerivedTypes.f90wrap_datatypes__nested__set__mu(self._handle, mu)
        
        @property
        def nu(self):
            """
            Element nu ftype=type(fixed_shape_arrays) pytype=Fixed_Shape_Arrays
            Defined at datatypes.fpp line 55
            """
            nu_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__nested__get__nu(self._handle)
            if tuple(nu_handle) in self._objs:
                nu = self._objs[tuple(nu_handle)]
            else:
                nu = datatypes.fixed_shape_arrays.from_handle(nu_handle)
                self._objs[tuple(nu_handle)] = nu
            return nu
        
        @nu.setter
        def nu(self, nu):
            nu = nu._handle
            _ExampleDerivedTypes.f90wrap_datatypes__nested__set__nu(self._handle, nu)
        
        def __str__(self):
            ret = ['<nested>{\n']
            ret.append('    mu : ')
            ret.append(repr(self.mu))
            ret.append(',\n    nu : ')
            ret.append(repr(self.nu))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("ExampleDerivedTypes.pointer_arrays")
    class pointer_arrays(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=pointer_arrays)
        Defined at datatypes.fpp lines 58-62
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for pointer_arrays
            
            self = Pointer_Arrays()
            Defined at datatypes.fpp lines 58-62
            
            Returns
            -------
            this : Pointer_Arrays
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for pointer_arrays
            
            Destructor for class Pointer_Arrays
            Defined at datatypes.fpp lines 58-62
            
            Parameters
            ----------
            this : Pointer_Arrays
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays_finalise(this=self._handle)
        
        @property
        def chi(self):
            """
            Element chi ftype=real(idp) pytype=float
            Defined at datatypes.fpp line 59
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays__array__chi(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                chi = self._arrays[array_hash]
            else:
                try:
                    chi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays__array__chi)
                except TypeError:
                    chi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = chi
            return chi
        
        @chi.setter
        def chi(self, chi):
            self.chi[...] = chi
        
        @property
        def psi(self):
            """
            Element psi ftype=real(idp) pytype=float
            Defined at datatypes.fpp line 60
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays__array__psi(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                psi = self._arrays[array_hash]
            else:
                try:
                    psi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays__array__psi)
                except TypeError:
                    psi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = psi
            return psi
        
        @psi.setter
        def psi(self, psi):
            self.psi[...] = psi
        
        @property
        def chi_shape(self):
            """
            Element chi_shape ftype=integer(4) pytype=int
            Defined at datatypes.fpp line 61
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays__array__chi_shape(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                chi_shape = self._arrays[array_hash]
            else:
                try:
                    chi_shape = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays__array__chi_shape)
                except TypeError:
                    chi_shape = f90wrap.runtime.direct_c_array(array_type, array_shape, \
                        array_handle)
                self._arrays[array_handle] = chi_shape
            return chi_shape
        
        @chi_shape.setter
        def chi_shape(self, chi_shape):
            self.chi_shape[...] = chi_shape
        
        @property
        def psi_shape(self):
            """
            Element psi_shape ftype=integer(4) pytype=int
            Defined at datatypes.fpp line 62
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays__array__psi_shape(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                psi_shape = self._arrays[array_hash]
            else:
                try:
                    psi_shape = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__pointer_arrays__array__psi_shape)
                except TypeError:
                    psi_shape = f90wrap.runtime.direct_c_array(array_type, array_shape, \
                        array_handle)
                self._arrays[array_handle] = psi_shape
            return psi_shape
        
        @psi_shape.setter
        def psi_shape(self, psi_shape):
            self.psi_shape[...] = psi_shape
        
        def __str__(self):
            ret = ['<pointer_arrays>{\n']
            ret.append('    chi : ')
            ret.append(repr(self.chi))
            ret.append(',\n    psi : ')
            ret.append(repr(self.psi))
            ret.append(',\n    chi_shape : ')
            ret.append(repr(self.chi_shape))
            ret.append(',\n    psi_shape : ')
            ret.append(repr(self.psi_shape))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("ExampleDerivedTypes.alloc_arrays_2")
    class alloc_arrays_2(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=alloc_arrays_2)
        Defined at datatypes.fpp lines 65-69
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for alloc_arrays_2
            
            self = Alloc_Arrays_2()
            Defined at datatypes.fpp lines 65-69
            
            Returns
            -------
            this : Alloc_Arrays_2
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for alloc_arrays_2
            
            Destructor for class Alloc_Arrays_2
            Defined at datatypes.fpp lines 65-69
            
            Parameters
            ----------
            this : Alloc_Arrays_2
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2_finalise(this=self._handle)
        
        @property
        def chi(self):
            """
            Element chi ftype=real(idp) pytype=float
            Defined at datatypes.fpp line 66
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2__array__chi(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                chi = self._arrays[array_hash]
            else:
                try:
                    chi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2__array__chi)
                except TypeError:
                    chi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = chi
            return chi
        
        @chi.setter
        def chi(self, chi):
            self.chi[...] = chi
        
        @property
        def psi(self):
            """
            Element psi ftype=real(idp) pytype=float
            Defined at datatypes.fpp line 67
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2__array__psi(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                psi = self._arrays[array_hash]
            else:
                try:
                    psi = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2__array__psi)
                except TypeError:
                    psi = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = psi
            return psi
        
        @psi.setter
        def psi(self, psi):
            self.psi[...] = psi
        
        @property
        def chi_shape(self):
            """
            Element chi_shape ftype=integer(4) pytype=int
            Defined at datatypes.fpp line 68
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2__array__chi_shape(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                chi_shape = self._arrays[array_hash]
            else:
                try:
                    chi_shape = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2__array__chi_shape)
                except TypeError:
                    chi_shape = f90wrap.runtime.direct_c_array(array_type, array_shape, \
                        array_handle)
                self._arrays[array_handle] = chi_shape
            return chi_shape
        
        @chi_shape.setter
        def chi_shape(self, chi_shape):
            self.chi_shape[...] = chi_shape
        
        @property
        def psi_shape(self):
            """
            Element psi_shape ftype=integer(4) pytype=int
            Defined at datatypes.fpp line 69
            """
            array_ndim, array_type, array_shape, array_handle = \
                _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2__array__psi_shape(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                psi_shape = self._arrays[array_hash]
            else:
                try:
                    psi_shape = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _ExampleDerivedTypes.f90wrap_datatypes__alloc_arrays_2__array__psi_shape)
                except TypeError:
                    psi_shape = f90wrap.runtime.direct_c_array(array_type, array_shape, \
                        array_handle)
                self._arrays[array_handle] = psi_shape
            return psi_shape
        
        @psi_shape.setter
        def psi_shape(self, psi_shape):
            self.psi_shape[...] = psi_shape
        
        def __str__(self):
            ret = ['<alloc_arrays_2>{\n']
            ret.append('    chi : ')
            ret.append(repr(self.chi))
            ret.append(',\n    psi : ')
            ret.append(repr(self.psi))
            ret.append(',\n    chi_shape : ')
            ret.append(repr(self.chi_shape))
            ret.append(',\n    psi_shape : ')
            ret.append(repr(self.psi_shape))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("ExampleDerivedTypes.array_nested")
    class array_nested(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=array_nested)
        Defined at datatypes.fpp lines 71-74
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for array_nested
            
            self = Array_Nested()
            Defined at datatypes.fpp lines 71-74
            
            Returns
            -------
            this : Array_Nested
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _ExampleDerivedTypes.f90wrap_datatypes__array_nested_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for array_nested
            
            Destructor for class Array_Nested
            Defined at datatypes.fpp lines 71-74
            
            Parameters
            ----------
            this : Array_Nested
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _ExampleDerivedTypes.f90wrap_datatypes__array_nested_finalise(this=self._handle)
        
        def init_array_xi(self):
            self.xi = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_getitem__xi,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_setitem__xi,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_len__xi,
                                                """
            Element xi ftype=type(different_types) pytype=Different_Types
            Defined at datatypes.fpp line 72
            """, Datatypes.different_types)
            return self.xi
        
        def init_array_omicron(self):
            self.omicron = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_getitem__omicron,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_setitem__omicron,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_len__omicron,
                                                """
            Element omicron ftype=type(fixed_shape_arrays) pytype=Fixed_Shape_Arrays
            Defined at datatypes.fpp line 73
            """, Datatypes.fixed_shape_arrays)
            return self.omicron
        
        def init_array_pi(self):
            self.pi = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_getitem__pi,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_setitem__pi,
                                                _ExampleDerivedTypes.f90wrap_datatypes__array_nested__array_len__pi,
                                                """
            Element pi ftype=type(alloc_arrays) pytype=Alloc_Arrays
            Defined at datatypes.fpp line 74
            """, Datatypes_Allocatable.alloc_arrays)
            return self.pi
        
        _dt_array_initialisers = [init_array_xi, init_array_omicron, init_array_pi]
        
    
    @staticmethod
    def init_array_nested(self, size_bn, interface_call=False):
        """
        init_array_nested(self, size_bn)
        Defined at datatypes.fpp lines 77-82
        
        Parameters
        ----------
        dertype : Array_Nested
        size_bn : int32
        """
        _ExampleDerivedTypes.f90wrap_datatypes__init_array_nested(dertype=self._handle, \
            size_bn=size_bn)
    
    @staticmethod
    def destroy_array_nested(self, interface_call=False):
        """
        destroy_array_nested(self)
        Defined at datatypes.fpp lines 84-88
        
        Parameters
        ----------
        dertype : Array_Nested
        """
        _ExampleDerivedTypes.f90wrap_datatypes__destroy_array_nested(dertype=self._handle)
    
    _dt_array_initialisers = []
    
    

datatypes = Datatypes()

class Library(f90wrap.runtime.FortranModule):
    """
    Module library
    Defined at library.fpp lines 5-154
    """
    @staticmethod
    def return_value_func(val_in, interface_call=False):
        """
        val_out = return_value_func(val_in)
        Defined at library.fpp lines 9-11
        
        Parameters
        ----------
        val_in : int32
        
        Returns
        -------
        val_out : int32
        """
        val_out = _ExampleDerivedTypes.f90wrap_library__return_value_func(val_in=val_in)
        return val_out
    
    @staticmethod
    def return_value_sub(val_in, interface_call=False):
        """
        val_out = return_value_sub(val_in)
        Defined at library.fpp lines 13-16
        
        Parameters
        ----------
        val_in : int32
        
        Returns
        -------
        val_out : int32
        """
        val_out = _ExampleDerivedTypes.f90wrap_library__return_value_sub(val_in=val_in)
        return val_out
    
    @staticmethod
    def return_a_dt_func(interface_call=False):
        """
        dt = return_a_dt_func()
        Defined at library.fpp lines 18-23
        
        Returns
        -------
        dt : Different_Types
        """
        dt = _ExampleDerivedTypes.f90wrap_library__return_a_dt_func()
        dt = \
            f90wrap.runtime.lookup_class("ExampleDerivedTypes.different_types").from_handle(dt, \
            alloc=True)
        return dt
    
    @staticmethod
    def do_array_stuff(n, x, y, br, co, interface_call=False):
        """
        do_array_stuff(n, x, y, br, co)
        Defined at library.fpp lines 26-38
        
        Parameters
        ----------
        n : int32
        x : float array
        y : float array
        br : float array
        co : float array
        """
        _ExampleDerivedTypes.f90wrap_library__do_array_stuff(n=n, x=x, y=y, br=br, \
            co=co)
    
    @staticmethod
    def only_manipulate(n, array, interface_call=False):
        """
        only_manipulate(n, array)
        Defined at library.fpp lines 40-48
        
        Parameters
        ----------
        n : int32
        array : float array
        """
        _ExampleDerivedTypes.f90wrap_library__only_manipulate(n=n, array=array)
    
    @staticmethod
    def set_derived_type(dt_beta, dt_delta, interface_call=False):
        """
        dt = set_derived_type(dt_beta, dt_delta)
        Defined at library.fpp lines 50-56
        
        Parameters
        ----------
        dt_beta : int32
        dt_delta : float64
        
        Returns
        -------
        dt : Different_Types
        """
        dt = _ExampleDerivedTypes.f90wrap_library__set_derived_type(dt_beta=dt_beta, \
            dt_delta=dt_delta)
        dt = \
            f90wrap.runtime.lookup_class("ExampleDerivedTypes.different_types").from_handle(dt, \
            alloc=True)
        return dt
    
    @staticmethod
    def modify_derived_types(self, dt2, dt3, interface_call=False):
        """
        modify_derived_types(self, dt2, dt3)
        Defined at library.fpp lines 58-66
        
        Parameters
        ----------
        dt1 : Different_Types
        dt2 : Different_Types
        dt3 : Different_Types
        """
        _ExampleDerivedTypes.f90wrap_library__modify_derived_types(dt1=self._handle, \
            dt2=dt2._handle, dt3=dt3._handle)
    
    @staticmethod
    def modify_dertype_fixed_shape_arrays(interface_call=False):
        """
        dertype = modify_dertype_fixed_shape_arrays()
        Defined at library.fpp lines 68-74
        
        Returns
        -------
        dertype : Fixed_Shape_Arrays
        """
        dertype = \
            _ExampleDerivedTypes.f90wrap_library__modify_dertype_fixed_shape_arrays()
        dertype = \
            f90wrap.runtime.lookup_class("ExampleDerivedTypes.fixed_shape_arrays").from_handle(dertype, \
            alloc=True)
        return dertype
    
    @staticmethod
    def return_dertype_pointer_arrays(m, n, interface_call=False):
        """
        dertype = return_dertype_pointer_arrays(m, n)
        Defined at library.fpp lines 76-83
        
        Parameters
        ----------
        m : int32
        n : int32
        
        Returns
        -------
        dertype : Pointer_Arrays
        """
        dertype = \
            _ExampleDerivedTypes.f90wrap_library__return_dertype_pointer_arrays(m=m, \
            n=n)
        dertype = \
            f90wrap.runtime.lookup_class("ExampleDerivedTypes.pointer_arrays").from_handle(dertype, \
            alloc=True)
        return dertype
    
    @staticmethod
    def modify_dertype_pointer_arrays(self, interface_call=False):
        """
        modify_dertype_pointer_arrays(self)
        Defined at library.fpp lines 85-92
        
        Parameters
        ----------
        dertype : Pointer_Arrays
        """
        _ExampleDerivedTypes.f90wrap_library__modify_dertype_pointer_arrays(dertype=self._handle)
    
    @staticmethod
    def return_dertype_alloc_arrays(m, n, interface_call=False):
        """
        dertype = return_dertype_alloc_arrays(m, n)
        Defined at library.fpp lines 94-101
        
        Parameters
        ----------
        m : int32
        n : int32
        
        Returns
        -------
        dertype : Alloc_Arrays
        """
        dertype = _ExampleDerivedTypes.f90wrap_library__return_dertype_alloc_arrays(m=m, \
            n=n)
        dertype = \
            f90wrap.runtime.lookup_class("ExampleDerivedTypes.alloc_arrays").from_handle(dertype, \
            alloc=True)
        return dertype
    
    @staticmethod
    def modify_dertype_alloc_arrays(self, interface_call=False):
        """
        modify_dertype_alloc_arrays(self)
        Defined at library.fpp lines 103-110
        
        Parameters
        ----------
        dertype : Alloc_Arrays
        """
        _ExampleDerivedTypes.f90wrap_library__modify_dertype_alloc_arrays(dertype=self._handle)
    
    _dt_array_initialisers = []
    

library = Library()

class Parameters(f90wrap.runtime.FortranModule):
    """
    Module parameters
    Defined at parameters.fpp lines 5-11
    """
    @property
    def idp(self):
        """
        Element idp ftype=integer pytype=int
        Defined at parameters.fpp line 9
        """
        return _ExampleDerivedTypes.f90wrap_parameters__get__idp()
    
    def get_idp(self):
        return self.idp
    
    @property
    def isp(self):
        """
        Element isp ftype=integer pytype=int
        Defined at parameters.fpp line 10
        """
        return _ExampleDerivedTypes.f90wrap_parameters__get__isp()
    
    def get_isp(self):
        return self.isp
    
    def __str__(self):
        ret = ['<parameters>{\n']
        ret.append('    idp : ')
        ret.append(repr(self.idp))
        ret.append(',\n    isp : ')
        ret.append(repr(self.isp))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

parameters = Parameters()

