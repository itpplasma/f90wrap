from __future__ import print_function, absolute_import, division
import _Example
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_Example = _SafeDirectCExecutor(_Example, module_import_name='_Example')

class Mcyldnad(f90wrap.runtime.FortranModule):
    """
    Module mcyldnad
    Defined at cyldnad.fpp lines 5-14
    """
    @staticmethod
    def cyldnad(self, height, interface_call=False):
        """
        vol = cyldnad(self, height)
        Defined at cyldnad.fpp lines 8-13
        
        Parameters
        ----------
        radius : Dual_Num
        height : Dual_Num
        
        Returns
        -------
        vol : Dual_Num
        """
        vol = _Example.f90wrap_mcyldnad__cyldnad(radius=self._handle, \
            height=height._handle)
        vol = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(vol, \
            alloc=True)
        return vol
    
    _dt_array_initialisers = []
    

mcyldnad = Mcyldnad()

class Dual_Num_Auto_Diff(f90wrap.runtime.FortranModule):
    """
    Module dual_num_auto_diff
    Defined at DNAD.fpp lines 112-1609
    """
    @f90wrap.runtime.register_class("Example.DUAL_NUM")
    class DUAL_NUM(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=dual_num)
        Defined at DNAD.fpp lines 119-124
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for dual_num
            
            self = Dual_Num()
            Defined at DNAD.fpp lines 119-124
            
            Returns
            -------
            this : Dual_Num
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _Example.f90wrap_dual_num_auto_diff__dual_num_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for dual_num
            
            Destructor for class Dual_Num
            Defined at DNAD.fpp lines 119-124
            
            Parameters
            ----------
            this : Dual_Num
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _Example.f90wrap_dual_num_auto_diff__dual_num_finalise(this=self._handle)
        
        @property
        def x_ad_(self):
            """
            Element x_ad_ ftype=real(dbl_ad) pytype=float
            Defined at DNAD.fpp line 123
            """
            return _Example.f90wrap_dual_num_auto_diff__dual_num__get__x_ad_(self._handle)
        
        @x_ad_.setter
        def x_ad_(self, x_ad_):
            _Example.f90wrap_dual_num_auto_diff__dual_num__set__x_ad_(self._handle, x_ad_)
        
        @property
        def xp_ad_(self):
            """
            Element xp_ad_ ftype=real(dbl_ad) pytype=float
            Defined at DNAD.fpp line 124
            """
            array_ndim, array_type, array_shape, array_handle = \
                _Example.f90wrap_dual_num_auto_diff__dual_num__array__xp_ad_(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                xp_ad_ = self._arrays[array_hash]
            else:
                try:
                    xp_ad_ = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _Example.f90wrap_dual_num_auto_diff__dual_num__array__xp_ad_)
                except TypeError:
                    xp_ad_ = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = xp_ad_
            return xp_ad_
        
        @xp_ad_.setter
        def xp_ad_(self, xp_ad_):
            self.xp_ad_[...] = xp_ad_
        
        def __str__(self):
            ret = ['<dual_num>{\n']
            ret.append('    x_ad_ : ')
            ret.append(repr(self.x_ad_))
            ret.append(',\n    xp_ad_ : ')
            ret.append(repr(self.xp_ad_))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def abs_d(self, interface_call=False):
        """
        res = abs_d(self)
        Defined at DNAD.fpp lines 1223-1235
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__abs_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def acos_d(self, interface_call=False):
        """
        res = acos_d(self)
        Defined at DNAD.fpp lines 1241-1251
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__acos_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def asin_d(self, interface_call=False):
        """
        res = asin_d(self)
        Defined at DNAD.fpp lines 1257-1267
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__asin_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def cos_d(self, interface_call=False):
        """
        res = cos_d(self)
        Defined at DNAD.fpp lines 1273-1279
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__cos_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def exp_d(self, interface_call=False):
        """
        res = exp_d(self)
        Defined at DNAD.fpp lines 1298-1304
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__exp_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def int_d(self, interface_call=False):
        """
        res = int_d(self)
        Defined at DNAD.fpp lines 1310-1315
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : int32
        """
        res = _Example.f90wrap_dual_num_auto_diff__int_d(u=self._handle)
        return res
    
    @staticmethod
    def log_d(self, interface_call=False):
        """
        res = log_d(self)
        Defined at DNAD.fpp lines 1323-1329
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__log_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def log10_d(self, interface_call=False):
        """
        res = log10_d(self)
        Defined at DNAD.fpp lines 1337-1343
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__log10_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def nint_d(self, interface_call=False):
        """
        res = nint_d(self)
        Defined at DNAD.fpp lines 1533-1536
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : int32
        """
        res = _Example.f90wrap_dual_num_auto_diff__nint_d(u=self._handle)
        return res
    
    @staticmethod
    def sin_d(self, interface_call=False):
        """
        res = sin_d(self)
        Defined at DNAD.fpp lines 1569-1575
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__sin_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def sqrt_d(self, interface_call=False):
        """
        res = sqrt_d(self)
        Defined at DNAD.fpp lines 1581-1592
        
        Parameters
        ----------
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__sqrt_d(u=self._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def max_dd(self, val2, val3=None, val4=None, val5=None, interface_call=False):
        """
        res = max_dd(self, val2[, val3, val4, val5])
        Defined at DNAD.fpp lines 1391-1408
        
        Parameters
        ----------
        val1 : Dual_Num
        val2 : Dual_Num
        val3 : Dual_Num
        val4 : Dual_Num
        val5 : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__max_dd(val1=self._handle, \
            val2=val2._handle, val3=None if val3 is None else val3._handle, val4=None if \
            val4 is None else val4._handle, val5=None if val5 is None else val5._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def max_di(self, n, interface_call=False):
        """
        res = max_di(self, n)
        Defined at DNAD.fpp lines 1413-1421
        
        Parameters
        ----------
        u : Dual_Num
        n : int32
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__max_di(u=self._handle, n=n)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def max_dr(self, n, interface_call=False):
        """
        res = max_dr(self, n)
        Defined at DNAD.fpp lines 1426-1434
        
        Parameters
        ----------
        u : Dual_Num
        n : float64
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__max_dr(u=self._handle, n=n)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def max_ds(self, n, interface_call=False):
        """
        res = max_ds(self, n)
        Defined at DNAD.fpp lines 1439-1447
        
        Parameters
        ----------
        u : Dual_Num
        n : float32
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__max_ds(u=self._handle, n=n)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def max_rd(r, u, interface_call=False):
        """
        res = max_rd(r, u)
        Defined at DNAD.fpp lines 1455-1463
        
        Parameters
        ----------
        r : float64
        u : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__max_rd(r=r, u=u._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def max(*args, **kwargs):
        """
        max(*args, **kwargs)
        Defined at DNAD.fpp lines 334-339
        
        Overloaded interface containing the following procedures:
          max_dd
          max_di
          max_dr
          max_ds
          max_rd
        """
        for proc in [Dual_Num_Auto_Diff.max_dd, Dual_Num_Auto_Diff.max_di, \
            Dual_Num_Auto_Diff.max_dr, Dual_Num_Auto_Diff.max_ds, \
            Dual_Num_Auto_Diff.max_rd]:
            exception=None
            try:
                return proc(*args, **kwargs, interface_call=True)
            except (TypeError, ValueError, AttributeError, IndexError, \
                numpy.exceptions.ComplexWarning) as err:
                exception = "'%s: %s'" % (type(err).__name__, str(err))
                continue
        
        argTypes=[]
        for arg in args:
            try:
                argTypes.append("%s: dims '%s', type '%s',"
                " type code '%s'"
                %(str(type(arg)),arg.ndim, arg.dtype, arg.dtype.num))
            except AttributeError:
                argTypes.append(str(type(arg)))
        raise TypeError("Not able to call a version of "
            "max compatible with the provided args:"
            "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
    
    @staticmethod
    def min_dd(self, val2, val3=None, val4=None, interface_call=False):
        """
        res = min_dd(self, val2[, val3, val4])
        Defined at DNAD.fpp lines 1478-1492
        
        Parameters
        ----------
        val1 : Dual_Num
        val2 : Dual_Num
        val3 : Dual_Num
        val4 : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__min_dd(val1=self._handle, \
            val2=val2._handle, val3=None if val3 is None else val3._handle, val4=None if \
            val4 is None else val4._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def min_dr(self, n, interface_call=False):
        """
        res = min_dr(self, n)
        Defined at DNAD.fpp lines 1497-1505
        
        Parameters
        ----------
        u : Dual_Num
        n : float64
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__min_dr(u=self._handle, n=n)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def min_ds(self, n, interface_call=False):
        """
        res = min_ds(self, n)
        Defined at DNAD.fpp lines 1510-1518
        
        Parameters
        ----------
        u : Dual_Num
        n : float32
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__min_ds(u=self._handle, n=n)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def min(*args, **kwargs):
        """
        min(*args, **kwargs)
        Defined at DNAD.fpp lines 350-353
        
        Overloaded interface containing the following procedures:
          min_dd
          min_dr
          min_ds
        """
        for proc in [Dual_Num_Auto_Diff.min_dd, Dual_Num_Auto_Diff.min_dr, \
            Dual_Num_Auto_Diff.min_ds]:
            exception=None
            try:
                return proc(*args, **kwargs, interface_call=True)
            except (TypeError, ValueError, AttributeError, IndexError, \
                numpy.exceptions.ComplexWarning) as err:
                exception = "'%s: %s'" % (type(err).__name__, str(err))
                continue
        
        argTypes=[]
        for arg in args:
            try:
                argTypes.append("%s: dims '%s', type '%s',"
                " type code '%s'"
                %(str(type(arg)),arg.ndim, arg.dtype, arg.dtype.num))
            except AttributeError:
                argTypes.append(str(type(arg)))
        raise TypeError("Not able to call a version of "
            "min compatible with the provided args:"
            "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
    
    @staticmethod
    def sign_dd(self, val2, interface_call=False):
        """
        res = sign_dd(self, val2)
        Defined at DNAD.fpp lines 1542-1549
        
        Parameters
        ----------
        val1 : Dual_Num
        val2 : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__sign_dd(val1=self._handle, \
            val2=val2._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def sign_rd(val1, val2, interface_call=False):
        """
        res = sign_rd(val1, val2)
        Defined at DNAD.fpp lines 1555-1563
        
        Parameters
        ----------
        val1 : float64
        val2 : Dual_Num
        
        Returns
        -------
        res : Dual_Num
        """
        res = _Example.f90wrap_dual_num_auto_diff__sign_rd(val1=val1, val2=val2._handle)
        res = f90wrap.runtime.lookup_class("Example.DUAL_NUM").from_handle(res, \
            alloc=True)
        return res
    
    @staticmethod
    def sign(*args, **kwargs):
        """
        sign(*args, **kwargs)
        Defined at DNAD.fpp lines 368-370
        
        Overloaded interface containing the following procedures:
          sign_dd
          sign_rd
        """
        for proc in [Dual_Num_Auto_Diff.sign_dd, Dual_Num_Auto_Diff.sign_rd]:
            exception=None
            try:
                return proc(*args, **kwargs, interface_call=True)
            except (TypeError, ValueError, AttributeError, IndexError, \
                numpy.exceptions.ComplexWarning) as err:
                exception = "'%s: %s'" % (type(err).__name__, str(err))
                continue
        
        argTypes=[]
        for arg in args:
            try:
                argTypes.append("%s: dims '%s', type '%s',"
                " type code '%s'"
                %(str(type(arg)),arg.ndim, arg.dtype, arg.dtype.num))
            except AttributeError:
                argTypes.append(str(type(arg)))
        raise TypeError("Not able to call a version of "
            "sign compatible with the provided args:"
            "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
    
    @property
    def ndv_ad(self):
        """
        Element ndv_ad ftype=integer(2) pytype=int
        Defined at DNAD.fpp line 114
        """
        return _Example.f90wrap_dual_num_auto_diff__get__ndv_ad()
    
    def get_ndv_ad(self):
        return self.ndv_ad
    
    def __str__(self):
        ret = ['<dual_num_auto_diff>{\n']
        ret.append('    ndv_ad : ')
        ret.append(repr(self.ndv_ad))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    
    

dual_num_auto_diff = Dual_Num_Auto_Diff()

