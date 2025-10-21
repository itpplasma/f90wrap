from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

class M_Base_Poly(f90wrap.runtime.FortranModule):
    """
    Module m_base_poly
    Defined at base_poly.fpp lines 5-17
    """
    @f90wrap.runtime.register_class("pywrapper.Polygone")
    class Polygone(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=polygone)
        Defined at base_poly.fpp lines 8-10
        """
        def __init__(self):
            raise(NotImplementedError("This is an abstract class"))
        
        def is_polygone(self, interface_call=False):
            """
            is_polygone = is_polygone(self)
            Defined at base_poly.fpp lines 13-16
            
            Parameters
            ----------
            this : Polygone
            
            Returns
            -------
            is_polygone : int32
            """
            if not isinstance(self, m_base_poly.Polygone) :
                msg = f"Expecting '{m_base_poly.Polygone}' but got '{type(self)}'"
                raise TypeError(msg)
            is_polygone = \
                _pywrapper.f90wrap_m_base_poly__is_polygone__binding__polygone(this=self._handle)
            return is_polygone
        
        _dt_array_initialisers = []
        
    
    _dt_array_initialisers = []
    
    if not hasattr(_pywrapper, \
        "f90wrap_m_base_poly__is_polygone__binding__polygone"):
        for _candidate in ["f90wrap_m_base_poly__is_polygone__binding__polygone"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_base_poly__is_polygone__binding__polygone", \
                    getattr(_pywrapper, _candidate))
                break
    
    @staticmethod
    def is_polygone(instance, *args, **kwargs):
        return instance.is_polygone(*args, **kwargs)
    

m_base_poly = M_Base_Poly()

class M_Geometry(f90wrap.runtime.FortranModule):
    """
    Module m_geometry
    Defined at main-oo.fpp lines 5-291
    """
    @f90wrap.runtime.register_class("pywrapper.Rectangle")
    class Rectangle(m_base_poly.Polygone):
        """
        Type(name=rectangle)
        Defined at main-oo.fpp lines 11-17
        """
        def __init__(self):
            raise(NotImplementedError("This is an abstract class"))
        
        def perimeter(self, interface_call=False):
            """
            perimeter = perimeter(self)
            Defined at main-oo.fpp lines 229-232
            
            Parameters
            ----------
            this : Rectangle
            
            Returns
            -------
            perimeter : float64
            """
            if not isinstance(self, m_geometry.Rectangle) :
                msg = f"Expecting '{m_geometry.Rectangle}' but got '{type(self)}'"
                raise TypeError(msg)
            perimeter = \
                _pywrapper.f90wrap_m_geometry__perimeter__binding__rectangle(this=self._handle)
            return perimeter
        
        def is_square(self, interface_call=False):
            """
            is_square = is_square(self)
            Defined at main-oo.fpp lines 239-242
            
            Parameters
            ----------
            this : Rectangle
            
            Returns
            -------
            is_square : int32
            """
            if not isinstance(self, m_geometry.Rectangle) :
                msg = f"Expecting '{m_geometry.Rectangle}' but got '{type(self)}'"
                raise TypeError(msg)
            is_square = \
                _pywrapper.f90wrap_m_geometry__is_square__binding__rectangle(this=self._handle)
            return is_square
        
        def area(self, interface_call=False):
            """
            area = area(self)
            Defined at main-oo.fpp lines 39-42
            
            Parameters
            ----------
            this : Rectangle
            
            Returns
            -------
            area : float64
            """
            if not isinstance(self, m_geometry.Rectangle) :
                msg = f"Expecting '{m_geometry.Rectangle}' but got '{type(self)}'"
                raise TypeError(msg)
            area = \
                _pywrapper.f90wrap_m_geometry__area__binding__rectangle(this=self._handle)
            return area
        
        def is_polygone(self, interface_call=False):
            """
            is_polygone = is_polygone(self)
            Defined at base_poly.fpp lines 13-16
            
            Parameters
            ----------
            this : Rectangle
            
            Returns
            -------
            is_polygone : int32
            """
            if not isinstance(self, m_geometry.Rectangle) :
                msg = f"Expecting '{m_geometry.Rectangle}' but got '{type(self)}'"
                raise TypeError(msg)
            is_polygone = \
                _pywrapper.f90wrap_m_base_poly__is_polygone__binding__polygone_rectangle(this=self._handle)
            return is_polygone
        
        @property
        def length(self):
            """
            Element length ftype=real(kind=8) pytype=float
            Defined at main-oo.fpp line 12
            """
            return _pywrapper.f90wrap_m_geometry__rectangle__get__length(self._handle)
        
        @length.setter
        def length(self, length):
            _pywrapper.f90wrap_m_geometry__rectangle__set__length(self._handle, length)
        
        @property
        def width(self):
            """
            Element width ftype=real(kind=8) pytype=float
            Defined at main-oo.fpp line 13
            """
            return _pywrapper.f90wrap_m_geometry__rectangle__get__width(self._handle)
        
        @width.setter
        def width(self, width):
            _pywrapper.f90wrap_m_geometry__rectangle__set__width(self._handle, width)
        
        def __str__(self):
            ret = ['<rectangle>{\n']
            ret.append('    length : ')
            ret.append(repr(self.length))
            ret.append(',\n    width : ')
            ret.append(repr(self.width))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.Square")
    class Square(Rectangle):
        """
        Type(name=square)
        Defined at main-oo.fpp lines 19-27
        """
        def __init__(self, length, handle=None):
            """
            construct_square = Square(length)
            Defined at main-oo.fpp lines 127-131
            
            Parameters
            ----------
            length : float32
            
            Returns
            -------
            construct_square : Square
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_geometry__construct_square(length=length)
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for square
            
            Destructor for class Square
            Defined at main-oo.fpp lines 19-27
            
            Parameters
            ----------
            this : Square
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_geometry__square_finalise(this=self._handle)
        
        def init(self, length, interface_call=False):
            """
            init(self, length)
            Defined at main-oo.fpp lines 250-254
            
            Parameters
            ----------
            this : Square
            length : float32
            """
            if not isinstance(self, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(length,(numpy.ndarray, numpy.generic)):
                if not interface_call and length.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                    12, 13}:
                    length = length.astype('float32')
                if length.ndim != 0 or length.dtype.num != 11:
                    raise TypeError("Expecting 'float' (code '11')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(length.dtype, length.dtype.num, length.ndim))
            elif not isinstance(length,float):
                raise TypeError("Expecting 'float' but got '%s'"%type(length))
            _pywrapper.f90wrap_m_geometry__init__binding__square(this=self._handle, \
                length=length)
        
        def is_square(self, interface_call=False):
            """
            is_square = is_square(self)
            Defined at main-oo.fpp lines 256-259
            
            Parameters
            ----------
            this : Square
            
            Returns
            -------
            is_square : int32
            """
            if not isinstance(self, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(self)}'"
                raise TypeError(msg)
            is_square = \
                _pywrapper.f90wrap_m_geometry__is_square__binding__square(this=self._handle)
            return is_square
        
        def area(self, interface_call=False):
            """
            area = area(self)
            Defined at main-oo.fpp lines 234-237
            
            Parameters
            ----------
            this : Square
            
            Returns
            -------
            area : float64
            """
            if not isinstance(self, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(self)}'"
                raise TypeError(msg)
            area = _pywrapper.f90wrap_m_geometry__area__binding__square(this=self._handle)
            return area
        
        def is_equal(self, other, interface_call=False):
            """
            is_equal = is_equal(self, other)
            Defined at main-oo.fpp lines 261-270
            
            Parameters
            ----------
            this : Square
            other : Rectangle
            
            Returns
            -------
            is_equal : int32
            """
            if not isinstance(self, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(self)}'"
                raise TypeError(msg)
            if not isinstance(other, m_geometry.Rectangle) :
                msg = f"Expecting '{m_geometry.Rectangle}' but got '{type(other)}'"
                raise TypeError(msg)
            is_equal = \
                _pywrapper.f90wrap_m_geometry__is_equal__binding__square(this=self._handle, \
                other=other._handle)
            return is_equal
        
        def copy(self, from_, interface_call=False):
            """
            copy(self, from_)
            Defined at main-oo.fpp lines 244-248
            
            Parameters
            ----------
            this : Square
            from_ : Square
            """
            if not isinstance(self, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(self)}'"
                raise TypeError(msg)
            if not isinstance(from_, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(from_)}'"
                raise TypeError(msg)
            _pywrapper.f90wrap_m_geometry__copy__binding__square(this=self._handle, \
                from_=from_._handle)
        
        def create_diamond(self, interface_call=False):
            """
            square_create_diamond = create_diamond(self)
            Defined at main-oo.fpp lines 288-291
            
            Parameters
            ----------
            this : Square
            
            Returns
            -------
            square_create_diamond : Diamond
            """
            if not isinstance(self, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(self)}'"
                raise TypeError(msg)
            square_create_diamond = \
                _pywrapper.f90wrap_m_geometry__create_diamond__binding__square(this=self._handle)
            square_create_diamond = \
                f90wrap.runtime.lookup_class("pywrapper.Diamond").from_handle(square_create_diamond, \
                alloc=True)
            return square_create_diamond
        
        def assignment(self, *args, **kwargs):
            """
            Binding(name=assignment(=))
            Defined at main-oo.fpp line 27
            """
            for proc in [self.copy]:
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
                "assignment compatible with the provided args:"
                "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
        
        def perimeter(self, interface_call=False):
            """
            perimeter = perimeter(self)
            Defined at main-oo.fpp lines 229-232
            
            Parameters
            ----------
            this : Square
            
            Returns
            -------
            perimeter : float64
            """
            if not isinstance(self, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(self)}'"
                raise TypeError(msg)
            perimeter = \
                _pywrapper.f90wrap_m_geometry__perimeter__binding__rectangle_square(this=self._handle)
            return perimeter
        
        def is_polygone(self, interface_call=False):
            """
            is_polygone = is_polygone(self)
            Defined at base_poly.fpp lines 13-16
            
            Parameters
            ----------
            this : Square
            
            Returns
            -------
            is_polygone : int32
            """
            if not isinstance(self, m_geometry.Square) :
                msg = f"Expecting '{m_geometry.Square}' but got '{type(self)}'"
                raise TypeError(msg)
            is_polygone = \
                _pywrapper.f90wrap_m_base_poly__is_polygone__binding__polygone_rectang5400(this=self._handle)
            return is_polygone
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.Diamond")
    class Diamond(m_base_poly.Polygone):
        """
        Type(name=diamond)
        Defined at main-oo.fpp lines 29-36
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for diamond
            
            self = Diamond()
            Defined at main-oo.fpp lines 29-36
            
            Returns
            -------
            this : Diamond
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_geometry__diamond_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for diamond
            
            Destructor for class Diamond
            Defined at main-oo.fpp lines 29-36
            
            Parameters
            ----------
            this : Diamond
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_geometry__diamond_finalise(this=self._handle)
        
        def init(self, width, length, interface_call=False):
            """
            init(self, width, length)
            Defined at main-oo.fpp lines 272-276
            
            Parameters
            ----------
            this : Diamond
            width : float64
            length : float64
            """
            if not isinstance(self, m_geometry.Diamond) :
                msg = f"Expecting '{m_geometry.Diamond}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(width,(numpy.ndarray, numpy.generic)):
                if not interface_call and width.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                    12, 13}:
                    width = width.astype('float64')
                if width.ndim != 0 or width.dtype.num != 12:
                    raise TypeError("Expecting 'float' (code '12')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(width.dtype, width.dtype.num, width.ndim))
            elif not isinstance(width,float):
                raise TypeError("Expecting 'float' but got '%s'"%type(width))
            if isinstance(length,(numpy.ndarray, numpy.generic)):
                if not interface_call and length.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                    12, 13}:
                    length = length.astype('float64')
                if length.ndim != 0 or length.dtype.num != 12:
                    raise TypeError("Expecting 'float' (code '12')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(length.dtype, length.dtype.num, length.ndim))
            elif not isinstance(length,float):
                raise TypeError("Expecting 'float' but got '%s'"%type(length))
            _pywrapper.f90wrap_m_geometry__init__binding__diamond(this=self._handle, \
                width=width, length=length)
        
        def info(self, interface_call=False):
            """
            info(self)
            Defined at main-oo.fpp lines 283-286
            
            Parameters
            ----------
            this : Diamond
            """
            if not isinstance(self, m_geometry.Diamond) :
                msg = f"Expecting '{m_geometry.Diamond}' but got '{type(self)}'"
                raise TypeError(msg)
            _pywrapper.f90wrap_m_geometry__info__binding__diamond(this=self._handle)
        
        def copy(self, other, interface_call=False):
            """
            copy(self, other)
            Defined at main-oo.fpp lines 278-281
            
            Parameters
            ----------
            this : Diamond
            other : Diamond
            """
            if not isinstance(self, m_geometry.Diamond) :
                msg = f"Expecting '{m_geometry.Diamond}' but got '{type(self)}'"
                raise TypeError(msg)
            if not isinstance(other, m_geometry.Diamond) :
                msg = f"Expecting '{m_geometry.Diamond}' but got '{type(other)}'"
                raise TypeError(msg)
            _pywrapper.f90wrap_m_geometry__copy__binding__diamond(this=self._handle, \
                other=other._handle)
        
        def assignment(self, *args, **kwargs):
            """
            Binding(name=assignment(=))
            Defined at main-oo.fpp line 36
            """
            for proc in [self.copy]:
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
                "assignment compatible with the provided args:"
                "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
        
        def is_polygone(self, interface_call=False):
            """
            is_polygone = is_polygone(self)
            Defined at base_poly.fpp lines 13-16
            
            Parameters
            ----------
            this : Diamond
            
            Returns
            -------
            is_polygone : int32
            """
            if not isinstance(self, m_geometry.Diamond) :
                msg = f"Expecting '{m_geometry.Diamond}' but got '{type(self)}'"
                raise TypeError(msg)
            is_polygone = \
                _pywrapper.f90wrap_m_base_poly__is_polygone__binding__polygone_diamond(this=self._handle)
            return is_polygone
        
        @property
        def length(self):
            """
            Element length ftype=real(kind=8) pytype=float
            Defined at main-oo.fpp line 30
            """
            return _pywrapper.f90wrap_m_geometry__diamond__get__length(self._handle)
        
        @length.setter
        def length(self, length):
            _pywrapper.f90wrap_m_geometry__diamond__set__length(self._handle, length)
        
        @property
        def width(self):
            """
            Element width ftype=real(kind=8) pytype=float
            Defined at main-oo.fpp line 31
            """
            return _pywrapper.f90wrap_m_geometry__diamond__get__width(self._handle)
        
        @width.setter
        def width(self, width):
            _pywrapper.f90wrap_m_geometry__diamond__set__width(self._handle, width)
        
        def __str__(self):
            ret = ['<diamond>{\n']
            ret.append('    length : ')
            ret.append(repr(self.length))
            ret.append(',\n    width : ')
            ret.append(repr(self.width))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.List_square")
    class List_square(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=list_square)
        Defined at main-oo.fpp lines 48-57
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for list_square
            
            self = List_Square()
            Defined at main-oo.fpp lines 48-57
            
            Returns
            -------
            this : List_Square
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_geometry__list_square_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for list_square
            
            Destructor for class List_Square
            Defined at main-oo.fpp lines 48-57
            
            Parameters
            ----------
            this : List_Square
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_geometry__list_square_finalise(this=self._handle)
        
        def init(self, n, interface_call=False):
            """
            init(self, n)
            Defined at main-oo.fpp lines 133-141
            
            Parameters
            ----------
            this : List_Square
            n : int32
            """
            if not isinstance(self, m_geometry.List_square) :
                msg = f"Expecting '{m_geometry.List_square}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(n,(numpy.ndarray, numpy.generic)):
                if not interface_call and n.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, 12, \
                    13}:
                    n = n.astype('int32')
                if n.ndim != 0 or n.dtype.num != 5:
                    raise TypeError("Expecting 'int' (code '5')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(n.dtype, n.dtype.num, n.ndim))
            elif not isinstance(n,int):
                raise TypeError("Expecting 'int' but got '%s'"%type(n))
            _pywrapper.f90wrap_m_geometry__init__binding__list_square(this=self._handle, \
                n=n)
        
        def init_array_alloc_type(self):
            self.alloc_type = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_getitem__alloc_type,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_setitem__alloc_type,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_len__alloc_type,
                                                """
            Element alloc_type ftype=type(square) pytype=Square
            Defined at main-oo.fpp line 49
            """, M_Geometry.Square)
            return self.alloc_type
        
        def init_array_ptr_type(self):
            self.ptr_type = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_getitem__ptr_type,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_setitem__ptr_type,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_len__ptr_type,
                                                """
            Element ptr_type ftype=type(square) pytype=Square
            Defined at main-oo.fpp line 50
            """, M_Geometry.Square)
            return self.ptr_type
        
        def init_array_alloc_class(self):
            self.alloc_class = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_getitem__alloc_class,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_setitem__alloc_class,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_len__alloc_class,
                                                """
            Element alloc_class ftype=class(square) pytype=Square
            Defined at main-oo.fpp line 51
            """, M_Geometry.Square)
            return self.alloc_class
        
        def init_array_ptr_class(self):
            self.ptr_class = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_getitem__ptr_class,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_setitem__ptr_class,
                                                _pywrapper.f90wrap_m_geometry__list_square__array_len__ptr_class,
                                                """
            Element ptr_class ftype=class(square) pytype=Square
            Defined at main-oo.fpp line 52
            """, M_Geometry.Square)
            return self.ptr_class
        
        @property
        def scalar_class(self):
            """
            Element scalar_class ftype=class(square) pytype=Square
            Defined at main-oo.fpp line 53
            """
            scalar_class_handle = \
                _pywrapper.f90wrap_m_geometry__list_square__get__scalar_class(self._handle)
            if tuple(scalar_class_handle) in self._objs:
                scalar_class = self._objs[tuple(scalar_class_handle)]
            else:
                scalar_class = m_geometry.Square.from_handle(scalar_class_handle)
                self._objs[tuple(scalar_class_handle)] = scalar_class
            return scalar_class
        
        @scalar_class.setter
        def scalar_class(self, scalar_class):
            scalar_class = scalar_class._handle
            _pywrapper.f90wrap_m_geometry__list_square__set__scalar_class(self._handle, \
                scalar_class)
        
        @property
        def scalar_type(self):
            """
            Element scalar_type ftype=type(square) pytype=Square
            Defined at main-oo.fpp line 54
            """
            scalar_type_handle = \
                _pywrapper.f90wrap_m_geometry__list_square__get__scalar_type(self._handle)
            if tuple(scalar_type_handle) in self._objs:
                scalar_type = self._objs[tuple(scalar_type_handle)]
            else:
                scalar_type = m_geometry.Square.from_handle(scalar_type_handle)
                self._objs[tuple(scalar_type_handle)] = scalar_type
            return scalar_type
        
        @scalar_type.setter
        def scalar_type(self, scalar_type):
            scalar_type = scalar_type._handle
            _pywrapper.f90wrap_m_geometry__list_square__set__scalar_type(self._handle, \
                scalar_type)
        
        @property
        def n(self):
            """
            Element n ftype=integer  pytype=int
            Defined at main-oo.fpp line 55
            """
            return _pywrapper.f90wrap_m_geometry__list_square__get__n(self._handle)
        
        @n.setter
        def n(self, n):
            _pywrapper.f90wrap_m_geometry__list_square__set__n(self._handle, n)
        
        def __str__(self):
            ret = ['<list_square>{\n']
            ret.append('    scalar_class : ')
            ret.append(repr(self.scalar_class))
            ret.append(',\n    scalar_type : ')
            ret.append(repr(self.scalar_type))
            ret.append(',\n    n : ')
            ret.append(repr(self.n))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = [init_array_alloc_type, init_array_ptr_type, \
            init_array_alloc_class, init_array_ptr_class]
        
    
    @f90wrap.runtime.register_class("pywrapper.Circle")
    class Circle(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=circle)
        Defined at main-oo.fpp lines 59-71
        """
        def __init__(self, rc, rb, handle=None):
            """
            construct_circle = Circle(rc, rb)
            Defined at main-oo.fpp lines 153-156
            
            Parameters
            ----------
            rc : float32
            rb : float32
            
            Returns
            -------
            construct_circle : Circle
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_geometry__construct_circle(rc=rc, rb=rb)
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def area(self, interface_call=False):
            """
            area = area(self)
            Defined at main-oo.fpp lines 173-176
            
            Parameters
            ----------
            this : Circle
            
            Returns
            -------
            area : float64
            """
            if not isinstance(self, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
                raise TypeError(msg)
            area = _pywrapper.f90wrap_m_geometry__area__binding__circle(this=self._handle)
            return area
        
        def print(self, interface_call=False):
            """
            print(self)
            Defined at main-oo.fpp lines 178-181
            
            Parameters
            ----------
            this : Circle
            """
            if not isinstance(self, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
                raise TypeError(msg)
            _pywrapper.f90wrap_m_geometry__print__binding__circle(this=self._handle)
        
        def obj_name(self, interface_call=False):
            """
            obj_name(self)
            Defined at main-oo.fpp lines 183-186
            
            Parameters
            ----------
            obj : Circle
            """
            if not isinstance(self, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
                raise TypeError(msg)
            _pywrapper.f90wrap_m_geometry__obj_name__binding__circle(obj=self._handle)
        
        def copy(self, from_, interface_call=False):
            """
            copy(self, from_)
            Defined at main-oo.fpp lines 188-191
            
            Parameters
            ----------
            this : Circle
            from_ : Circle
            """
            if not isinstance(self, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
                raise TypeError(msg)
            if not isinstance(from_, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(from_)}'"
                raise TypeError(msg)
            _pywrapper.f90wrap_m_geometry__copy__binding__circle(this=self._handle, \
                from_=from_._handle)
        
        def init(self, radius, interface_call=False):
            """
            init(self, radius)
            Defined at main-oo.fpp lines 193-196
            
            Parameters
            ----------
            this : Circle
            radius : float32
            """
            if not isinstance(self, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(radius,(numpy.ndarray, numpy.generic)):
                if not interface_call and radius.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                    12, 13}:
                    radius = radius.astype('float32')
                if radius.ndim != 0 or radius.dtype.num != 11:
                    raise TypeError("Expecting 'float' (code '11')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(radius.dtype, radius.dtype.num, radius.ndim))
            elif not isinstance(radius,float):
                raise TypeError("Expecting 'float' but got '%s'"%type(radius))
            _pywrapper.f90wrap_m_geometry__init__binding__circle(this=self._handle, \
                radius=radius)
        
        def private_method(self, interface_call=False):
            """
            private_method(self)
            Defined at main-oo.fpp lines 198-199
            
            Parameters
            ----------
            this : Circle
            """
            if not isinstance(self, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
                raise TypeError(msg)
            _pywrapper.f90wrap_m_geometry__private_method__binding__circle(this=self._handle)
        
        def perimeter_4(self, radius, interface_call=False):
            """
            perimeter = perimeter_4(self, radius)
            Defined at main-oo.fpp lines 217-221
            
            Parameters
            ----------
            this : Circle
            radius : float32
            
            Returns
            -------
            perimeter : float32
            """
            if not isinstance(self, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(radius,(numpy.ndarray, numpy.generic)):
                if not interface_call and radius.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                    12, 13}:
                    radius = radius.astype('float32')
                if radius.ndim != 0 or radius.dtype.num != 11:
                    raise TypeError("Expecting 'float' (code '11')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(radius.dtype, radius.dtype.num, radius.ndim))
            elif not isinstance(radius,float):
                raise TypeError("Expecting 'float' but got '%s'"%type(radius))
            perimeter = \
                _pywrapper.f90wrap_m_geometry__perimeter_4__binding__circle(this=self._handle, \
                radius=radius)
            return perimeter
        
        def perimeter_8(self, radius, interface_call=False):
            """
            perimeter = perimeter_8(self, radius)
            Defined at main-oo.fpp lines 223-227
            
            Parameters
            ----------
            this : Circle
            radius : float64
            
            Returns
            -------
            perimeter : float64
            """
            if not isinstance(self, m_geometry.Circle) :
                msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(radius,(numpy.ndarray, numpy.generic)):
                if not interface_call and radius.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                    12, 13}:
                    radius = radius.astype('float64')
                if radius.ndim != 0 or radius.dtype.num != 12:
                    raise TypeError("Expecting 'float' (code '12')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(radius.dtype, radius.dtype.num, radius.ndim))
            elif not isinstance(radius,float):
                raise TypeError("Expecting 'float' but got '%s'"%type(radius))
            perimeter = \
                _pywrapper.f90wrap_m_geometry__perimeter_8__binding__circle(this=self._handle, \
                radius=radius)
            return perimeter
        
        def perimeter(self, *args, **kwargs):
            """
            Binding(name=perimeter)
            Defined at main-oo.fpp line 70
            """
            for proc in [self.perimeter_8, self.perimeter_4]:
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
                "perimeter compatible with the provided args:"
                "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
        
        def __del__(self):
            """
            Destructor for class Circle
            Defined at main-oo.fpp lines 201-202
            
            Parameters
            ----------
            this : Circle
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_geometry__circle_free__binding__circle(this=self._handle)
        
        @property
        def radius(self):
            """
            Element radius ftype=real(kind=8) pytype=float
            Defined at main-oo.fpp line 60
            """
            return _pywrapper.f90wrap_m_geometry__circle__get__radius(self._handle)
        
        @radius.setter
        def radius(self, radius):
            _pywrapper.f90wrap_m_geometry__circle__set__radius(self._handle, radius)
        
        def __str__(self):
            ret = ['<circle>{\n']
            ret.append('    radius : ')
            ret.append(repr(self.radius))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.Ball")
    class Ball(Circle):
        """
        Type(name=ball)
        Defined at main-oo.fpp lines 73-77
        """
        def __init__(self, rc, rb, handle=None):
            """
            construct_ball = Ball(rc, rb)
            Defined at main-oo.fpp lines 158-161
            
            Parameters
            ----------
            rc : float32
            rb : float32
            
            Returns
            -------
            construct_ball : Ball
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_geometry__construct_ball(rc=rc, rb=rb)
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for ball
            
            Destructor for class Ball
            Defined at main-oo.fpp lines 73-77
            
            Parameters
            ----------
            this : Ball
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_geometry__ball_finalise(this=self._handle)
        
        def volume(self, interface_call=False):
            """
            volume = volume(self)
            Defined at main-oo.fpp lines 209-212
            
            Parameters
            ----------
            this : Ball
            
            Returns
            -------
            volume : float64
            """
            if not isinstance(self, m_geometry.Ball) :
                msg = f"Expecting '{m_geometry.Ball}' but got '{type(self)}'"
                raise TypeError(msg)
            volume = _pywrapper.f90wrap_m_geometry__volume__binding__ball(this=self._handle)
            return volume
        
        def area(self, interface_call=False):
            """
            area = area(self)
            Defined at main-oo.fpp lines 204-207
            
            Parameters
            ----------
            this : Ball
            
            Returns
            -------
            area : float64
            """
            if not isinstance(self, m_geometry.Ball) :
                msg = f"Expecting '{m_geometry.Ball}' but got '{type(self)}'"
                raise TypeError(msg)
            area = _pywrapper.f90wrap_m_geometry__area__binding__ball(this=self._handle)
            return area
        
        def private_method(self, interface_call=False):
            """
            private_method(self)
            Defined at main-oo.fpp lines 214-215
            
            Parameters
            ----------
            this : Ball
            """
            if not isinstance(self, m_geometry.Ball) :
                msg = f"Expecting '{m_geometry.Ball}' but got '{type(self)}'"
                raise TypeError(msg)
            _pywrapper.f90wrap_m_geometry__private_method__binding__ball(this=self._handle)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.List_circle")
    class List_circle(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=list_circle)
        Defined at main-oo.fpp lines 85-94
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for list_circle
            
            self = List_Circle()
            Defined at main-oo.fpp lines 85-94
            
            Returns
            -------
            this : List_Circle
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_geometry__list_circle_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for list_circle
            
            Destructor for class List_Circle
            Defined at main-oo.fpp lines 85-94
            
            Parameters
            ----------
            this : List_Circle
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_geometry__list_circle_finalise(this=self._handle)
        
        def init(self, n, interface_call=False):
            """
            init(self, n)
            Defined at main-oo.fpp lines 143-151
            
            Parameters
            ----------
            this : List_Circle
            n : int32
            """
            if not isinstance(self, m_geometry.List_circle) :
                msg = f"Expecting '{m_geometry.List_circle}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(n,(numpy.ndarray, numpy.generic)):
                if not interface_call and n.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, 12, \
                    13}:
                    n = n.astype('int32')
                if n.ndim != 0 or n.dtype.num != 5:
                    raise TypeError("Expecting 'int' (code '5')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(n.dtype, n.dtype.num, n.ndim))
            elif not isinstance(n,int):
                raise TypeError("Expecting 'int' but got '%s'"%type(n))
            _pywrapper.f90wrap_m_geometry__init__binding__list_circle(this=self._handle, \
                n=n)
        
        def init_array_alloc_type(self):
            self.alloc_type = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_getitem__alloc_type,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_setitem__alloc_type,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_len__alloc_type,
                                                """
            Element alloc_type ftype=type(circle) pytype=Circle
            Defined at main-oo.fpp line 86
            """, M_Geometry.Circle)
            return self.alloc_type
        
        def init_array_ptr_type(self):
            self.ptr_type = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_getitem__ptr_type,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_setitem__ptr_type,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_len__ptr_type,
                                                """
            Element ptr_type ftype=type(circle) pytype=Circle
            Defined at main-oo.fpp line 87
            """, M_Geometry.Circle)
            return self.ptr_type
        
        def init_array_alloc_class(self):
            self.alloc_class = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_getitem__alloc_class,
                                                None,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_len__alloc_class,
                                                """
            Element alloc_class ftype=class(circle) pytype=Circle
            Defined at main-oo.fpp line 88
            """, M_Geometry.Circle)
            return self.alloc_class
        
        def init_array_ptr_class(self):
            self.ptr_class = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_getitem__ptr_class,
                                                None,
                                                _pywrapper.f90wrap_m_geometry__list_circle__array_len__ptr_class,
                                                """
            Element ptr_class ftype=class(circle) pytype=Circle
            Defined at main-oo.fpp line 89
            """, M_Geometry.Circle)
            return self.ptr_class
        
        @property
        def scalar_class(self):
            """
            Element scalar_class ftype=class(circle) pytype=Circle
            Defined at main-oo.fpp line 90
            """
            scalar_class_handle = \
                _pywrapper.f90wrap_m_geometry__list_circle__get__scalar_class(self._handle)
            if tuple(scalar_class_handle) in self._objs:
                scalar_class = self._objs[tuple(scalar_class_handle)]
            else:
                scalar_class = m_geometry.Circle.from_handle(scalar_class_handle)
                self._objs[tuple(scalar_class_handle)] = scalar_class
            return scalar_class
        
        @scalar_class.setter
        def scalar_class(self, scalar_class):
            scalar_class = scalar_class._handle
            _pywrapper.f90wrap_m_geometry__list_circle__set__scalar_class(self._handle, \
                scalar_class)
        
        @property
        def scalar_type(self):
            """
            Element scalar_type ftype=type(circle) pytype=Circle
            Defined at main-oo.fpp line 91
            """
            scalar_type_handle = \
                _pywrapper.f90wrap_m_geometry__list_circle__get__scalar_type(self._handle)
            if tuple(scalar_type_handle) in self._objs:
                scalar_type = self._objs[tuple(scalar_type_handle)]
            else:
                scalar_type = m_geometry.Circle.from_handle(scalar_type_handle)
                self._objs[tuple(scalar_type_handle)] = scalar_type
            return scalar_type
        
        @scalar_type.setter
        def scalar_type(self, scalar_type):
            scalar_type = scalar_type._handle
            _pywrapper.f90wrap_m_geometry__list_circle__set__scalar_type(self._handle, \
                scalar_type)
        
        @property
        def n(self):
            """
            Element n ftype=integer  pytype=int
            Defined at main-oo.fpp line 92
            """
            return _pywrapper.f90wrap_m_geometry__list_circle__get__n(self._handle)
        
        @n.setter
        def n(self, n):
            _pywrapper.f90wrap_m_geometry__list_circle__set__n(self._handle, n)
        
        def __str__(self):
            ret = ['<list_circle>{\n']
            ret.append('    scalar_class : ')
            ret.append(repr(self.scalar_class))
            ret.append(',\n    scalar_type : ')
            ret.append(repr(self.scalar_type))
            ret.append(',\n    n : ')
            ret.append(repr(self.n))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = [init_array_alloc_type, init_array_ptr_type, \
            init_array_alloc_class, init_array_ptr_class]
        
    
    @f90wrap.runtime.register_class("pywrapper.Array")
    class Array(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=array)
        Defined at main-oo.fpp lines 96-100
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for array
            
            self = Array()
            Defined at main-oo.fpp lines 96-100
            
            Returns
            -------
            this : Array
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_geometry__array_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for array
            
            Destructor for class Array
            Defined at main-oo.fpp lines 96-100
            
            Parameters
            ----------
            this : Array
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_geometry__array_finalise(this=self._handle)
        
        def init(self, n, interface_call=False):
            """
            init(self, n)
            Defined at main-oo.fpp lines 113-117
            
            Parameters
            ----------
            this : Array
            n : int32
            """
            if not isinstance(self, m_geometry.Array) :
                msg = f"Expecting '{m_geometry.Array}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(n,(numpy.ndarray, numpy.generic)):
                if not interface_call and n.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, 12, \
                    13}:
                    n = n.astype('int32')
                if n.ndim != 0 or n.dtype.num != 5:
                    raise TypeError("Expecting 'int' (code '5')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(n.dtype, n.dtype.num, n.ndim))
            elif not isinstance(n,int):
                raise TypeError("Expecting 'int' but got '%s'"%type(n))
            _pywrapper.f90wrap_m_geometry__init__binding__array(this=self._handle, n=n)
        
        @property
        def buf(self):
            """
            Element buf ftype=real pytype=float
            Defined at main-oo.fpp line 97
            """
            array_ndim, array_type, array_shape, array_handle = \
                _pywrapper.f90wrap_m_geometry__array__array__buf(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                buf = self._arrays[array_hash]
            else:
                try:
                    buf = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _pywrapper.f90wrap_m_geometry__array__array__buf)
                except TypeError:
                    buf = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = buf
            return buf
        
        @buf.setter
        def buf(self, buf):
            self.buf[...] = buf
        
        @property
        def values(self):
            """
            Element values ftype=real pytype=float
            Defined at main-oo.fpp line 98
            """
            array_ndim, array_type, array_shape, array_handle = \
                _pywrapper.f90wrap_m_geometry__array__array__values(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                values = self._arrays[array_hash]
            else:
                try:
                    values = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _pywrapper.f90wrap_m_geometry__array__array__values)
                except TypeError:
                    values = f90wrap.runtime.direct_c_array(array_type, array_shape, array_handle)
                self._arrays[array_handle] = values
            return values
        
        @values.setter
        def values(self, values):
            self.values[...] = values
        
        def __str__(self):
            ret = ['<array>{\n']
            ret.append('    buf : ')
            ret.append(repr(self.buf))
            ret.append(',\n    values : ')
            ret.append(repr(self.values))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.Array_3d")
    class Array_3d(Array):
        """
        Type(name=array_3d)
        Defined at main-oo.fpp lines 102-105
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for array_3d
            
            self = Array_3D()
            Defined at main-oo.fpp lines 102-105
            
            Returns
            -------
            this : Array_3D
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_geometry__array_3d_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for array_3d
            
            Destructor for class Array_3D
            Defined at main-oo.fpp lines 102-105
            
            Parameters
            ----------
            this : Array_3D
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_geometry__array_3d_finalise(this=self._handle)
        
        def init_3d(self, n1, n2, n3, interface_call=False):
            """
            init_3d(self, n1, n2, n3)
            Defined at main-oo.fpp lines 119-125
            
            Parameters
            ----------
            this : Array_3D
            n1 : int32
            n2 : int32
            n3 : int32
            """
            if not isinstance(self, m_geometry.Array_3d) :
                msg = f"Expecting '{m_geometry.Array_3d}' but got '{type(self)}'"
                raise TypeError(msg)
            if isinstance(n1,(numpy.ndarray, numpy.generic)):
                if not interface_call and n1.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, 12, \
                    13}:
                    n1 = n1.astype('int32')
                if n1.ndim != 0 or n1.dtype.num != 5:
                    raise TypeError("Expecting 'int' (code '5')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(n1.dtype, n1.dtype.num, n1.ndim))
            elif not isinstance(n1,int):
                raise TypeError("Expecting 'int' but got '%s'"%type(n1))
            if isinstance(n2,(numpy.ndarray, numpy.generic)):
                if not interface_call and n2.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, 12, \
                    13}:
                    n2 = n2.astype('int32')
                if n2.ndim != 0 or n2.dtype.num != 5:
                    raise TypeError("Expecting 'int' (code '5')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(n2.dtype, n2.dtype.num, n2.ndim))
            elif not isinstance(n2,int):
                raise TypeError("Expecting 'int' but got '%s'"%type(n2))
            if isinstance(n3,(numpy.ndarray, numpy.generic)):
                if not interface_call and n3.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, 12, \
                    13}:
                    n3 = n3.astype('int32')
                if n3.ndim != 0 or n3.dtype.num != 5:
                    raise TypeError("Expecting 'int' (code '5')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(n3.dtype, n3.dtype.num, n3.ndim))
            elif not isinstance(n3,int):
                raise TypeError("Expecting 'int' but got '%s'"%type(n3))
            _pywrapper.f90wrap_m_geometry__init_3d__binding__array_3d(this=self._handle, \
                n1=n1, n2=n2, n3=n3)
        
        @property
        def values_3d(self):
            """
            Element values_3d ftype=real pytype=float
            Defined at main-oo.fpp line 103
            """
            array_ndim, array_type, array_shape, array_handle = \
                _pywrapper.f90wrap_m_geometry__array_3d__array__values_3d(self._handle)
            array_hash = hash((array_ndim, array_type, tuple(array_shape), array_handle))
            if array_hash in self._arrays:
                values_3d = self._arrays[array_hash]
            else:
                try:
                    values_3d = f90wrap.runtime.get_array(f90wrap.runtime.sizeof_fortran_t,
                                            self._handle,
                                            _pywrapper.f90wrap_m_geometry__array_3d__array__values_3d)
                except TypeError:
                    values_3d = f90wrap.runtime.direct_c_array(array_type, array_shape, \
                        array_handle)
                self._arrays[array_handle] = values_3d
            return values_3d
        
        @values_3d.setter
        def values_3d(self, values_3d):
            self.values_3d[...] = values_3d
        
        def __str__(self):
            ret = ['<array_3d>{\n']
            ret.append('    values_3d : ')
            ret.append(repr(self.values_3d))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def get_circle_radius(self, interface_call=False):
        """
        radius = get_circle_radius(self)
        Defined at main-oo.fpp lines 163-166
        
        Parameters
        ----------
        my_circle : Circle
        
        Returns
        -------
        radius : float64
        """
        if not isinstance(self, m_geometry.Circle) :
            msg = f"Expecting '{m_geometry.Circle}' but got '{type(self)}'"
            raise TypeError(msg)
        radius = \
            _pywrapper.f90wrap_m_geometry__get_circle_radius(my_circle=self._handle)
        return radius
    
    @staticmethod
    def get_ball_radius(self, interface_call=False):
        """
        radius = get_ball_radius(self)
        Defined at main-oo.fpp lines 168-171
        
        Parameters
        ----------
        my_ball : Ball
        
        Returns
        -------
        radius : float64
        """
        if not isinstance(self, m_geometry.Ball) :
            msg = f"Expecting '{m_geometry.Ball}' but got '{type(self)}'"
            raise TypeError(msg)
        radius = _pywrapper.f90wrap_m_geometry__get_ball_radius(my_ball=self._handle)
        return radius
    
    @property
    def pi(self):
        """
        Element pi ftype=real(kind=8) pytype=float
        Defined at main-oo.fpp line 10
        """
        return _pywrapper.f90wrap_m_geometry__get__pi()
    
    def get_pi(self):
        return self.pi
    
    def __str__(self):
        ret = ['<m_geometry>{\n']
        ret.append('    pi : ')
        ret.append(repr(self.pi))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    
    if not hasattr(_pywrapper, "f90wrap_m_geometry__perimeter__binding__rectangle"):
        for _candidate in ["f90wrap_m_geometry__perimeter__binding__rectangle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__perimeter__binding__rectangle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__is_square__binding__rectangle"):
        for _candidate in ["f90wrap_m_geometry__is_square__binding__rectangle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__is_square__binding__rectangle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__area__binding__rectangle"):
        for _candidate in ["f90wrap_m_geometry__area__binding__rectangle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__area__binding__rectangle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, \
        "f90wrap_m_geometry__is_polygone__binding__rectangle"):
        for _candidate in \
            ["f90wrap_m_geometry__is_polygone__binding__polygone_rectangle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__is_polygone__binding__rectangle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__init__binding__square"):
        for _candidate in ["f90wrap_m_geometry__init__binding__square"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__init__binding__square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__is_square__binding__square"):
        for _candidate in ["f90wrap_m_geometry__is_square__binding__square"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__is_square__binding__square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__area__binding__square"):
        for _candidate in ["f90wrap_m_geometry__area__binding__square"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__area__binding__square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__is_equal__binding__square"):
        for _candidate in ["f90wrap_m_geometry__is_equal__binding__square"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__is_equal__binding__square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__copy__binding__square"):
        for _candidate in ["f90wrap_m_geometry__copy__binding__square"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__copy__binding__square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, \
        "f90wrap_m_geometry__create_diamond__binding__square"):
        for _candidate in ["f90wrap_m_geometry__create_diamond__binding__square"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__create_diamond__binding__square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__perimeter__binding__square"):
        for _candidate in ["f90wrap_m_geometry__perimeter__binding__rectangle_square"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__perimeter__binding__square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__is_polygone__binding__square"):
        for _candidate in \
            ["f90wrap_m_geometry__is_polygone__binding__polygone_rectangl6eb2"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__is_polygone__binding__square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__init__binding__diamond"):
        for _candidate in ["f90wrap_m_geometry__init__binding__diamond"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__init__binding__diamond", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__info__binding__diamond"):
        for _candidate in ["f90wrap_m_geometry__info__binding__diamond"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__info__binding__diamond", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__copy__binding__diamond"):
        for _candidate in ["f90wrap_m_geometry__copy__binding__diamond"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__copy__binding__diamond", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__is_polygone__binding__diamond"):
        for _candidate in \
            ["f90wrap_m_geometry__is_polygone__binding__polygone_diamond"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__is_polygone__binding__diamond", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__init__binding__list_square"):
        for _candidate in ["f90wrap_m_geometry__init__binding__list_square"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__init__binding__list_square", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__area__binding__circle"):
        for _candidate in ["f90wrap_m_geometry__area__binding__circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__area__binding__circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__print__binding__circle"):
        for _candidate in ["f90wrap_m_geometry__print__binding__circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__print__binding__circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__obj_name__binding__circle"):
        for _candidate in ["f90wrap_m_geometry__obj_name__binding__circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__obj_name__binding__circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__copy__binding__circle"):
        for _candidate in ["f90wrap_m_geometry__copy__binding__circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__copy__binding__circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__init__binding__circle"):
        for _candidate in ["f90wrap_m_geometry__init__binding__circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__init__binding__circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, \
        "f90wrap_m_geometry__private_method__binding__circle"):
        for _candidate in ["f90wrap_m_geometry__private_method__binding__circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__private_method__binding__circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__perimeter_4__binding__circle"):
        for _candidate in ["f90wrap_m_geometry__perimeter_4__binding__circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__perimeter_4__binding__circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__perimeter_8__binding__circle"):
        for _candidate in ["f90wrap_m_geometry__perimeter_8__binding__circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__perimeter_8__binding__circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__volume__binding__ball"):
        for _candidate in ["f90wrap_m_geometry__volume__binding__ball"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__volume__binding__ball", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__area__binding__ball"):
        for _candidate in ["f90wrap_m_geometry__area__binding__ball"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__area__binding__ball", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__private_method__binding__ball"):
        for _candidate in ["f90wrap_m_geometry__private_method__binding__ball"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__private_method__binding__ball", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__init__binding__list_circle"):
        for _candidate in ["f90wrap_m_geometry__init__binding__list_circle"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__init__binding__list_circle", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__init__binding__array"):
        for _candidate in ["f90wrap_m_geometry__init__binding__array"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__init__binding__array", \
                    getattr(_pywrapper, _candidate))
                break
    if not hasattr(_pywrapper, "f90wrap_m_geometry__init_3d__binding__array_3d"):
        for _candidate in ["f90wrap_m_geometry__init_3d__binding__array_3d"]:
            if hasattr(_pywrapper, _candidate):
                setattr(_pywrapper, "f90wrap_m_geometry__init_3d__binding__array_3d", \
                    getattr(_pywrapper, _candidate))
                break
    
    @staticmethod
    def area(instance, *args, **kwargs):
        return instance.area(*args, **kwargs)
    
    @staticmethod
    def copy(instance, *args, **kwargs):
        return instance.copy(*args, **kwargs)
    
    @staticmethod
    def create_diamond(instance, *args, **kwargs):
        return instance.create_diamond(*args, **kwargs)
    
    @staticmethod
    def info(instance, *args, **kwargs):
        return instance.info(*args, **kwargs)
    
    @staticmethod
    def init(instance, *args, **kwargs):
        return instance.init(*args, **kwargs)
    
    @staticmethod
    def init_3d(instance, *args, **kwargs):
        return instance.init_3d(*args, **kwargs)
    
    @staticmethod
    def is_equal(instance, *args, **kwargs):
        return instance.is_equal(*args, **kwargs)
    
    @staticmethod
    def is_polygone(instance, *args, **kwargs):
        return instance.is_polygone(*args, **kwargs)
    
    @staticmethod
    def is_square(instance, *args, **kwargs):
        return instance.is_square(*args, **kwargs)
    
    @staticmethod
    def obj_name(instance, *args, **kwargs):
        return instance.obj_name(*args, **kwargs)
    
    @staticmethod
    def perimeter(instance, *args, **kwargs):
        return instance.perimeter(*args, **kwargs)
    
    @staticmethod
    def perimeter_4(instance, *args, **kwargs):
        return instance.perimeter_4(*args, **kwargs)
    
    @staticmethod
    def perimeter_8(instance, *args, **kwargs):
        return instance.perimeter_8(*args, **kwargs)
    
    @staticmethod
    def print(instance, *args, **kwargs):
        return instance.print(*args, **kwargs)
    
    @staticmethod
    def private_method(instance, *args, **kwargs):
        return instance.private_method(*args, **kwargs)
    
    @staticmethod
    def volume(instance, *args, **kwargs):
        return instance.volume(*args, **kwargs)
    

m_geometry = M_Geometry()

