from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings

class M_Type_Test(f90wrap.runtime.FortranModule):
    """
    Module m_type_test
    Defined at main.fpp lines 5-163
    """
    @f90wrap.runtime.register_class("pywrapper.t_square")
    class t_square(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_square)
        Defined at main.fpp lines 8-9
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_square
            
            self = T_Square()
            Defined at main.fpp lines 8-9
            
            Returns
            -------
            this : T_Square
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_type_test__t_square_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_square
            
            Destructor for class T_Square
            Defined at main.fpp lines 8-9
            
            Parameters
            ----------
            this : T_Square
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_type_test__t_square_finalise(this=self._handle)
        
        @property
        def length(self):
            """
            Element length ftype=real  pytype=float
            Defined at main.fpp line 9
            """
            return _pywrapper.f90wrap_m_type_test__t_square__get__length(self._handle)
        
        @length.setter
        def length(self, length):
            _pywrapper.f90wrap_m_type_test__t_square__set__length(self._handle, length)
        
        def __str__(self):
            ret = ['<t_square>{\n']
            ret.append('    length : ')
            ret.append(repr(self.length))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @f90wrap.runtime.register_class("pywrapper.t_circle")
    class t_circle(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_circle)
        Defined at main.fpp lines 11-12
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_circle
            
            self = T_Circle()
            Defined at main.fpp lines 11-12
            
            Returns
            -------
            this : T_Circle
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _pywrapper.f90wrap_m_type_test__t_circle_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_circle
            
            Destructor for class T_Circle
            Defined at main.fpp lines 11-12
            
            Parameters
            ----------
            this : T_Circle
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_type_test__t_circle_finalise(this=self._handle)
        
        @property
        def radius(self):
            """
            Element radius ftype=real  pytype=float
            Defined at main.fpp line 12
            """
            return _pywrapper.f90wrap_m_type_test__t_circle__get__radius(self._handle)
        
        @radius.setter
        def radius(self, radius):
            _pywrapper.f90wrap_m_type_test__t_circle__set__radius(self._handle, radius)
        
        def __str__(self):
            ret = ['<t_circle>{\n']
            ret.append('    radius : ')
            ret.append(repr(self.radius))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def is_circle_circle(self, output, interface_call=False):
        """
        is_circle_circle(self, output)
        Defined at main.fpp lines 63-66
        
        Parameters
        ----------
        circle : T_Circle
        output : int array
        """
        if not isinstance(self, m_type_test.t_circle) :
            msg = f"Expecting '{m_type_test.t_circle}' but got '{type(self)}'"
            raise TypeError(msg)
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 1 or output.dtype.num != 5:
                raise TypeError("Expecting 'int' (code '5')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__is_circle_circle(circle=self._handle, \
            output=output)
    
    @staticmethod
    def is_circle_square(self, output, interface_call=False):
        """
        is_circle_square(self, output)
        Defined at main.fpp lines 68-71
        
        Parameters
        ----------
        square : T_Square
        output : int array
        """
        if not isinstance(self, m_type_test.t_square) :
            msg = f"Expecting '{m_type_test.t_square}' but got '{type(self)}'"
            raise TypeError(msg)
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 1 or output.dtype.num != 5:
                raise TypeError("Expecting 'int' (code '5')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__is_circle_square(square=self._handle, \
            output=output)
    
    @staticmethod
    def is_circle(*args, **kwargs):
        """
        is_circle(*args, **kwargs)
        Defined at main.fpp lines 14-16
        
        Overloaded interface containing the following procedures:
          is_circle_circle
          is_circle_square
        """
        for proc in [M_Type_Test.is_circle_circle, M_Type_Test.is_circle_square]:
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
            "is_circle compatible with the provided args:"
            "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
    
    @staticmethod
    def write_array_int32_0d(output, interface_call=False):
        """
        write_array_int32_0d(output)
        Defined at main.fpp lines 73-75
        
        Parameters
        ----------
        output : int32
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 0 or output.dtype.num != 5:
                raise TypeError("Expecting 'int' (code '5')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        elif not isinstance(output,int):
            raise TypeError("Expecting 'int' but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_int32_0d(output=output)
    
    @staticmethod
    def write_array_int64_0d(output, interface_call=False):
        """
        write_array_int64_0d(output)
        Defined at main.fpp lines 77-79
        
        Parameters
        ----------
        output : int64
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 0 or output.dtype.num != 7:
                raise TypeError("Expecting 'int' (code '7')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        elif not isinstance(output,int):
            raise TypeError("Expecting 'int' but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_int64_0d(output=output)
    
    @staticmethod
    def write_array_real32_0d(output, interface_call=False):
        """
        write_array_real32_0d(output)
        Defined at main.fpp lines 81-83
        
        Parameters
        ----------
        output : float32
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 0 or output.dtype.num != 11:
                raise TypeError("Expecting 'float' (code '11')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        elif not isinstance(output,float):
            raise TypeError("Expecting 'float' but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_real32_0d(output=output)
    
    @staticmethod
    def write_array_real64_0d(output, interface_call=False):
        """
        write_array_real64_0d(output)
        Defined at main.fpp lines 85-87
        
        Parameters
        ----------
        output : float64
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 0 or output.dtype.num != 12:
                raise TypeError("Expecting 'float' (code '12')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        elif not isinstance(output,float):
            raise TypeError("Expecting 'float' but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_real64_0d(output=output)
    
    @staticmethod
    def write_array_int_1d(output, interface_call=False):
        """
        write_array_int_1d(output)
        Defined at main.fpp lines 89-91
        
        Parameters
        ----------
        output : int array
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 1 or output.dtype.num != 5:
                raise TypeError("Expecting 'int' (code '5')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_int_1d(output=output)
    
    @staticmethod
    def write_array_int_2d(output, interface_call=False):
        """
        write_array_int_2d(output)
        Defined at main.fpp lines 93-95
        
        Parameters
        ----------
        output : int array
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 2 or output.dtype.num != 5:
                raise TypeError("Expecting 'int' (code '5')"
                " with dim '2' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_int_2d(output=output)
    
    @staticmethod
    def write_array_real(output, interface_call=False):
        """
        write_array_real(output)
        Defined at main.fpp lines 97-99
        
        Parameters
        ----------
        output : float array
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 1 or output.dtype.num != 11:
                raise TypeError("Expecting 'float' (code '11')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_real(output=output)
    
    @staticmethod
    def write_array_double(output, interface_call=False):
        """
        write_array_double(output)
        Defined at main.fpp lines 101-103
        
        Parameters
        ----------
        output : unknown array
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 1 or output.dtype.num != 12:
                raise TypeError("Expecting 'unknown' (code '12')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_double(output=output)
    
    @staticmethod
    def write_array_bool(output, interface_call=False):
        """
        write_array_bool(output)
        Defined at main.fpp lines 105-107
        
        Parameters
        ----------
        output : bool array
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 1 or output.dtype.num != 5:
                raise TypeError("Expecting 'bool' (code '5')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        _pywrapper.f90wrap_m_type_test__write_array_bool(output=output)
    
    @staticmethod
    def write_array(*args, **kwargs):
        """
        write_array(*args, **kwargs)
        Defined at main.fpp lines 18-27
        
        Overloaded interface containing the following procedures:
          write_array_int32_0d
          write_array_int64_0d
          write_array_real32_0d
          write_array_real64_0d
          write_array_int_1d
          write_array_int_2d
          write_array_real
          write_array_double
          write_array_bool
        """
        for proc in [M_Type_Test.write_array_int32_0d, M_Type_Test.write_array_int64_0d, \
            M_Type_Test.write_array_real32_0d, M_Type_Test.write_array_real64_0d, \
            M_Type_Test.write_array_int_1d, M_Type_Test.write_array_int_2d, \
            M_Type_Test.write_array_real, M_Type_Test.write_array_double, \
            M_Type_Test.write_array_bool]:
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
            "write_array compatible with the provided args:"
            "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
    
    @staticmethod
    def optional_scalar_real(output, opt_output=None, interface_call=False):
        """
        optional_scalar_real(output[, opt_output])
        Defined at main.fpp lines 109-115
        
        Parameters
        ----------
        output : float array
        opt_output : float32
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 1 or output.dtype.num != 11:
                raise TypeError("Expecting 'float' (code '11')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        if opt_output is not None:
            if isinstance(opt_output,(numpy.ndarray, numpy.generic)):
                if opt_output.ndim != 0 or opt_output.dtype.num != 11:
                    raise TypeError("Expecting 'float' (code '11')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(opt_output.dtype, opt_output.dtype.num, opt_output.ndim))
            elif not isinstance(opt_output,float):
                raise TypeError("Expecting 'float' but got '%s'"%type(opt_output))
        _pywrapper.f90wrap_m_type_test__optional_scalar_real(output=output, \
            opt_output=opt_output)
    
    @staticmethod
    def optional_scalar_int(output, opt_output=None, interface_call=False):
        """
        optional_scalar_int(output[, opt_output])
        Defined at main.fpp lines 117-123
        
        Parameters
        ----------
        output : int array
        opt_output : int32
        """
        if isinstance(output,(numpy.ndarray, numpy.generic)):
            if output.ndim != 1 or output.dtype.num != 5:
                raise TypeError("Expecting 'int' (code '5')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(output.dtype, output.dtype.num, output.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(output))
        if opt_output is not None:
            if isinstance(opt_output,(numpy.ndarray, numpy.generic)):
                if opt_output.ndim != 0 or opt_output.dtype.num != 5:
                    raise TypeError("Expecting 'int' (code '5')"
                    " with dim '0' but got '%s' (code '%s') with dim '%s'"
                    %(opt_output.dtype, opt_output.dtype.num, opt_output.ndim))
            elif not isinstance(opt_output,int):
                raise TypeError("Expecting 'int' but got '%s'"%type(opt_output))
        _pywrapper.f90wrap_m_type_test__optional_scalar_int(output=output, \
            opt_output=opt_output)
    
    @staticmethod
    def optional_scalar(*args, **kwargs):
        """
        optional_scalar(*args, **kwargs)
        Defined at main.fpp lines 29-31
        
        Overloaded interface containing the following procedures:
          optional_scalar_real
          optional_scalar_int
        """
        for proc in [M_Type_Test.optional_scalar_real, M_Type_Test.optional_scalar_int]:
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
            "optional_scalar compatible with the provided args:"
            "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
    
    @staticmethod
    def in_scalar_int8(input, interface_call=False):
        """
        output = in_scalar_int8(input)
        Defined at main.fpp lines 125-128
        
        Parameters
        ----------
        input : int8
        
        Returns
        -------
        output : int32
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if not interface_call and input.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                12, 13}:
                input = input.astype('int8')
            if input.ndim != 0 or input.dtype.num != 1:
                raise TypeError("Expecting 'int' (code '1')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        elif not isinstance(input,int):
            raise TypeError("Expecting 'int' but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_type_test__in_scalar_int8(input=input)
        return output
    
    @staticmethod
    def in_scalar_int16(input, interface_call=False):
        """
        output = in_scalar_int16(input)
        Defined at main.fpp lines 130-133
        
        Parameters
        ----------
        input : int16
        
        Returns
        -------
        output : int32
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if not interface_call and input.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                12, 13}:
                input = input.astype('int16')
            if input.ndim != 0 or input.dtype.num != 3:
                raise TypeError("Expecting 'int' (code '3')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        elif not isinstance(input,int):
            raise TypeError("Expecting 'int' but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_type_test__in_scalar_int16(input=input)
        return output
    
    @staticmethod
    def in_scalar_int32(input, interface_call=False):
        """
        output = in_scalar_int32(input)
        Defined at main.fpp lines 135-138
        
        Parameters
        ----------
        input : int32
        
        Returns
        -------
        output : int32
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if not interface_call and input.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                12, 13}:
                input = input.astype('int32')
            if input.ndim != 0 or input.dtype.num != 5:
                raise TypeError("Expecting 'int' (code '5')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        elif not isinstance(input,int):
            raise TypeError("Expecting 'int' but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_type_test__in_scalar_int32(input=input)
        return output
    
    @staticmethod
    def in_scalar_int64(input, interface_call=False):
        """
        output = in_scalar_int64(input)
        Defined at main.fpp lines 140-143
        
        Parameters
        ----------
        input : int64
        
        Returns
        -------
        output : int32
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if not interface_call and input.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                12, 13}:
                input = input.astype('int64')
            if input.ndim != 0 or input.dtype.num != 7:
                raise TypeError("Expecting 'int' (code '7')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        elif not isinstance(input,int):
            raise TypeError("Expecting 'int' but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_type_test__in_scalar_int64(input=input)
        return output
    
    @staticmethod
    def in_scalar_real32(input, interface_call=False):
        """
        output = in_scalar_real32(input)
        Defined at main.fpp lines 145-148
        
        Parameters
        ----------
        input : float32
        
        Returns
        -------
        output : int32
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if not interface_call and input.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                12, 13}:
                input = input.astype('float32')
            if input.ndim != 0 or input.dtype.num != 11:
                raise TypeError("Expecting 'float' (code '11')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        elif not isinstance(input,float):
            raise TypeError("Expecting 'float' but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_type_test__in_scalar_real32(input=input)
        return output
    
    @staticmethod
    def in_scalar_real64(input, interface_call=False):
        """
        output = in_scalar_real64(input)
        Defined at main.fpp lines 150-153
        
        Parameters
        ----------
        input : float64
        
        Returns
        -------
        output : int32
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if not interface_call and input.dtype.num in {3, 4, 5, 6, 7, 8, 9, 10, 23, 11, \
                12, 13}:
                input = input.astype('float64')
            if input.ndim != 0 or input.dtype.num != 12:
                raise TypeError("Expecting 'float' (code '12')"
                " with dim '0' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        elif not isinstance(input,float):
            raise TypeError("Expecting 'float' but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_type_test__in_scalar_real64(input=input)
        return output
    
    @staticmethod
    def in_array_int64(input, interface_call=False):
        """
        output = in_array_int64(input)
        Defined at main.fpp lines 155-158
        
        Parameters
        ----------
        input : int array
        
        Returns
        -------
        output : int32
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if input.ndim != 1 or input.dtype.num != 7:
                raise TypeError("Expecting 'int' (code '7')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_type_test__in_array_int64(input=input)
        return output
    
    @staticmethod
    def in_array_real64(input, interface_call=False):
        """
        output = in_array_real64(input)
        Defined at main.fpp lines 160-163
        
        Parameters
        ----------
        input : float array
        
        Returns
        -------
        output : int32
        """
        if isinstance(input,(numpy.ndarray, numpy.generic)):
            if input.ndim != 1 or input.dtype.num != 12:
                raise TypeError("Expecting 'float' (code '12')"
                " with dim '1' but got '%s' (code '%s') with dim '%s'"
                %(input.dtype, input.dtype.num, input.ndim))
        else:
            raise TypeError("Expecting numpy array but got '%s'"%type(input))
        output = _pywrapper.f90wrap_m_type_test__in_array_real64(input=input)
        return output
    
    @staticmethod
    def in_scalar(*args, **kwargs):
        """
        in_scalar(*args, **kwargs)
        Defined at main.fpp lines 33-41
        
        Overloaded interface containing the following procedures:
          in_scalar_int8
          in_scalar_int16
          in_scalar_int32
          in_scalar_int64
          in_scalar_real32
          in_scalar_real64
          in_array_int64
          in_array_real64
        """
        for proc in [M_Type_Test.in_scalar_int8, M_Type_Test.in_scalar_int16, \
            M_Type_Test.in_scalar_int32, M_Type_Test.in_scalar_int64, \
            M_Type_Test.in_scalar_real32, M_Type_Test.in_scalar_real64, \
            M_Type_Test.in_array_int64, M_Type_Test.in_array_real64]:
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
            "in_scalar compatible with the provided args:"
            "\n%s\nLast exception was: %s"%("\n".join(argTypes), exception))
    
    _dt_array_initialisers = []
    
    

m_type_test = M_Type_Test()

