from __future__ import print_function, absolute_import, division
import _pywrapper
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_pywrapper = _SafeDirectCExecutor(_pywrapper, module_import_name='_pywrapper')

class M_Circle(f90wrap.runtime.FortranModule):
    """
    File: main.f90
    Test program docstring
    
    Author: test_author
    Copyright: test_copyright
    
    Module m_circle
    Defined at main.f90 lines 7-151
    """
    @f90wrap.runtime.register_class("pywrapper.t_circle")
    class t_circle(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_circle)
        Defined at main.f90 lines 10-11
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_circle
            
            self = T_Circle()
            Defined at main.f90 lines 10-11
            
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
                result = _pywrapper.f90wrap_m_circle__t_circle_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_circle
            
            Destructor for class T_Circle
            Defined at main.f90 lines 10-11
            
            Parameters
            ----------
            this : T_Circle
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _pywrapper.f90wrap_m_circle__t_circle_finalise(this=self._handle)
        
        @property
        def radius(self):
            """
            Element radius ftype=real  pytype=float
            Defined at main.f90 line 11
            """
            return _pywrapper.f90wrap_m_circle__t_circle__get__radius(self._handle)
        
        @radius.setter
        def radius(self, radius):
            _pywrapper.f90wrap_m_circle__t_circle__set__radius(self._handle, radius)
        
        def __str__(self):
            ret = ['<t_circle>{\n']
            ret.append('    radius : ')
            ret.append(repr(self.radius))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def construct_circle(self, radius, interface_call=False):
        """
        Initialize circle
        
        construct_circle(self, radius)
        Defined at main.f90 lines 26-29
        
        Parameters
        ----------
        circle : T_Circle
            t_circle to initialize [in,out]
        radius : float32
            radius of the circle [in]
        """
        _pywrapper.f90wrap_m_circle__construct_circle(circle=self._handle, \
            radius=radius)
    
    @staticmethod
    def construct_circle_more_doc(self, radius, interface_call=False):
        """
        Initialize circle with more doc
        
        Author: test_author
        Copyright: test_copyright
        
        construct_circle_more_doc(self, radius)
        Defined at main.f90 lines 39-42
        
        Parameters
        ----------
        circle : T_Circle
            t_circle to initialize [in,out]
        radius : float32
            radius of the circle [in]
        """
        _pywrapper.f90wrap_m_circle__construct_circle_more_doc(circle=self._handle, \
            radius=radius)
    
    @staticmethod
    def no_direction(self, radius, interface_call=False):
        """
        Without direction
        
        no_direction(self, radius)
        Defined at main.f90 lines 50-53
        
        Parameters
        ----------
        circle : T_Circle
            t_circle to initialize
        radius : float32
            radius of the circle
        """
        _pywrapper.f90wrap_m_circle__no_direction(circle=self._handle, radius=radius)
    
    @staticmethod
    def incomplete_doc_sub(self, radius, interface_call=False):
        """
        Incomplete doc
        
        incomplete_doc_sub(self, radius)
        Defined at main.f90 lines 60-63
        
        Parameters
        ----------
        circle : T_Circle
        radius : float32
            radius of the circle [in]
        """
        _pywrapper.f90wrap_m_circle__incomplete_doc_sub(circle=self._handle, \
            radius=radius)
    
    @staticmethod
    def doc_inside(self, radius, interface_call=False):
        """
        ===========================================================================
        >
         \\brief Doc inside
         \\param[in,out] circle      t_circle to initialize
         \\param[in]     radius      radius of the circle
        <
        
        doc_inside(self, radius)
        Defined at main.f90 lines 65-74
        
        Parameters
        ----------
        circle : T_Circle
        radius : float32
        """
        _pywrapper.f90wrap_m_circle__doc_inside(circle=self._handle, radius=radius)
    
    @staticmethod
    def output_1(interface_call=False):
        """
        subroutine output_1 outputs 1
        
        output = output_1()
        Defined at main.f90 lines 81-83
        
        Returns
        -------
        output : float32
            this is 1 [out]
        """
        output = _pywrapper.f90wrap_m_circle__output_1()
        return output
    
    @staticmethod
    def function_2(input, interface_call=False):
        """
        this is a function
        
        function_2 = function_2(input)
        Defined at main.f90 lines 91-93
        
        Parameters
        ----------
        input : str
            value [in]
        
        Returns
        -------
        function_2 : int32
            return value
        """
        function_2 = _pywrapper.f90wrap_m_circle__function_2(input=input)
        return function_2
    
    @staticmethod
    def details_doc(self, radius, interface_call=False):
        """
        Initialize circle
        
        Those are very informative details
        
        details_doc(self, radius)
        Defined at main.f90 lines 102-104
        
        Parameters
        ----------
        circle : T_Circle
            t_circle to initialize [in,out]
        radius : float32
            radius of the circle [in]
        """
        _pywrapper.f90wrap_m_circle__details_doc(circle=self._handle, radius=radius)
    
    @staticmethod
    def details_with_parenthesis(self, radius, interface_call=False):
        """
        Initialize circle
        
        Those are very informative details (with parenthesis)
        
        details_with_parenthesis(self, radius)
        Defined at main.f90 lines 113-115
        
        Parameters
        ----------
        circle : T_Circle
            t_circle to initialize [in,out]
        radius : float32
            radius of the circle [in]
        """
        _pywrapper.f90wrap_m_circle__details_with_parenthesis(circle=self._handle, \
            radius=radius)
    
    @staticmethod
    def multiline_details(self, radius, interface_call=False):
        """
        Initialize circle
        
        First details line
        Second details line
        
        multiline_details(self, radius)
        Defined at main.f90 lines 125-127
        
        Parameters
        ----------
        circle : T_Circle
            t_circle to initialize [in,out]
        radius : float32
            radius of the circle [in]
        """
        _pywrapper.f90wrap_m_circle__multiline_details(circle=self._handle, \
            radius=radius)
    
    @staticmethod
    def empty_lines_details(self, radius, interface_call=False):
        """
        Initialize circle
        
        First details line
        
        Second details line after a empty line
        
        empty_lines_details(self, radius)
        Defined at main.f90 lines 138-140
        
        Parameters
        ----------
        circle : T_Circle
            t_circle to initialize [in,out]
        radius : float32
            radius of the circle [in]
        """
        _pywrapper.f90wrap_m_circle__empty_lines_details(circle=self._handle, \
            radius=radius)
    
    @staticmethod
    def long_line_brief(self, radius, interface_call=False):
        """
        This is a very long brief that takes up a lot of space and contains lots of \
            information, it should probably be wrapped to the next line, but we will \
            continue regardless
        
        Those are very informative details
        
        long_line_brief(self, radius)
        Defined at main.f90 lines 149-151
        
        Parameters
        ----------
        circle : T_Circle
            t_circle to initialize [in,out]
        radius : float32
            radius of the circle [in]
        """
        _pywrapper.f90wrap_m_circle__long_line_brief(circle=self._handle, radius=radius)
    
    _dt_array_initialisers = []
    
    

m_circle = M_Circle()

