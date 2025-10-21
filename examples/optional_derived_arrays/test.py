from __future__ import print_function, absolute_import, division
import _test
import f90wrap.runtime
import logging
import numpy
import warnings

class Io(f90wrap.runtime.FortranModule):
    """
    Module io
    Defined at test.fpp lines 5-20
    """
    @f90wrap.runtime.register_class("test.keyword")
    class keyword(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=keyword)
        Defined at test.fpp lines 8-11
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for keyword
            
            self = Keyword()
            Defined at test.fpp lines 8-11
            
            Returns
            -------
            this : Keyword
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = _test.f90wrap_io__keyword_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for keyword
            
            Destructor for class Keyword
            Defined at test.fpp lines 8-11
            
            Parameters
            ----------
            this : Keyword
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _test.f90wrap_io__keyword_finalise(this=self._handle)
        
        @property
        def key(self):
            """
            Element key ftype=character(len=10) pytype=str
            Defined at test.fpp line 9
            """
            return _test.f90wrap_io__keyword__get__key(self._handle)
        
        @key.setter
        def key(self, key):
            _test.f90wrap_io__keyword__set__key(self._handle, key)
        
        @property
        def typ(self):
            """
            Element typ ftype=character(len=3) pytype=str
            Defined at test.fpp line 10
            """
            return _test.f90wrap_io__keyword__get__typ(self._handle)
        
        @typ.setter
        def typ(self, typ):
            _test.f90wrap_io__keyword__set__typ(self._handle, typ)
        
        @property
        def description(self):
            """
            Element description ftype=character(len=10) pytype=str
            Defined at test.fpp line 11
            """
            return _test.f90wrap_io__keyword__get__description(self._handle)
        
        @description.setter
        def description(self, description):
            _test.f90wrap_io__keyword__set__description(self._handle, description)
        
        def __str__(self):
            ret = ['<keyword>{\n']
            ret.append('    key : ')
            ret.append(repr(self.key))
            ret.append(',\n    typ : ')
            ret.append(repr(self.typ))
            ret.append(',\n    description : ')
            ret.append(repr(self.description))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    
    @staticmethod
    def io_freeform_open(filename, interface_call=False):
        """
        io_freeform_open(filename)
        Defined at test.fpp lines 15-20
        
        Parameters
        ----------
        filename : str
        """
        _test.f90wrap_io__io_freeform_open(filename=filename)
    
    _dt_array_initialisers = []
    
    

io = Io()

