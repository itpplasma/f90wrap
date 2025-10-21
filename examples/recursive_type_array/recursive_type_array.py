from __future__ import print_function, absolute_import, division
import _recursive_type_array
import f90wrap.runtime
import logging
import numpy
import warnings

class Mod_Recursive_Type_Array(f90wrap.runtime.FortranModule):
    """
    Module mod_recursive_type_array
    Defined at test.fpp lines 5-18
    """
    @f90wrap.runtime.register_class("recursive_type_array.t_node")
    class t_node(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=t_node)
        Defined at test.fpp lines 7-8
        """
        def __init__(self, handle=None):
            """
            Automatically generated constructor for t_node
            
            self = T_Node()
            Defined at test.fpp lines 7-8
            
            Returns
            -------
            this : T_Node
                Object to be constructed
            
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            if handle is not None:
                self._handle = handle
                self._alloc = True
            else:
                result = \
                    _recursive_type_array.f90wrap_mod_recursive_type_array__t_node_initialise()
                self._handle = result[0] if isinstance(result, tuple) else result
                self._alloc = True
        
        def __del__(self):
            """
            Automatically generated destructor for t_node
            
            Destructor for class T_Node
            Defined at test.fpp lines 7-8
            
            Parameters
            ----------
            this : T_Node
                Object to be destructed
            
            """
            if getattr(self, '_alloc', False):
                _recursive_type_array.f90wrap_mod_recursive_type_array__t_node_finalise(this=self._handle)
        
        def init_array_node(self):
            self.node = f90wrap.runtime.FortranDerivedTypeArray(self,
                                                _recursive_type_array.f90wrap_mod_recursive_type_array__t_node__array_getitem__node,
                                                _recursive_type_array.f90wrap_mod_recursive_type_array__t_node__array_setitem__node,
                                                _recursive_type_array.f90wrap_mod_recursive_type_array__t_node__array_len__node,
                                                """
            Element node ftype=type(t_node) pytype=T_Node
            Defined at test.fpp line 8
            """, Mod_Recursive_Type_Array.t_node)
            return self.node
        
        _dt_array_initialisers = [init_array_node]
        
    
    @staticmethod
    def allocate_node(self, n_node, interface_call=False):
        """
        allocate_node(self, n_node)
        Defined at test.fpp lines 11-14
        
        Parameters
        ----------
        root : T_Node
        n_node : int32
        """
        _recursive_type_array.f90wrap_mod_recursive_type_array__allocate_node(root=self._handle, \
            n_node=n_node)
    
    @staticmethod
    def deallocate_node(self, interface_call=False):
        """
        deallocate_node(self)
        Defined at test.fpp lines 16-18
        
        Parameters
        ----------
        root : T_Node
        """
        _recursive_type_array.f90wrap_mod_recursive_type_array__deallocate_node(root=self._handle)
    
    _dt_array_initialisers = []
    
    

mod_recursive_type_array = Mod_Recursive_Type_Array()

