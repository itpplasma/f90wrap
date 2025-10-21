"""
Module tree
Defined at tree.fpp lines 5-21
"""
from __future__ import print_function, absolute_import, division
import _ExampleRecursive_pkg
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("ExampleRecursive_pkg.node")
class node(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=node)
    Defined at tree.fpp lines 7-9
    """
    def __init__(self, handle=None):
        """
        self = Node()
        Defined at tree.fpp lines 12-16
        
        Returns
        -------
        root : Node
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        if handle is not None:
            self._handle = handle
            self._alloc = True
        else:
            result = _ExampleRecursive_pkg.f90wrap_tree__treeallocate()
            self._handle = result[0] if isinstance(result, tuple) else result
            self._alloc = True
    
    def __del__(self):
        """
        Destructor for class Node
        Defined at tree.fpp lines 18-21
        
        Parameters
        ----------
        root : Node
        """
        if getattr(self, '_alloc', False):
            _ExampleRecursive_pkg.f90wrap_tree__treedeallocate(root=self._handle)
    
    @property
    def left(self):
        """
        Element left ftype=type(node) pytype=Node
        Defined at tree.fpp line 8
        """
        left_handle = _ExampleRecursive_pkg.f90wrap_tree__node__get__left(self._handle)
        if tuple(left_handle) in self._objs:
            left = self._objs[tuple(left_handle)]
        else:
            left = node.from_handle(left_handle)
            self._objs[tuple(left_handle)] = left
        return left
    
    @left.setter
    def left(self, left):
        left = left._handle
        _ExampleRecursive_pkg.f90wrap_tree__node__set__left(self._handle, left)
    
    @property
    def right(self):
        """
        Element right ftype=type(node) pytype=Node
        Defined at tree.fpp line 9
        """
        right_handle = \
            _ExampleRecursive_pkg.f90wrap_tree__node__get__right(self._handle)
        if tuple(right_handle) in self._objs:
            right = self._objs[tuple(right_handle)]
        else:
            right = node.from_handle(right_handle)
            self._objs[tuple(right_handle)] = right
        return right
    
    @right.setter
    def right(self, right):
        right = right._handle
        _ExampleRecursive_pkg.f90wrap_tree__node__set__right(self._handle, right)
    
    def __str__(self):
        ret = ['<node>{\n']
        ret.append('    left : ')
        ret.append(repr(self.left))
        ret.append(',\n    right : ')
        ret.append(repr(self.right))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    


_array_initialisers = []
_dt_array_initialisers = []


try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "tree".')

for func in _dt_array_initialisers:
    func()
