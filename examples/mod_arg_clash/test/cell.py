"""
Module cell
Defined at test.fpp lines 5-21
"""
from __future__ import print_function, absolute_import, division
import _test
import f90wrap.runtime
import logging
import numpy
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=numpy.exceptions.ComplexWarning)
_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("test.unit_cell")
class unit_cell(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=unit_cell)
    Defined at test.fpp lines 8-10
    """
    def __init__(self, handle=None):
        """
        Automatically generated constructor for unit_cell
        
        self = Unit_Cell()
        Defined at test.fpp lines 8-10
        
        Returns
        -------
        this : Unit_Cell
            Object to be constructed
        
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        if handle is not None:
            self._handle = handle
            self._alloc = True
        else:
            result = _test.f90wrap_cell__unit_cell_initialise()
            self._handle = result[0] if isinstance(result, tuple) else result
            self._alloc = True
    
    def __del__(self):
        """
        Automatically generated destructor for unit_cell
        
        Destructor for class Unit_Cell
        Defined at test.fpp lines 8-10
        
        Parameters
        ----------
        this : Unit_Cell
            Object to be destructed
        
        """
        if getattr(self, '_alloc', False):
            _test.f90wrap_cell__unit_cell_finalise(this=self._handle)
    
    @property
    def num_species(self):
        """
        Element num_species ftype=integer          pytype=int
        Defined at test.fpp line 9
        """
        return _test.f90wrap_cell__unit_cell__get__num_species(self._handle)
    
    @num_species.setter
    def num_species(self, num_species):
        _test.f90wrap_cell__unit_cell__set__num_species(self._handle, num_species)
    
    @property
    def species_symbol(self):
        """
        Element species_symbol ftype=character(len=8) pytype=str
        Defined at test.fpp line 10
        """
        return _test.f90wrap_cell__unit_cell__get__species_symbol(self._handle)
    
    @species_symbol.setter
    def species_symbol(self, species_symbol):
        _test.f90wrap_cell__unit_cell__set__species_symbol(self._handle, species_symbol)
    
    def __str__(self):
        ret = ['<unit_cell>{\n']
        ret.append('    num_species : ')
        ret.append(repr(self.num_species))
        ret.append(',\n    species_symbol : ')
        ret.append(repr(self.species_symbol))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

def cell_dosomething(self, num_species, species_symbol, interface_call=False):
    """
    cell_dosomething(self, num_species, species_symbol)
    Defined at test.fpp lines 14-21
    
    Parameters
    ----------
    cell_ : Unit_Cell
    num_species : int32
    species_symbol : str
    """
    _test.f90wrap_cell__cell_dosomething(cell_=self._handle, \
        num_species=num_species, species_symbol=species_symbol)


_array_initialisers = []
_dt_array_initialisers = []


try:
    for func in _array_initialisers:
        func()
except ValueError:
    logger.debug('unallocated array(s) detected on import of module "cell".')

for func in _dt_array_initialisers:
    func()
