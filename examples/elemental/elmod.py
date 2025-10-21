from __future__ import print_function, absolute_import, division
import _elmod
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_elmod = _SafeDirectCExecutor(_elmod, module_import_name='_elmod')

class Elemental_Module(f90wrap.runtime.FortranModule):
    """
    Module elemental_module
    Defined at elemental_module.fpp lines 5-15
    """
    @staticmethod
    def sinc(x, interface_call=False):
        """
        sinc = sinc(x)
        Defined at elemental_module.fpp lines 8-15
        
        Parameters
        ----------
        x : float64
        
        Returns
        -------
        sinc : float64
        """
        sinc = _elmod.f90wrap_elemental_module__sinc(x=x)
        return sinc
    
    _dt_array_initialisers = []
    

elemental_module = Elemental_Module()

