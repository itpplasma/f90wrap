from __future__ import print_function, absolute_import, division
import _mockdtpkg
import f90wrap.runtime
import logging
import numpy
import warnings
import mockdtpkg.gaussian
import mockdtpkg.constant_parameters
import mockdtpkg.defineallproperties
import mockdtpkg.precision
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_mockdtpkg = _SafeDirectCExecutor(_mockdtpkg, module_import_name='_mockdtpkg')

def set_defaults(self, interface_call=False):
    """
    =======================================================================
                             EXECUTABLE CODE
    =======================================================================
    
    set_defaults(self)
    Defined at ./Source/HeliSrc/set_defaults.fpp lines 9-26
    
    Parameters
    ----------
    solver : Solveroptionsdef
    """
    _mockdtpkg.f90wrap_set_defaults(solver=self._handle)

def assign_constants(interface_call=False):
    """
    =======================================================================
    local constants (one-time use)
    =======================================================================
    
    assign_constants()
    Defined at ./Source/BasicDefs/assign_constants.fpp lines 8-74
    
    """
    _mockdtpkg.f90wrap_assign_constants()

