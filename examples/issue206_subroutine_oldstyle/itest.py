from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_itest = _SafeDirectCExecutor(_itest, module_import_name='_itest')

def routine_with_oldstyle_asterisk(interface_call=False):
    """
    routine_with_oldstyle_asterisk()
    Defined at subroutine_oldstyle.fpp lines 5-8
    
    """
    _itest.f90wrap_routine_with_oldstyle_asterisk()

