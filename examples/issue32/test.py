from __future__ import print_function, absolute_import, division
import _test
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_test = _SafeDirectCExecutor(_test, module_import_name='_test')

def foo(a, b, interface_call=False):
    """
    foo(a, b)
    Defined at test.fpp lines 5-7
    
    Parameters
    ----------
    a : float32
    b : int32
    """
    _test.f90wrap_foo(a=a, b=b)

