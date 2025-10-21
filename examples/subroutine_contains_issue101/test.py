from __future__ import print_function, absolute_import, division
import _test
import f90wrap.runtime
import logging
import numpy
import warnings
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_test = _SafeDirectCExecutor(_test, module_import_name='_test')

def routine_member_procedures(in1, in2, interface_call=False):
    """
    out1, out2 = routine_member_procedures(in1, in2)
    Defined at test.fpp lines 5-28
    
    Parameters
    ----------
    in1 : int32
    in2 : int32
    
    Returns
    -------
    out1 : int32
    out2 : int32
    """
    out1, out2 = _test.f90wrap_routine_member_procedures(in1=in1, in2=in2)
    return out1, out2

def routine_member_procedures2(in1, in2, interface_call=False):
    """
    out1, out2 = routine_member_procedures2(in1, in2)
    Defined at test.fpp lines 30-87
    
    Parameters
    ----------
    in1 : int32
    in2 : int32
    
    Returns
    -------
    out1 : int32
    out2 : int32
    """
    out1, out2 = _test.f90wrap_routine_member_procedures2(in1=in1, in2=in2)
    return out1, out2

def function_member_procedures(in1, in2, interface_call=False):
    """
    out1, out2, out3 = function_member_procedures(in1, in2)
    Defined at test.fpp lines 89-109
    
    Parameters
    ----------
    in1 : int32
    in2 : int32
    
    Returns
    -------
    out1 : int32
    out2 : int32
    out3 : int32
    """
    out1, out2, out3 = _test.f90wrap_function_member_procedures(in1=in1, in2=in2)
    return out1, out2, out3

