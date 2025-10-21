from __future__ import print_function, absolute_import, division
import _test
import f90wrap.runtime
import logging
import numpy
import warnings

def wrap(def_, opt=None, interface_call=False):
    """
    wrap(def_[, opt])
    Defined at main.fpp lines 5-10
    
    Parameters
    ----------
    def_ : int32
    opt : int32
    """
    _test.f90wrap_wrap(def_=def_, opt=opt)

