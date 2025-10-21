from __future__ import print_function, absolute_import, division
from .. import _ExampleArray_relative
import f90wrap.runtime
import logging
import numpy
import warnings
from . import library
from . import parameters
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_ExampleArray_relative = _SafeDirectCExecutor(_ExampleArray_relative, \
    module_import_name='_ExampleArray_relative')

