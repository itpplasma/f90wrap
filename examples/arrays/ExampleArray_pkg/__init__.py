from __future__ import print_function, absolute_import, division
import _ExampleArray_pkg
import f90wrap.runtime
import logging
import numpy
import warnings
import ExampleArray_pkg.library
import ExampleArray_pkg.parameters
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_ExampleArray_pkg = _SafeDirectCExecutor(_ExampleArray_pkg, \
    module_import_name='_ExampleArray_pkg')

