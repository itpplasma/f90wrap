from __future__ import print_function, absolute_import, division
import _ExampleStrings_pkg
import f90wrap.runtime
import logging
import numpy
import warnings
import ExampleStrings_pkg.string_io
from f90wrap.safe_executor import SafeDirectCExecutor as _SafeDirectCExecutor
_ExampleStrings_pkg = _SafeDirectCExecutor(_ExampleStrings_pkg, \
    module_import_name='_ExampleStrings_pkg')

