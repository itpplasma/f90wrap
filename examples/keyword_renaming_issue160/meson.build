project('directc',
  ['c', 'fortran'],
  version : '0.1',
  meson_version: '>= 1.1.0',
  default_options : [
    'warning_level=1',
    'buildtype=release'
    ])

add_languages('fortran', native: false)
ff = meson.get_compiler('fortran')
if ff.has_argument('-Wno-conversion')
  add_project_arguments('-Wno-conversion', language: 'fortran')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

# Find gfortran library
cc = meson.get_compiler('c')
gfortran_dep = cc.find_library('gfortran', required: true)

name = 'keywordr_rename'
source = 'rename.f90 _keywordr_rename-f2pywrappers.f f90wrap_keywordr_rename.f90 f90wrap_rename.f90'
source = source.split()
c_source = '_' + name + '.c'
source += [c_source]

# additional_flags
add_project_arguments('-xf95-cpp-input', language: 'fortran')
add_project_arguments('-fPIC', language: 'fortran')

py.extension_module('_' + name,
  source,
  include_directories: inc_np,
  dependencies : [py_dep, gfortran_dep],
  install : true
  )
