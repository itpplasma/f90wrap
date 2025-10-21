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

name = 'pywrapper'
source = 'base_type.f90 composition_type.f90 inheritance_type.f90 main.f90 f90wrap_base_type.f90 f90wrap_composition_type.f90 f90wrap_inheritance_type.f90 f90wrap_main.f90'
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
