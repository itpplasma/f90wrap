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

name = 'mockdtpkg'
source = './Source/HeliSrc/set_defaults.F90 ./Source/BasicDefs/assign_constants.F90 ./Source/BasicDefs/aa1_modules.F90 ./Source/BasicDefs/aa2_defineAllProperties.F90 ./Source/BasicDefs/aa0_typelist.F90 f90wrap_aa0_typelist.f90 f90wrap_aa1_modules.f90 f90wrap_aa2_defineAllProperties.f90 f90wrap_mockdtpkg.f90'
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
