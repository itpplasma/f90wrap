# This Makefile shows manual build steps for f2py and Direct-C modes.
# For simple cases, you can use: f90wrap --build -m ExampleArray *.fpp -k kind_map
# This Makefile is for advanced use cases with external libraries or custom flags.

#=======================================================================
#                   define the compiler names
#=======================================================================

CC       = gcc
F90      = gfortran
#F90      = ifort
#F90      =  /opt/intel/composer_xe_2015.3.187/bin/intel64/ifort
PYTHON   = python
DIRECTC  ?= no

#=======================================================================
#                     additional flags
#=======================================================================

ifeq ($(F90),gfortran)
	FPP      = gfortran -E
	FPP_F90FLAGS = -x f95-cpp-input -fPIC
	F90FLAGS = -fPIC
    FCOMP    = gfortran
    LIBS     =
endif

ifeq ($(F90),ifort)

	FPP      = gfortran -E # gfortran f90wrap temp files only. not compilation
	FPP_F90FLAGS = -x f95-cpp-input -fPIC
	F90FLAGS = -fpscomp logicals -fPIC # use 1 and 0 for True and False
    FCOMP    = intelem # for f2py
    LIBS =
endif

CFLAGS  = -fPIC #     ==> universal for ifort, gfortran, pgi

ifeq ($(DIRECTC),yes)
    PYTHON_INCLUDES := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
    NUMPY_INCLUDES  := $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
    CFLAGS          += -I$(PYTHON_INCLUDES) -I$(NUMPY_INCLUDES)
    UNAME           := $(shell uname)
    ifeq ($(UNAME),Darwin)
        LDFLAGS     := -bundle -undefined dynamic_lookup
    else
        LDFLAGS     := -shared
    endif
endif

#=======================================================================
#=======================================================================

UNAME = $(shell uname)

ifeq (${UNAME}, Darwin)
  LIBTOOL = libtool -static -o
else
  LIBTOOL = ar src
endif

# ======================================================================
# PROJECT CONFIG, do not put spaced behind the variables
# ======================================================================
# Python module name
PYTHON_MODN = ExampleArray
# mapping between Fortran and C types
KIND_MAP = kind_map

#=======================================================================
#
#=======================================================================

#VPATH	=

#=======================================================================
#       List all source files required for the project
#=======================================================================

# names (without suffix), f90 sources
LIBSRC_SOURCES = parameters library

# file names
LIBSRC_FILES = $(addsuffix .f90,${LIBSRC_SOURCES})

# object files
LIBSRC_OBJECTS = $(addsuffix .o,${LIBSRC_SOURCES})

# only used when cleaning up
LIBSRC_FPP_FILES = $(addsuffix .fpp,${LIBSRC_SOURCES})

#=======================================================================
#       List all source files that require a Python interface
#=======================================================================

# names (without suffix), f90 sources
LIBSRC_WRAP_SOURCES = library parameters

# file names
LIBSRC_WRAP_FILES = $(addsuffix .f90,${LIBSRC_WRAP_SOURCES})

# object files
LIBSRC_WRAP_OBJECTS = $(addsuffix .o,${LIBSRC_WRAP_SOURCES})

# fpp files
LIBSRC_WRAP_FPP_FILES = $(addsuffix .fpp,${LIBSRC_WRAP_SOURCES})

#=======================================================================
#                 Relevant suffixes
#=======================================================================

.SUFFIXES: .f90 .fpp

#=======================================================================
#
#=======================================================================

.PHONY: all clean


all: _${PYTHON_MODN}.so _${PYTHON_MODN}_pkg.so test


clean:
	-rm -f ${LIBSRC_OBJECTS} ${LIBSRC_FPP_FILES} libsrc.a _*.so \
	*.mod *.fpp f90wrap*.f90 f90wrap*.o *.o ${PYTHON_MODN}.py
	-rm -f _*.c _*.o
	-rm -rf ${PYTHON_MODN}*/ ${PYTHON_MODN}_pkg/ ${PYTHON_MODN}_relative/
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/ build/


.f90.o:
	${F90} ${F90FLAGS} -c $< -o $@


.c.o:
	${CC} ${CFLAGS} -c $< -o $@


.f90.fpp:
	${FPP} ${FPP_F90FLAGS} $<  -o $@


libsrc.a: ${LIBSRC_OBJECTS}
	${LIBTOOL} $@ $?


ifeq ($(DIRECTC),yes)

_${PYTHON_MODN}.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap --direct-c -m ${PYTHON_MODN} ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v
	${F90} ${F90FLAGS} -c f90wrap_*.f90
	${CC} ${CFLAGS} -c _${PYTHON_MODN}.c
	${CC} ${LDFLAGS} -o _${PYTHON_MODN}.so _${PYTHON_MODN}.o f90wrap_*.o ${LIBSRC_OBJECTS}

_${PYTHON_MODN}_pkg.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap --direct-c -m ${PYTHON_MODN}_pkg ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v -P
	${F90} ${F90FLAGS} -c f90wrap_*.f90
	${CC} ${CFLAGS} -c _${PYTHON_MODN}_pkg.c
	${CC} ${LDFLAGS} -o _${PYTHON_MODN}_pkg.so _${PYTHON_MODN}_pkg.o f90wrap_*.o ${LIBSRC_OBJECTS}

_${PYTHON_MODN}_relative.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap --direct-c -m ${PYTHON_MODN}_relative ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v -P --relative
	${F90} ${F90FLAGS} -c f90wrap_*.f90
	${CC} ${CFLAGS} -c _${PYTHON_MODN}_relative.c
	${CC} ${LDFLAGS} -o _${PYTHON_MODN}_relative.so _${PYTHON_MODN}_relative.o f90wrap_*.o ${LIBSRC_OBJECTS}
	mkdir -p ${PYTHON_MODN}_top
	mv ${PYTHON_MODN}_relative/ _${PYTHON_MODN}_relative.so ${PYTHON_MODN}_top/
	touch ${PYTHON_MODN}_top/__init__.py

else

_${PYTHON_MODN}.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m ${PYTHON_MODN} ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v
	f2py-f90wrap --fcompiler=$(FCOMP) --build-dir build -c -m _${PYTHON_MODN} -L. -lsrc f90wrap*.f90


_${PYTHON_MODN}_pkg.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m ${PYTHON_MODN}_pkg ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v -P
	f2py-f90wrap --fcompiler=$(FCOMP) --build-dir build -c -m _${PYTHON_MODN}_pkg -L. -lsrc f90wrap*.f90

_${PYTHON_MODN}_relative.so: libsrc.a ${LIBSRC_FPP_FILES}
	f90wrap -m ${PYTHON_MODN}_relative ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v -P --relative
	f2py-f90wrap --fcompiler=$(FCOMP) --build-dir build -c -m _${PYTHON_MODN}_relative -L. -lsrc f90wrap*.f90
	mkdir -p ${PYTHON_MODN}_top
	mv ${PYTHON_MODN}_relative/ _${PYTHON_MODN}_relative.*so ${PYTHON_MODN}_top/
	touch ${PYTHON_MODN}_top/__init__.py

endif


test: _${PYTHON_MODN}_pkg.so _${PYTHON_MODN}.so  _${PYTHON_MODN}_relative.so 
	${PYTHON} tests.py

