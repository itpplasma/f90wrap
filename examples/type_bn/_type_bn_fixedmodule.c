/* C Extension module for _type_bn_fixed */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <complex.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Shared capsule helper functions */
/* Note: The capsule_helpers.h file should be in the same directory as this generated code
   or you can adjust the include path as needed */
#include "capsule_helpers.h"

/* Fortran subroutine prototypes */


/* Derived type definitions */

/* Define capsule destructor for type_face */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(type_face)

/* Fortran derived type: type_face */
typedef struct {
    PyObject_HEAD
    void* fortran_ptr;  /* Opaque pointer to Fortran type instance */
    int owns_memory;     /* 1 if we own the Fortran memory */
} Pytype_face;

/* Forward declarations for type_face methods */
static PyObject* type_face_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void type_face_dealloc(Pytype_face *self);

/* Constructor for type_face */
static PyObject* type_face_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Pytype_face *self;

    self = (Pytype_face *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->fortran_ptr = NULL;
        self->owns_memory = 0;

        /* Allocate Fortran type instance */
        self->fortran_ptr = malloc(sizeof(int) * 8);  /* sizeof_fortran_t */
        if (self->fortran_ptr == NULL) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate Fortran type");
            return NULL;
        }
        self->owns_memory = 1;
    }

    return (PyObject *)self;
}

/* Destructor for type_face */
static void type_face_dealloc(Pytype_face *self) {
    if (self->fortran_ptr != NULL && self->owns_memory) {
        free(self->fortran_ptr);
        self->fortran_ptr = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Getter for type_face.type_bn */
static PyObject* type_face_get_type_bn(Pytype_face *self, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    int value;
    extern void __module_structure_MOD_f90wrap_type_face__get__type_bn(void*, int*);

    __module_structure_MOD_f90wrap_type_face__get__type_bn(self->fortran_ptr, &value);
    return PyLong_FromLong(value);
}

/* Setter for type_face.type_bn */
static int type_face_set_type_bn(Pytype_face *self, PyObject *value, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete type_bn");
        return -1;
    }

    int c_value;
    extern void __module_structure_MOD_f90wrap_type_face__set__type_bn(void*, int*);

    c_value = (int)PyLong_AsLong(value);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert type_bn");
        return -1;
    }

    __module_structure_MOD_f90wrap_type_face__set__type_bn(self->fortran_ptr, &c_value);
    return 0;
}

/* GetSet table for type_face */
static PyGetSetDef type_face_getsetters[] = {
    {"type_bn", (getter)type_face_get_type_bn, (setter)type_face_set_type_bn, "type_bn", NULL},
    {NULL}  /* Sentinel */
};

/* Type-bound method: type_face.type_face_initialise */
static PyObject* type_face_type_face_initialise(Pytype_face *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    void* this;

    this = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __module_structure_MOD_type_face_initialise(void*, void**);
    __module_structure_MOD_type_face_initialise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type type_face as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "type_face_capsule", type_face_capsule_destructor);
}

/* Type-bound method: type_face.type_face_finalise */
static PyObject* type_face_type_face_finalise(Pytype_face *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    PyObject *py_this = NULL;
    void* this;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type type_face */
    this = f90wrap_unwrap_capsule(py_this, "type_face");
    if (this == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __module_structure_MOD_type_face_finalise(void*, void**);
    __module_structure_MOD_type_face_finalise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type type_face as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "type_face_capsule", type_face_capsule_destructor);
}

/* Method table for type_face */
static PyMethodDef type_face_methods[] = {
    {"type_face_initialise", (PyCFunction)type_face_type_face_initialise, METH_VARARGS, "Type-bound method type_face_initialise"},
    {"type_face_finalise", (PyCFunction)type_face_type_face_finalise, METH_VARARGS, "Type-bound method type_face_finalise"},
    {NULL}  /* Sentinel */
};

/* Type object for type_face */
static PyTypeObject type_faceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_type_bn_fixed.type_face",
    .tp_basicsize = sizeof(Pytype_face),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)type_face_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Fortran derived type type_face",
    .tp_methods = type_face_methods,
    .tp_getset = type_face_getsetters,
    .tp_new = type_face_new,
};


/* Fortran subroutine prototypes */

/* Python wrapper functions */


/* Wrapper for wrap_type_face_create */
static char wrap_type_face_create__doc__[] = "Create a new type_face instance";

static PyObject* wrap_type_face_create(PyObject *self, PyObject *args, PyObject *kwargs) {

    /* Allocate new type_face instance */
    void* ptr = NULL;

    extern void __module_structure_MOD_f90wrap_type_face__allocate(void**);
    __module_structure_MOD_f90wrap_type_face__allocate(&ptr);

    if (ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate derived type");
        return NULL;
    }

    return f90wrap_create_capsule(ptr, "type_face_capsule", type_face_capsule_destructor);
}



/* Wrapper for wrap_type_face_destroy */
static char wrap_type_face_destroy__doc__[] = "Destroy a type_face instance";

static PyObject* wrap_type_face_destroy(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_capsule = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_capsule)) {
        return NULL;
    }

    void* ptr = f90wrap_unwrap_capsule(py_capsule, "type_face");
    if (ptr == NULL) {
        return NULL; /* Exception already set by GetPointer */
    }

    /* Deallocate type_face instance */
    extern void __module_structure_MOD_f90wrap_type_face__deallocate(void**);
    __module_structure_MOD_f90wrap_type_face__deallocate(&ptr);

    f90wrap_clear_capsule(py_capsule);

    Py_RETURN_NONE;
}



/* Method table */
static PyMethodDef _type_bn_fixed_methods[] = {
    {"wrap_type_face_create", (PyCFunction)wrap_type_face_create, METH_VARARGS, wrap_type_face_create__doc__},
    {"wrap_type_face_destroy", (PyCFunction)wrap_type_face_destroy, METH_VARARGS, wrap_type_face_destroy__doc__},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module definition */
static struct PyModuleDef _type_bn_fixed_module = {
    PyModuleDef_HEAD_INIT,
    "_type_bn_fixed",
    "Fortran module _type_bn_fixed wrapped with f90wrap",
    -1,
    _type_bn_fixed_methods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit__type_bn_fixed(void) {
    PyObject *module;

    /* Import NumPy C API */
    import_array();

    /* Initialize type_face type */
    if (PyType_Ready(&type_faceType) < 0) {
        return NULL;
    }

    /* Create module */
    module = PyModule_Create(&_type_bn_fixed_module);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&type_faceType);
    if (PyModule_AddObject(module, "type_face", (PyObject *)&type_faceType) < 0) {
        Py_DECREF(&type_faceType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
