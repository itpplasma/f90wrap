/* C Extension module for _issue258_derived_type_attributes_direct */

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

/* Define capsule destructor for t_inner */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(t_inner)
/* Define capsule destructor for t_outer */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(t_outer)
/* Define capsule destructor for c_ptr (Fortran intrinsic type) */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(c_ptr)

/* Fortran derived type: t_inner */
typedef struct {
    PyObject_HEAD
    void* fortran_ptr;  /* Opaque pointer to Fortran type instance */
    int owns_memory;     /* 1 if we own the Fortran memory */
} Pyt_inner;

/* Forward declarations for t_inner methods */
static PyObject* t_inner_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void t_inner_dealloc(Pyt_inner *self);

/* Constructor for t_inner */
static PyObject* t_inner_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Pyt_inner *self;

    self = (Pyt_inner *)type->tp_alloc(type, 0);
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

/* Destructor for t_inner */
static void t_inner_dealloc(Pyt_inner *self) {
    if (self->fortran_ptr != NULL && self->owns_memory) {
        free(self->fortran_ptr);
        self->fortran_ptr = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Getter for t_inner.value */
static PyObject* t_inner_get_value(Pyt_inner *self, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    int value;
    extern void __dta_ct_MOD_f90wrap_t_inner__get__value(void*, int*);

    __dta_ct_MOD_f90wrap_t_inner__get__value(self->fortran_ptr, &value);
    return PyLong_FromLong(value);
}

/* Setter for t_inner.value */
static int t_inner_set_value(Pyt_inner *self, PyObject *value, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete value");
        return -1;
    }

    int c_value;
    extern void __dta_ct_MOD_f90wrap_t_inner__set__value(void*, int*);

    c_value = (int)PyLong_AsLong(value);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert value");
        return -1;
    }

    __dta_ct_MOD_f90wrap_t_inner__set__value(self->fortran_ptr, &c_value);
    return 0;
}

/* GetSet table for t_inner */
static PyGetSetDef t_inner_getsetters[] = {
    {"value", (getter)t_inner_get_value, (setter)t_inner_set_value, "value", NULL},
    {NULL}  /* Sentinel */
};

/* Type-bound method: t_inner.new_inner */
static PyObject* t_inner_new_inner(Pyt_inner *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    PyObject *py_value = NULL;
    int value;

    if (!PyArg_ParseTuple(args, "i", &py_value)) {
        return NULL;
    }

    value = (int)PyLong_AsLong(py_value);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument value");
        return NULL;
    }

    /* Call Fortran function */
    extern void* __dta_ct_MOD_new_inner(void*, int*);
    void* result;
    result = __dta_ct_MOD_new_inner(self->fortran_ptr, &value);

    /* Return derived type t_inner as PyCapsule */
    if (result == NULL) {
        Py_RETURN_NONE;
    }

    return f90wrap_create_capsule(result, "t_inner_capsule", t_inner_capsule_destructor);
}

/* Type-bound method: t_inner.t_inner_finalise */
static PyObject* t_inner_t_inner_finalise(Pyt_inner *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    PyObject *py_this = NULL;
    void* this;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_inner */
    this = f90wrap_unwrap_capsule(py_this, "t_inner");
    if (this == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __dta_ct_MOD_t_inner_finalise(void*, void**);
    __dta_ct_MOD_t_inner_finalise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type t_inner as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "t_inner_capsule", t_inner_capsule_destructor);
}

/* Method table for t_inner */
static PyMethodDef t_inner_methods[] = {
    {"new_inner", (PyCFunction)t_inner_new_inner, METH_VARARGS, "Type-bound method new_inner"},
    {"t_inner_finalise", (PyCFunction)t_inner_t_inner_finalise, METH_VARARGS, "Type-bound method t_inner_finalise"},
    {NULL}  /* Sentinel */
};

/* Type object for t_inner */
static PyTypeObject t_innerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_issue258_derived_type_attributes_direct.t_inner",
    .tp_basicsize = sizeof(Pyt_inner),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)t_inner_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Fortran derived type t_inner",
    .tp_methods = t_inner_methods,
    .tp_getset = t_inner_getsetters,
    .tp_new = t_inner_new,
};


/* Fortran derived type: t_outer */
typedef struct {
    PyObject_HEAD
    void* fortran_ptr;  /* Opaque pointer to Fortran type instance */
    int owns_memory;     /* 1 if we own the Fortran memory */
} Pyt_outer;

/* Forward declarations for t_outer methods */
static PyObject* t_outer_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void t_outer_dealloc(Pyt_outer *self);

/* Constructor for t_outer */
static PyObject* t_outer_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Pyt_outer *self;

    self = (Pyt_outer *)type->tp_alloc(type, 0);
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

/* Destructor for t_outer */
static void t_outer_dealloc(Pyt_outer *self) {
    if (self->fortran_ptr != NULL && self->owns_memory) {
        free(self->fortran_ptr);
        self->fortran_ptr = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Getter for t_outer.value */
static PyObject* t_outer_get_value(Pyt_outer *self, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    int value;
    extern void __dta_ct_MOD_f90wrap_t_outer__get__value(void*, int*);

    __dta_ct_MOD_f90wrap_t_outer__get__value(self->fortran_ptr, &value);
    return PyLong_FromLong(value);
}

/* Setter for t_outer.value */
static int t_outer_set_value(Pyt_outer *self, PyObject *value, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete value");
        return -1;
    }

    int c_value;
    extern void __dta_ct_MOD_f90wrap_t_outer__set__value(void*, int*);

    c_value = (int)PyLong_AsLong(value);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert value");
        return -1;
    }

    __dta_ct_MOD_f90wrap_t_outer__set__value(self->fortran_ptr, &c_value);
    return 0;
}

/* Getter for t_outer.inner */
static PyObject* t_outer_get_inner(Pyt_outer *self, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    /* Nested derived type element getter for inner */
    /* Returns instance of t_inner */
    extern void __dta_ct_MOD_f90wrap_t_outer__get__inner(void*, void*);

    /* TODO: Create t_inner instance and transfer pointer */
    /* This requires accessing the t_innerType object */
    Py_RETURN_NONE;
}

/* Setter for t_outer.inner */
static int t_outer_set_inner(Pyt_outer *self, PyObject *value, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete inner");
        return -1;
    }

    /* Nested derived type element setter for inner */
    /* Accepts t_inner instance */
    extern void __dta_ct_MOD_f90wrap_t_outer__set__inner(void*, void*);

    /* TODO: Validate t_inner instance and transfer pointer */
    /* This requires type checking against t_innerType */
    return 0;
}

/* GetSet table for t_outer */
static PyGetSetDef t_outer_getsetters[] = {
    {"value", (getter)t_outer_get_value, (setter)t_outer_set_value, "value", NULL},
    {"inner", (getter)t_outer_get_inner, (setter)t_outer_set_inner, "inner", NULL},
    {NULL}  /* Sentinel */
};

/* Type-bound method: t_outer.new_outer */
static PyObject* t_outer_new_outer(Pyt_outer *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    PyObject *py_value = NULL;
    PyObject *py_inner = NULL;
    int value;
    void* inner;

    if (!PyArg_ParseTuple(args, "iO", &py_value, &py_inner)) {
        return NULL;
    }

    value = (int)PyLong_AsLong(py_value);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument value");
        return NULL;
    }
    /* Unwrap PyCapsule for derived type t_inner */
    inner = f90wrap_unwrap_capsule(py_inner, "t_inner");
    if (inner == NULL) {
        return NULL;
    }


    /* Call Fortran function */
    extern void* __dta_ct_MOD_new_outer(void*, int*, void**);
    void* result;
    result = __dta_ct_MOD_new_outer(self->fortran_ptr, &value, &inner);

    /* Return derived type t_outer as PyCapsule */
    if (result == NULL) {
        Py_RETURN_NONE;
    }

    return f90wrap_create_capsule(result, "t_outer_capsule", t_outer_capsule_destructor);
}

/* Type-bound method: t_outer.t_outer_finalise */
static PyObject* t_outer_t_outer_finalise(Pyt_outer *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    PyObject *py_this = NULL;
    void* this;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_outer */
    this = f90wrap_unwrap_capsule(py_this, "t_outer");
    if (this == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __dta_ct_MOD_t_outer_finalise(void*, void**);
    __dta_ct_MOD_t_outer_finalise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type t_outer as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "t_outer_capsule", t_outer_capsule_destructor);
}

/* Method table for t_outer */
static PyMethodDef t_outer_methods[] = {
    {"new_outer", (PyCFunction)t_outer_new_outer, METH_VARARGS, "Type-bound method new_outer"},
    {"t_outer_finalise", (PyCFunction)t_outer_t_outer_finalise, METH_VARARGS, "Type-bound method t_outer_finalise"},
    {NULL}  /* Sentinel */
};

/* Type object for t_outer */
static PyTypeObject t_outerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_issue258_derived_type_attributes_direct.t_outer",
    .tp_basicsize = sizeof(Pyt_outer),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)t_outer_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Fortran derived type t_outer",
    .tp_methods = t_outer_methods,
    .tp_getset = t_outer_getsetters,
    .tp_new = t_outer_new,
};


/* Fortran subroutine prototypes */

/* Python wrapper functions */


/* Wrapper for wrap_t_inner_create */
static char wrap_t_inner_create__doc__[] = "Create a new t_inner instance";

static PyObject* wrap_t_inner_create(PyObject *self, PyObject *args, PyObject *kwargs) {

    /* Allocate new t_inner instance */
    void* ptr = NULL;

    extern void __dta_ct_MOD_f90wrap_t_inner__allocate(void**);
    __dta_ct_MOD_f90wrap_t_inner__allocate(&ptr);

    if (ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate derived type");
        return NULL;
    }

    return f90wrap_create_capsule(ptr, "t_inner_capsule", t_inner_capsule_destructor);
}



/* Wrapper for wrap_t_inner_destroy */
static char wrap_t_inner_destroy__doc__[] = "Destroy a t_inner instance";

static PyObject* wrap_t_inner_destroy(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_capsule = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_capsule)) {
        return NULL;
    }

    void* ptr = f90wrap_unwrap_capsule(py_capsule, "t_inner");
    if (ptr == NULL) {
        return NULL; /* Exception already set by GetPointer */
    }

    /* Deallocate t_inner instance */
    extern void __dta_ct_MOD_f90wrap_t_inner__deallocate(void**);
    __dta_ct_MOD_f90wrap_t_inner__deallocate(&ptr);

    f90wrap_clear_capsule(py_capsule);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_t_outer_create */
static char wrap_t_outer_create__doc__[] = "Create a new t_outer instance";

static PyObject* wrap_t_outer_create(PyObject *self, PyObject *args, PyObject *kwargs) {

    /* Allocate new t_outer instance */
    void* ptr = NULL;

    extern void __dta_ct_MOD_f90wrap_t_outer__allocate(void**);
    __dta_ct_MOD_f90wrap_t_outer__allocate(&ptr);

    if (ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate derived type");
        return NULL;
    }

    return f90wrap_create_capsule(ptr, "t_outer_capsule", t_outer_capsule_destructor);
}



/* Wrapper for wrap_t_outer_destroy */
static char wrap_t_outer_destroy__doc__[] = "Destroy a t_outer instance";

static PyObject* wrap_t_outer_destroy(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_capsule = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_capsule)) {
        return NULL;
    }

    void* ptr = f90wrap_unwrap_capsule(py_capsule, "t_outer");
    if (ptr == NULL) {
        return NULL; /* Exception already set by GetPointer */
    }

    /* Deallocate t_outer instance */
    extern void __dta_ct_MOD_f90wrap_t_outer__deallocate(void**);
    __dta_ct_MOD_f90wrap_t_outer__deallocate(&ptr);

    f90wrap_clear_capsule(py_capsule);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_dta_ct__t_outer_print */
static char wrap_dta_ct__t_outer_print__doc__[] = "Wrapper for t_outer_print";

static PyObject* wrap_dta_ct__t_outer_print(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_outer = NULL;
    void* outer;

    if (!PyArg_ParseTuple(args, "O", &py_outer)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_outer */
    outer = f90wrap_unwrap_capsule(py_outer, "t_outer");
    if (outer == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __dta_ct_MOD_t_outer_print(void**);
    __dta_ct_MOD_t_outer_print(&outer);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_dta_ct__get_outer_inner */
static char wrap_dta_ct__get_outer_inner__doc__[] = "Wrapper for get_outer_inner";

static PyObject* wrap_dta_ct__get_outer_inner(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_outer = NULL;
    void* outer;

    if (!PyArg_ParseTuple(args, "O", &py_outer)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_outer */
    outer = f90wrap_unwrap_capsule(py_outer, "t_outer");
    if (outer == NULL) {
        return NULL;
    }


    /* Call Fortran function */
    extern void* __dta_ct_MOD_get_outer_inner(void**);
    void* result;
    result = __dta_ct_MOD_get_outer_inner(&outer);

    /* Return derived type t_inner as PyCapsule */
    if (result == NULL) {
        Py_RETURN_NONE;
    }

    return f90wrap_create_capsule(result, "t_inner_capsule", t_inner_capsule_destructor);
}



/* Wrapper for wrap_dta_cc__t_inner_print */
static char wrap_dta_cc__t_inner_print__doc__[] = "Wrapper for t_inner_print";

static PyObject* wrap_dta_cc__t_inner_print(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_inner = NULL;
    void* inner;

    if (!PyArg_ParseTuple(args, "O", &py_inner)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_inner */
    inner = f90wrap_unwrap_capsule(py_inner, "t_inner");
    if (inner == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __dta_cc_MOD_t_inner_print(void**);
    __dta_cc_MOD_t_inner_print(&inner);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_dta_cc__t_outer_print */
static char wrap_dta_cc__t_outer_print__doc__[] = "Wrapper for t_outer_print";

static PyObject* wrap_dta_cc__t_outer_print(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_outer = NULL;
    void* outer;

    if (!PyArg_ParseTuple(args, "O", &py_outer)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_outer */
    outer = f90wrap_unwrap_capsule(py_outer, "t_outer");
    if (outer == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __dta_cc_MOD_t_outer_print(void**);
    __dta_cc_MOD_t_outer_print(&outer);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_dta_cc__get_outer_inner */
static char wrap_dta_cc__get_outer_inner__doc__[] = "Wrapper for get_outer_inner";

static PyObject* wrap_dta_cc__get_outer_inner(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_outer = NULL;
    void* outer;

    if (!PyArg_ParseTuple(args, "O", &py_outer)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_outer */
    outer = f90wrap_unwrap_capsule(py_outer, "t_outer");
    if (outer == NULL) {
        return NULL;
    }


    /* Call Fortran function */
    extern void* __dta_cc_MOD_get_outer_inner(void**);
    void* result;
    result = __dta_cc_MOD_get_outer_inner(&outer);

    /* Return derived type t_inner as PyCapsule */
    if (result == NULL) {
        Py_RETURN_NONE;
    }

    return f90wrap_create_capsule(result, "t_inner_capsule", t_inner_capsule_destructor);
}



/* Wrapper for wrap_dta_tt__get_outer_inner */
static char wrap_dta_tt__get_outer_inner__doc__[] = "Wrapper for get_outer_inner";

static PyObject* wrap_dta_tt__get_outer_inner(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_outer = NULL;
    void* outer;

    if (!PyArg_ParseTuple(args, "O", &py_outer)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_outer */
    outer = f90wrap_unwrap_capsule(py_outer, "t_outer");
    if (outer == NULL) {
        return NULL;
    }


    /* Call Fortran function */
    extern void* __dta_tt_MOD_get_outer_inner(void**);
    void* result;
    result = __dta_tt_MOD_get_outer_inner(&outer);

    /* Return derived type t_inner as PyCapsule */
    if (result == NULL) {
        Py_RETURN_NONE;
    }

    return f90wrap_create_capsule(result, "t_inner_capsule", t_inner_capsule_destructor);
}



/* Wrapper for wrap_dta_tc__t_inner_print */
static char wrap_dta_tc__t_inner_print__doc__[] = "Wrapper for t_inner_print";

static PyObject* wrap_dta_tc__t_inner_print(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_inner = NULL;
    void* inner;

    if (!PyArg_ParseTuple(args, "O", &py_inner)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_inner */
    inner = f90wrap_unwrap_capsule(py_inner, "t_inner");
    if (inner == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __dta_tc_MOD_t_inner_print(void**);
    __dta_tc_MOD_t_inner_print(&inner);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_dta_tc__get_outer_inner */
static char wrap_dta_tc__get_outer_inner__doc__[] = "Wrapper for get_outer_inner";

static PyObject* wrap_dta_tc__get_outer_inner(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_outer = NULL;
    void* outer;

    if (!PyArg_ParseTuple(args, "O", &py_outer)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_outer */
    outer = f90wrap_unwrap_capsule(py_outer, "t_outer");
    if (outer == NULL) {
        return NULL;
    }


    /* Call Fortran function */
    extern void* __dta_tc_MOD_get_outer_inner(void**);
    void* result;
    result = __dta_tc_MOD_get_outer_inner(&outer);

    /* Return derived type t_inner as PyCapsule */
    if (result == NULL) {
        Py_RETURN_NONE;
    }

    return f90wrap_create_capsule(result, "t_inner_capsule", t_inner_capsule_destructor);
}



/* Method table */
static PyMethodDef _issue258_derived_type_attributes_direct_methods[] = {
    {"wrap_t_inner_create", (PyCFunction)wrap_t_inner_create, METH_VARARGS, wrap_t_inner_create__doc__},
    {"wrap_t_inner_destroy", (PyCFunction)wrap_t_inner_destroy, METH_VARARGS, wrap_t_inner_destroy__doc__},
    {"wrap_t_outer_create", (PyCFunction)wrap_t_outer_create, METH_VARARGS, wrap_t_outer_create__doc__},
    {"wrap_t_outer_destroy", (PyCFunction)wrap_t_outer_destroy, METH_VARARGS, wrap_t_outer_destroy__doc__},
    {"wrap_dta_ct__t_outer_print", (PyCFunction)wrap_dta_ct__t_outer_print, METH_VARARGS, wrap_dta_ct__t_outer_print__doc__},
    {"wrap_dta_ct__get_outer_inner", (PyCFunction)wrap_dta_ct__get_outer_inner, METH_VARARGS, wrap_dta_ct__get_outer_inner__doc__},
    {"wrap_dta_cc__t_inner_print", (PyCFunction)wrap_dta_cc__t_inner_print, METH_VARARGS, wrap_dta_cc__t_inner_print__doc__},
    {"wrap_dta_cc__t_outer_print", (PyCFunction)wrap_dta_cc__t_outer_print, METH_VARARGS, wrap_dta_cc__t_outer_print__doc__},
    {"wrap_dta_cc__get_outer_inner", (PyCFunction)wrap_dta_cc__get_outer_inner, METH_VARARGS, wrap_dta_cc__get_outer_inner__doc__},
    {"wrap_dta_tt__get_outer_inner", (PyCFunction)wrap_dta_tt__get_outer_inner, METH_VARARGS, wrap_dta_tt__get_outer_inner__doc__},
    {"wrap_dta_tc__t_inner_print", (PyCFunction)wrap_dta_tc__t_inner_print, METH_VARARGS, wrap_dta_tc__t_inner_print__doc__},
    {"wrap_dta_tc__get_outer_inner", (PyCFunction)wrap_dta_tc__get_outer_inner, METH_VARARGS, wrap_dta_tc__get_outer_inner__doc__},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module definition */
static struct PyModuleDef _issue258_derived_type_attributes_direct_module = {
    PyModuleDef_HEAD_INIT,
    "_issue258_derived_type_attributes_direct",
    "Fortran module _issue258_derived_type_attributes_direct wrapped with f90wrap",
    -1,
    _issue258_derived_type_attributes_direct_methods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit__issue258_derived_type_attributes_direct(void) {
    PyObject *module;

    /* Import NumPy C API */
    import_array();

    /* Initialize t_inner type */
    if (PyType_Ready(&t_innerType) < 0) {
        return NULL;
    }

    /* Initialize t_outer type */
    if (PyType_Ready(&t_outerType) < 0) {
        return NULL;
    }

    /* Create module */
    module = PyModule_Create(&_issue258_derived_type_attributes_direct_module);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&t_innerType);
    if (PyModule_AddObject(module, "t_inner", (PyObject *)&t_innerType) < 0) {
        Py_DECREF(&t_innerType);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&t_outerType);
    if (PyModule_AddObject(module, "t_outer", (PyObject *)&t_outerType) < 0) {
        Py_DECREF(&t_outerType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
