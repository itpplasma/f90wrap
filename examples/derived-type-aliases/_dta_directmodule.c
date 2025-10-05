/* C Extension module for _dta_direct */

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

/* Define capsule destructor for mytype */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(mytype)
/* Define capsule destructor for othertype */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(othertype)
/* Define capsule destructor for c_ptr (Fortran intrinsic type) */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(c_ptr)

/* Fortran derived type: mytype */
typedef struct {
    PyObject_HEAD
    void* fortran_ptr;  /* Opaque pointer to Fortran type instance */
    int owns_memory;     /* 1 if we own the Fortran memory */
} Pymytype;

/* Forward declarations for mytype methods */
static PyObject* mytype_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void mytype_dealloc(Pymytype *self);

/* Constructor for mytype */
static PyObject* mytype_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Pymytype *self;

    self = (Pymytype *)type->tp_alloc(type, 0);
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

/* Destructor for mytype */
static void mytype_dealloc(Pymytype *self) {
    if (self->fortran_ptr != NULL && self->owns_memory) {
        free(self->fortran_ptr);
        self->fortran_ptr = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Getter for mytype.a */
static PyObject* mytype_get_a(Pymytype *self, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    int value;
    extern void __mytype_mod_MOD_f90wrap_mytype__get__a(void*, int*);

    __mytype_mod_MOD_f90wrap_mytype__get__a(self->fortran_ptr, &value);
    return PyLong_FromLong(value);
}

/* Setter for mytype.a */
static int mytype_set_a(Pymytype *self, PyObject *value, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete a");
        return -1;
    }

    int c_value;
    extern void __mytype_mod_MOD_f90wrap_mytype__set__a(void*, int*);

    c_value = (int)PyLong_AsLong(value);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert a");
        return -1;
    }

    __mytype_mod_MOD_f90wrap_mytype__set__a(self->fortran_ptr, &c_value);
    return 0;
}

/* GetSet table for mytype */
static PyGetSetDef mytype_getsetters[] = {
    {"a", (getter)mytype_get_a, (setter)mytype_set_a, "a", NULL},
    {NULL}  /* Sentinel */
};

/* Type-bound method: mytype.mytype_initialise */
static PyObject* mytype_mytype_initialise(Pymytype *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    void* this;

    this = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __mytype_mod_MOD_mytype_initialise(void*, void**);
    __mytype_mod_MOD_mytype_initialise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type mytype as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "mytype_capsule", mytype_capsule_destructor);
}

/* Type-bound method: mytype.mytype_finalise */
static PyObject* mytype_mytype_finalise(Pymytype *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    PyObject *py_this = NULL;
    void* this;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type mytype */
    this = f90wrap_unwrap_capsule(py_this, "mytype");
    if (this == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __mytype_mod_MOD_mytype_finalise(void*, void**);
    __mytype_mod_MOD_mytype_finalise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type mytype as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "mytype_capsule", mytype_capsule_destructor);
}

/* Method table for mytype */
static PyMethodDef mytype_methods[] = {
    {"mytype_initialise", (PyCFunction)mytype_mytype_initialise, METH_VARARGS, "Type-bound method mytype_initialise"},
    {"mytype_finalise", (PyCFunction)mytype_mytype_finalise, METH_VARARGS, "Type-bound method mytype_finalise"},
    {NULL}  /* Sentinel */
};

/* Type object for mytype */
static PyTypeObject mytypeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_dta_direct.mytype",
    .tp_basicsize = sizeof(Pymytype),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)mytype_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Fortran derived type mytype",
    .tp_methods = mytype_methods,
    .tp_getset = mytype_getsetters,
    .tp_new = mytype_new,
};


/* Fortran derived type: othertype */
typedef struct {
    PyObject_HEAD
    void* fortran_ptr;  /* Opaque pointer to Fortran type instance */
    int owns_memory;     /* 1 if we own the Fortran memory */
} Pyothertype;

/* Forward declarations for othertype methods */
static PyObject* othertype_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void othertype_dealloc(Pyothertype *self);

/* Constructor for othertype */
static PyObject* othertype_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Pyothertype *self;

    self = (Pyothertype *)type->tp_alloc(type, 0);
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

/* Destructor for othertype */
static void othertype_dealloc(Pyothertype *self) {
    if (self->fortran_ptr != NULL && self->owns_memory) {
        free(self->fortran_ptr);
        self->fortran_ptr = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Getter for othertype.a */
static PyObject* othertype_get_a(Pyothertype *self, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    int value;
    extern void __othertype_mod_MOD_f90wrap_othertype__get__a(void*, int*);

    __othertype_mod_MOD_f90wrap_othertype__get__a(self->fortran_ptr, &value);
    return PyLong_FromLong(value);
}

/* Setter for othertype.a */
static int othertype_set_a(Pyothertype *self, PyObject *value, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete a");
        return -1;
    }

    int c_value;
    extern void __othertype_mod_MOD_f90wrap_othertype__set__a(void*, int*);

    c_value = (int)PyLong_AsLong(value);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert a");
        return -1;
    }

    __othertype_mod_MOD_f90wrap_othertype__set__a(self->fortran_ptr, &c_value);
    return 0;
}

/* GetSet table for othertype */
static PyGetSetDef othertype_getsetters[] = {
    {"a", (getter)othertype_get_a, (setter)othertype_set_a, "a", NULL},
    {NULL}  /* Sentinel */
};

/* Type-bound method: othertype.othertype_initialise */
static PyObject* othertype_othertype_initialise(Pyothertype *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    void* this;

    this = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __othertype_mod_MOD_othertype_initialise(void*, void**);
    __othertype_mod_MOD_othertype_initialise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type othertype as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "othertype_capsule", othertype_capsule_destructor);
}

/* Type-bound method: othertype.othertype_finalise */
static PyObject* othertype_othertype_finalise(Pyothertype *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    PyObject *py_this = NULL;
    void* this;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type othertype */
    this = f90wrap_unwrap_capsule(py_this, "othertype");
    if (this == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __othertype_mod_MOD_othertype_finalise(void*, void**);
    __othertype_mod_MOD_othertype_finalise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type othertype as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "othertype_capsule", othertype_capsule_destructor);
}

/* Method table for othertype */
static PyMethodDef othertype_methods[] = {
    {"othertype_initialise", (PyCFunction)othertype_othertype_initialise, METH_VARARGS, "Type-bound method othertype_initialise"},
    {"othertype_finalise", (PyCFunction)othertype_othertype_finalise, METH_VARARGS, "Type-bound method othertype_finalise"},
    {NULL}  /* Sentinel */
};

/* Type object for othertype */
static PyTypeObject othertypeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_dta_direct.othertype",
    .tp_basicsize = sizeof(Pyothertype),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)othertype_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Fortran derived type othertype",
    .tp_methods = othertype_methods,
    .tp_getset = othertype_getsetters,
    .tp_new = othertype_new,
};


/* Fortran subroutine prototypes */

/* Python wrapper functions */


/* Wrapper for wrap_mytype_create */
static char wrap_mytype_create__doc__[] = "Create a new mytype instance";

static PyObject* wrap_mytype_create(PyObject *self, PyObject *args, PyObject *kwargs) {

    /* Allocate new mytype instance */
    void* ptr = NULL;

    extern void __mytype_mod_MOD_f90wrap_mytype__allocate(void**);
    __mytype_mod_MOD_f90wrap_mytype__allocate(&ptr);

    if (ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate derived type");
        return NULL;
    }

    return f90wrap_create_capsule(ptr, "mytype_capsule", mytype_capsule_destructor);
}



/* Wrapper for wrap_mytype_destroy */
static char wrap_mytype_destroy__doc__[] = "Destroy a mytype instance";

static PyObject* wrap_mytype_destroy(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_capsule = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_capsule)) {
        return NULL;
    }

    void* ptr = f90wrap_unwrap_capsule(py_capsule, "mytype");
    if (ptr == NULL) {
        return NULL; /* Exception already set by GetPointer */
    }

    /* Deallocate mytype instance */
    extern void __mytype_mod_MOD_f90wrap_mytype__deallocate(void**);
    __mytype_mod_MOD_f90wrap_mytype__deallocate(&ptr);

    f90wrap_clear_capsule(py_capsule);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_othertype_create */
static char wrap_othertype_create__doc__[] = "Create a new othertype instance";

static PyObject* wrap_othertype_create(PyObject *self, PyObject *args, PyObject *kwargs) {

    /* Allocate new othertype instance */
    void* ptr = NULL;

    extern void __othertype_mod_MOD_f90wrap_othertype__allocate(void**);
    __othertype_mod_MOD_f90wrap_othertype__allocate(&ptr);

    if (ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate derived type");
        return NULL;
    }

    return f90wrap_create_capsule(ptr, "othertype_capsule", othertype_capsule_destructor);
}



/* Wrapper for wrap_othertype_destroy */
static char wrap_othertype_destroy__doc__[] = "Destroy a othertype instance";

static PyObject* wrap_othertype_destroy(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_capsule = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_capsule)) {
        return NULL;
    }

    void* ptr = f90wrap_unwrap_capsule(py_capsule, "othertype");
    if (ptr == NULL) {
        return NULL; /* Exception already set by GetPointer */
    }

    /* Deallocate othertype instance */
    extern void __othertype_mod_MOD_f90wrap_othertype__deallocate(void**);
    __othertype_mod_MOD_f90wrap_othertype__deallocate(&ptr);

    f90wrap_clear_capsule(py_capsule);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_mytype_mod__plus_b */
static char wrap_mytype_mod__plus_b__doc__[] = "Wrapper for plus_b";

static PyObject* wrap_mytype_mod__plus_b(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_obj = NULL;
    PyObject *py_b = NULL;
    void* obj;
    int b;
    int c;

    if (!PyArg_ParseTuple(args, "Oi", &py_obj, &py_b)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type mytype */
    obj = f90wrap_unwrap_capsule(py_obj, "mytype");
    if (obj == NULL) {
        return NULL;
    }

    b = (int)PyLong_AsLong(py_b);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument b");
        return NULL;
    }
    c = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __mytype_mod_MOD_plus_b(void**, int*, int*);
    __mytype_mod_MOD_plus_b(&obj, &b, &c);

    /* Build return tuple for output arguments */
    return PyLong_FromLong(c);
}



/* Wrapper for wrap_mytype_mod__constructor */
static char wrap_mytype_mod__constructor__doc__[] = "Wrapper for constructor";

static PyObject* wrap_mytype_mod__constructor(PyObject *self, PyObject *args, PyObject *kwargs) {



    /* Call Fortran function */
    extern void* __mytype_mod_MOD_constructor(void);
    void* result;
    result = __mytype_mod_MOD_constructor();

    /* Return derived type mytype as PyCapsule */
    if (result == NULL) {
        Py_RETURN_NONE;
    }

    return f90wrap_create_capsule(result, "mytype_capsule", mytype_capsule_destructor);
}



/* Wrapper for wrap_othertype_mod__plus_b */
static char wrap_othertype_mod__plus_b__doc__[] = "Wrapper for plus_b";

static PyObject* wrap_othertype_mod__plus_b(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_obj = NULL;
    PyObject *py_b = NULL;
    void* obj;
    int b;
    int c;

    if (!PyArg_ParseTuple(args, "Oi", &py_obj, &py_b)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type othertype */
    obj = f90wrap_unwrap_capsule(py_obj, "othertype");
    if (obj == NULL) {
        return NULL;
    }

    b = (int)PyLong_AsLong(py_b);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument b");
        return NULL;
    }
    c = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __othertype_mod_MOD_plus_b(void**, int*, int*);
    __othertype_mod_MOD_plus_b(&obj, &b, &c);

    /* Build return tuple for output arguments */
    return PyLong_FromLong(c);
}



/* Wrapper for wrap_othertype_mod__constructor */
static char wrap_othertype_mod__constructor__doc__[] = "Wrapper for constructor";

static PyObject* wrap_othertype_mod__constructor(PyObject *self, PyObject *args, PyObject *kwargs) {



    /* Call Fortran function */
    extern void* __othertype_mod_MOD_constructor(void);
    void* result;
    result = __othertype_mod_MOD_constructor();

    /* Return derived type othertype as PyCapsule */
    if (result == NULL) {
        Py_RETURN_NONE;
    }

    return f90wrap_create_capsule(result, "othertype_capsule", othertype_capsule_destructor);
}



/* Method table */
static PyMethodDef _dta_direct_methods[] = {
    {"wrap_mytype_create", (PyCFunction)wrap_mytype_create, METH_VARARGS, wrap_mytype_create__doc__},
    {"wrap_mytype_destroy", (PyCFunction)wrap_mytype_destroy, METH_VARARGS, wrap_mytype_destroy__doc__},
    {"wrap_othertype_create", (PyCFunction)wrap_othertype_create, METH_VARARGS, wrap_othertype_create__doc__},
    {"wrap_othertype_destroy", (PyCFunction)wrap_othertype_destroy, METH_VARARGS, wrap_othertype_destroy__doc__},
    {"wrap_mytype_mod__plus_b", (PyCFunction)wrap_mytype_mod__plus_b, METH_VARARGS, wrap_mytype_mod__plus_b__doc__},
    {"wrap_mytype_mod__constructor", (PyCFunction)wrap_mytype_mod__constructor, METH_VARARGS, wrap_mytype_mod__constructor__doc__},
    {"wrap_othertype_mod__plus_b", (PyCFunction)wrap_othertype_mod__plus_b, METH_VARARGS, wrap_othertype_mod__plus_b__doc__},
    {"wrap_othertype_mod__constructor", (PyCFunction)wrap_othertype_mod__constructor, METH_VARARGS, wrap_othertype_mod__constructor__doc__},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module definition */
static struct PyModuleDef _dta_direct_module = {
    PyModuleDef_HEAD_INIT,
    "_dta_direct",
    "Fortran module _dta_direct wrapped with f90wrap",
    -1,
    _dta_direct_methods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit__dta_direct(void) {
    PyObject *module;

    /* Import NumPy C API */
    import_array();

    /* Initialize mytype type */
    if (PyType_Ready(&mytypeType) < 0) {
        return NULL;
    }

    /* Initialize othertype type */
    if (PyType_Ready(&othertypeType) < 0) {
        return NULL;
    }

    /* Create module */
    module = PyModule_Create(&_dta_direct_module);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&mytypeType);
    if (PyModule_AddObject(module, "mytype", (PyObject *)&mytypeType) < 0) {
        Py_DECREF(&mytypeType);
        Py_DECREF(module);
        return NULL;
    }

    Py_INCREF(&othertypeType);
    if (PyModule_AddObject(module, "othertype", (PyObject *)&othertypeType) < 0) {
        Py_DECREF(&othertypeType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
