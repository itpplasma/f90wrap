/* C Extension module for _docstring_direct */

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

/* Define capsule destructor for t_circle */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(t_circle)
/* Define capsule destructor for c_ptr (Fortran intrinsic type) */
F90WRAP_DEFINE_SIMPLE_DESTRUCTOR(c_ptr)

/* Fortran derived type: t_circle */
typedef struct {
    PyObject_HEAD
    void* fortran_ptr;  /* Opaque pointer to Fortran type instance */
    int owns_memory;     /* 1 if we own the Fortran memory */
} Pyt_circle;

/* Forward declarations for t_circle methods */
static PyObject* t_circle_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void t_circle_dealloc(Pyt_circle *self);

/* Constructor for t_circle */
static PyObject* t_circle_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Pyt_circle *self;

    self = (Pyt_circle *)type->tp_alloc(type, 0);
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

/* Destructor for t_circle */
static void t_circle_dealloc(Pyt_circle *self) {
    if (self->fortran_ptr != NULL && self->owns_memory) {
        free(self->fortran_ptr);
        self->fortran_ptr = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Getter for t_circle.radius */
static PyObject* t_circle_get_radius(Pyt_circle *self, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    float value;
    extern void __m_circle_MOD_f90wrap_t_circle__get__radius(void*, float*);

    __m_circle_MOD_f90wrap_t_circle__get__radius(self->fortran_ptr, &value);
    return PyFloat_FromDouble(value);
}

/* Setter for t_circle.radius */
static int t_circle_set_radius(Pyt_circle *self, PyObject *value, void *closure) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return -1;
    }

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete radius");
        return -1;
    }

    float c_value;
    extern void __m_circle_MOD_f90wrap_t_circle__set__radius(void*, float*);

    c_value = (float)PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert radius");
        return -1;
    }

    __m_circle_MOD_f90wrap_t_circle__set__radius(self->fortran_ptr, &c_value);
    return 0;
}

/* GetSet table for t_circle */
static PyGetSetDef t_circle_getsetters[] = {
    {"radius", (getter)t_circle_get_radius, (setter)t_circle_set_radius, "radius", NULL},
    {NULL}  /* Sentinel */
};

/* Type-bound method: t_circle.t_circle_initialise */
static PyObject* t_circle_t_circle_initialise(Pyt_circle *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    void* this;

    this = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_t_circle_initialise(void*, void**);
    __m_circle_MOD_t_circle_initialise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type t_circle as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "t_circle_capsule", t_circle_capsule_destructor);
}

/* Type-bound method: t_circle.t_circle_finalise */
static PyObject* t_circle_t_circle_finalise(Pyt_circle *self, PyObject *args) {
    if (self->fortran_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Fortran type not initialized");
        return NULL;
    }

    PyObject *py_this = NULL;
    void* this;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    this = f90wrap_unwrap_capsule(py_this, "t_circle");
    if (this == NULL) {
        return NULL;
    }


    /* Call Fortran subroutine */
    extern void __m_circle_MOD_t_circle_finalise(void*, void**);
    __m_circle_MOD_t_circle_finalise(self->fortran_ptr, &this);

    /* Build return tuple for output arguments */
    /* Return derived type t_circle as PyCapsule */
    if (this == NULL) {
        Py_RETURN_NONE;
    }
    return f90wrap_create_capsule(this, "t_circle_capsule", t_circle_capsule_destructor);
}

/* Method table for t_circle */
static PyMethodDef t_circle_methods[] = {
    {"t_circle_initialise", (PyCFunction)t_circle_t_circle_initialise, METH_VARARGS, "Type-bound method t_circle_initialise"},
    {"t_circle_finalise", (PyCFunction)t_circle_t_circle_finalise, METH_VARARGS, "Type-bound method t_circle_finalise"},
    {NULL}  /* Sentinel */
};

/* Type object for t_circle */
static PyTypeObject t_circleType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_docstring_direct.t_circle",
    .tp_basicsize = sizeof(Pyt_circle),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)t_circle_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Fortran derived type t_circle",
    .tp_methods = t_circle_methods,
    .tp_getset = t_circle_getsetters,
    .tp_new = t_circle_new,
};


/* Fortran subroutine prototypes */

/* Python wrapper functions */


/* Wrapper for wrap_t_circle_create */
static char wrap_t_circle_create__doc__[] = "Create a new t_circle instance";

static PyObject* wrap_t_circle_create(PyObject *self, PyObject *args, PyObject *kwargs) {

    /* Allocate new t_circle instance */
    void* ptr = NULL;

    extern void __m_circle_MOD_f90wrap_t_circle__allocate(void**);
    __m_circle_MOD_f90wrap_t_circle__allocate(&ptr);

    if (ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate derived type");
        return NULL;
    }

    return f90wrap_create_capsule(ptr, "t_circle_capsule", t_circle_capsule_destructor);
}



/* Wrapper for wrap_t_circle_destroy */
static char wrap_t_circle_destroy__doc__[] = "Destroy a t_circle instance";

static PyObject* wrap_t_circle_destroy(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_capsule = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_capsule)) {
        return NULL;
    }

    void* ptr = f90wrap_unwrap_capsule(py_capsule, "t_circle");
    if (ptr == NULL) {
        return NULL; /* Exception already set by GetPointer */
    }

    /* Deallocate t_circle instance */
    extern void __m_circle_MOD_f90wrap_t_circle__deallocate(void**);
    __m_circle_MOD_f90wrap_t_circle__deallocate(&ptr);

    f90wrap_clear_capsule(py_capsule);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__details_doc */
static char wrap_m_circle__details_doc__doc__[] = "Initialize circle\n Those are very informative details\n";

static PyObject* wrap_m_circle__details_doc(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_details_doc(void**, float*);
    __m_circle_MOD_details_doc(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__details_with_parenthesis */
static char wrap_m_circle__details_with_parenthesis__doc__[] = "Initialize circle\n Those are very informative details (with parenthesis)\n";

static PyObject* wrap_m_circle__details_with_parenthesis(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_details_with_parenthesis(void**, float*);
    __m_circle_MOD_details_with_parenthesis(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__output_1 */
static char wrap_m_circle__output_1__doc__[] = "subroutine output_1 outputs 1\n";

static PyObject* wrap_m_circle__output_1(PyObject *self, PyObject *args, PyObject *kwargs) {

    float output;

    output = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_output_1(float*);
    __m_circle_MOD_output_1(&output);

    /* Build return tuple for output arguments */
    return PyFloat_FromDouble(output);
}



/* Wrapper for wrap_m_circle__long_line_brief */
static char wrap_m_circle__long_line_brief__doc__[] = "This is a very long brief that takes up a lot of space and contains lots of information, it should probably be wrapped to the next line, but we will continue regardless\n Those are very informative details\n";

static PyObject* wrap_m_circle__long_line_brief(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_long_line_brief(void**, float*);
    __m_circle_MOD_long_line_brief(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__function_2 */
static char wrap_m_circle__function_2__doc__[] = "this is a function\n";

static PyObject* wrap_m_circle__function_2(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_input = NULL;
    char* input;

    if (!PyArg_ParseTuple(args, "s", &py_input)) {
        return NULL;
    }

    input = (char*)PyUnicode_AsUTF8(py_input);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument input");
        return NULL;
    }

    /* Call Fortran function */
    extern int __m_circle_MOD_function_2(char**);
    int result;
    result = __m_circle_MOD_function_2(&input);

    return PyLong_FromLong(result);
}



/* Wrapper for wrap_m_circle__multiline_details */
static char wrap_m_circle__multiline_details__doc__[] = "Initialize circle\n First details line\nSecond details line\n";

static PyObject* wrap_m_circle__multiline_details(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_multiline_details(void**, float*);
    __m_circle_MOD_multiline_details(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__construct_circle */
static char wrap_m_circle__construct_circle__doc__[] = "Initialize circle\n";

static PyObject* wrap_m_circle__construct_circle(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_construct_circle(void**, float*);
    __m_circle_MOD_construct_circle(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__empty_lines_details */
static char wrap_m_circle__empty_lines_details__doc__[] = "Initialize circle\n First details line\n\nSecond details line after a empty line\n";

static PyObject* wrap_m_circle__empty_lines_details(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_empty_lines_details(void**, float*);
    __m_circle_MOD_empty_lines_details(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__doc_inside */
static char wrap_m_circle__doc_inside__doc__[] = "=========================================================================== >  \brief Doc inside  \param[in,out] circle      t_circle to initialize  \param[in]     radius      radius of the circle <";

static PyObject* wrap_m_circle__doc_inside(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_doc_inside(void**, float*);
    __m_circle_MOD_doc_inside(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__no_direction */
static char wrap_m_circle__no_direction__doc__[] = "Without direction\n";

static PyObject* wrap_m_circle__no_direction(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_no_direction(void**, float*);
    __m_circle_MOD_no_direction(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__construct_circle_more_doc */
static char wrap_m_circle__construct_circle_more_doc__doc__[] = "Initialize circle with more doc\n Author: test_author Copyright: test_copyright";

static PyObject* wrap_m_circle__construct_circle_more_doc(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_construct_circle_more_doc(void**, float*);
    __m_circle_MOD_construct_circle_more_doc(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_m_circle__incomplete_doc_sub */
static char wrap_m_circle__incomplete_doc_sub__doc__[] = "Incomplete doc\n";

static PyObject* wrap_m_circle__incomplete_doc_sub(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_circle = NULL;
    PyObject *py_radius = NULL;
    void* circle;
    float radius;

    if (!PyArg_ParseTuple(args, "Of", &py_circle, &py_radius)) {
        return NULL;
    }

    /* Unwrap PyCapsule for derived type t_circle */
    circle = f90wrap_unwrap_capsule(py_circle, "t_circle");
    if (circle == NULL) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }

    /* Call Fortran subroutine */
    extern void __m_circle_MOD_incomplete_doc_sub(void**, float*);
    __m_circle_MOD_incomplete_doc_sub(&circle, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_t_circle__get__radius */
static char wrap_f90wrap_t_circle__get__radius__doc__[] = "Wrapper for f90wrap_t_circle__get__radius";

static PyObject* wrap_f90wrap_t_circle__get__radius(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_this = NULL;
    float f90wrap_radius;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Extract Fortran array from NumPy py_this */
    if (!PyArray_Check(py_this)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for this");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_this) != 1) {
        PyErr_Format(PyExc_ValueError, "Array this must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_this));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_this) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array this has wrong dtype");
        return NULL;
    }

    PyArrayObject *this_data_array = (PyArrayObject*)py_this;
    if (!PyArray_IS_F_CONTIGUOUS(this_data_array)) {
        this_data_array = (PyArrayObject*)PyArray_FromArray(
            this_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (this_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* this_data = (int*)PyArray_DATA(this_data_array);

    f90wrap_radius = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void f90wrap_t_circle__get__radius_(void*, float*);
    f90wrap_t_circle__get__radius_(this_data, &f90wrap_radius);

    /* Build return tuple for output arguments */
    return PyFloat_FromDouble(f90wrap_radius);
}



/* Wrapper for wrap_f90wrap_t_circle__set__radius */
static char wrap_f90wrap_t_circle__set__radius__doc__[] = "Wrapper for f90wrap_t_circle__set__radius";

static PyObject* wrap_f90wrap_t_circle__set__radius(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_f90wrap_radius = NULL;
    PyObject *py_this = NULL;
    float f90wrap_radius;

    if (!PyArg_ParseTuple(args, "fO", &py_f90wrap_radius, &py_this)) {
        return NULL;
    }

    f90wrap_radius = (float)PyFloat_AsDouble(py_f90wrap_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument f90wrap_radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_this */
    if (!PyArray_Check(py_this)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for this");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_this) != 1) {
        PyErr_Format(PyExc_ValueError, "Array this must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_this));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_this) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array this has wrong dtype");
        return NULL;
    }

    PyArrayObject *this_data_array = (PyArrayObject*)py_this;
    if (!PyArray_IS_F_CONTIGUOUS(this_data_array)) {
        this_data_array = (PyArrayObject*)PyArray_FromArray(
            this_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (this_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* this_data = (int*)PyArray_DATA(this_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_t_circle__set__radius_(void*, float*);
    f90wrap_t_circle__set__radius_(this_data, &f90wrap_radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__t_circle_initialise */
static char wrap_f90wrap_m_circle__t_circle_initialise__doc__[] = "Wrapper for f90wrap_m_circle__t_circle_initialise";

static PyObject* wrap_f90wrap_m_circle__t_circle_initialise(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_this = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Extract Fortran array from NumPy py_this */
    if (!PyArray_Check(py_this)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for this");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_this) != 1) {
        PyErr_Format(PyExc_ValueError, "Array this must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_this));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_this) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array this has wrong dtype");
        return NULL;
    }

    PyArrayObject *this_data_array = (PyArrayObject*)py_this;
    if (!PyArray_IS_F_CONTIGUOUS(this_data_array)) {
        this_data_array = (PyArrayObject*)PyArray_FromArray(
            this_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (this_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* this_data = (int*)PyArray_DATA(this_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__t_circle_initialise_(void*);
    f90wrap_m_circle__t_circle_initialise_(this_data);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__t_circle_finalise */
static char wrap_f90wrap_m_circle__t_circle_finalise__doc__[] = "Wrapper for f90wrap_m_circle__t_circle_finalise";

static PyObject* wrap_f90wrap_m_circle__t_circle_finalise(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_this = NULL;

    if (!PyArg_ParseTuple(args, "O", &py_this)) {
        return NULL;
    }

    /* Extract Fortran array from NumPy py_this */
    if (!PyArray_Check(py_this)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for this");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_this) != 1) {
        PyErr_Format(PyExc_ValueError, "Array this must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_this));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_this) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array this has wrong dtype");
        return NULL;
    }

    PyArrayObject *this_data_array = (PyArrayObject*)py_this;
    if (!PyArray_IS_F_CONTIGUOUS(this_data_array)) {
        this_data_array = (PyArrayObject*)PyArray_FromArray(
            this_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (this_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* this_data = (int*)PyArray_DATA(this_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__t_circle_finalise_(void*);
    f90wrap_m_circle__t_circle_finalise_(this_data);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__output_1 */
static char wrap_f90wrap_m_circle__output_1__doc__[] = "Wrapper for f90wrap_m_circle__output_1";

static PyObject* wrap_f90wrap_m_circle__output_1(PyObject *self, PyObject *args, PyObject *kwargs) {

    float output;

    output = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__output_1_(float*);
    f90wrap_m_circle__output_1_(&output);

    /* Build return tuple for output arguments */
    return PyFloat_FromDouble(output);
}



/* Wrapper for wrap_f90wrap_m_circle__construct_circle_more_doc */
static char wrap_f90wrap_m_circle__construct_circle_more_doc__doc__[] = "Wrapper for f90wrap_m_circle__construct_circle_more_doc";

static PyObject* wrap_f90wrap_m_circle__construct_circle_more_doc(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__construct_circle_more_doc_(void*, float*);
    f90wrap_m_circle__construct_circle_more_doc_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__details_doc */
static char wrap_f90wrap_m_circle__details_doc__doc__[] = "Wrapper for f90wrap_m_circle__details_doc";

static PyObject* wrap_f90wrap_m_circle__details_doc(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__details_doc_(void*, float*);
    f90wrap_m_circle__details_doc_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__no_direction */
static char wrap_f90wrap_m_circle__no_direction__doc__[] = "Wrapper for f90wrap_m_circle__no_direction";

static PyObject* wrap_f90wrap_m_circle__no_direction(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__no_direction_(void*, float*);
    f90wrap_m_circle__no_direction_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__details_with_parenthesis */
static char wrap_f90wrap_m_circle__details_with_parenthesis__doc__[] = "Wrapper for f90wrap_m_circle__details_with_parenthesis";

static PyObject* wrap_f90wrap_m_circle__details_with_parenthesis(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__details_with_parenthesis_(void*, float*);
    f90wrap_m_circle__details_with_parenthesis_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__multiline_details */
static char wrap_f90wrap_m_circle__multiline_details__doc__[] = "Wrapper for f90wrap_m_circle__multiline_details";

static PyObject* wrap_f90wrap_m_circle__multiline_details(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__multiline_details_(void*, float*);
    f90wrap_m_circle__multiline_details_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__construct_circle */
static char wrap_f90wrap_m_circle__construct_circle__doc__[] = "Wrapper for f90wrap_m_circle__construct_circle";

static PyObject* wrap_f90wrap_m_circle__construct_circle(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__construct_circle_(void*, float*);
    f90wrap_m_circle__construct_circle_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__empty_lines_details */
static char wrap_f90wrap_m_circle__empty_lines_details__doc__[] = "Wrapper for f90wrap_m_circle__empty_lines_details";

static PyObject* wrap_f90wrap_m_circle__empty_lines_details(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__empty_lines_details_(void*, float*);
    f90wrap_m_circle__empty_lines_details_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__incomplete_doc_sub */
static char wrap_f90wrap_m_circle__incomplete_doc_sub__doc__[] = "Wrapper for f90wrap_m_circle__incomplete_doc_sub";

static PyObject* wrap_f90wrap_m_circle__incomplete_doc_sub(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__incomplete_doc_sub_(void*, float*);
    f90wrap_m_circle__incomplete_doc_sub_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__doc_inside */
static char wrap_f90wrap_m_circle__doc_inside__doc__[] = "Wrapper for f90wrap_m_circle__doc_inside";

static PyObject* wrap_f90wrap_m_circle__doc_inside(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__doc_inside_(void*, float*);
    f90wrap_m_circle__doc_inside_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Wrapper for wrap_f90wrap_m_circle__function_2 */
static char wrap_f90wrap_m_circle__function_2__doc__[] = "Wrapper for f90wrap_m_circle__function_2";

static PyObject* wrap_f90wrap_m_circle__function_2(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_input = NULL;
    int input;
    int ret_function_2;

    if (!PyArg_ParseTuple(args, "i", &py_input)) {
        return NULL;
    }

    input = (int)PyLong_AsLong(py_input);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument input");
        return NULL;
    }
    ret_function_2 = 0;  /* Initialize output argument */

    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__function_2_(int*, int*);
    f90wrap_m_circle__function_2_(&input, &ret_function_2);

    /* Build return tuple for output arguments */
    return PyLong_FromLong(ret_function_2);
}



/* Wrapper for wrap_f90wrap_m_circle__long_line_brief */
static char wrap_f90wrap_m_circle__long_line_brief__doc__[] = "Wrapper for f90wrap_m_circle__long_line_brief";

static PyObject* wrap_f90wrap_m_circle__long_line_brief(PyObject *self, PyObject *args, PyObject *kwargs) {

    PyObject *py_radius = NULL;
    PyObject *py_circle = NULL;
    float radius;

    if (!PyArg_ParseTuple(args, "fO", &py_radius, &py_circle)) {
        return NULL;
    }

    radius = (float)PyFloat_AsDouble(py_radius);
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert argument radius");
        return NULL;
    }
    /* Extract Fortran array from NumPy py_circle */
    if (!PyArray_Check(py_circle)) {
        PyErr_SetString(PyExc_TypeError, "Expected NumPy array for circle");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject*)py_circle) != 1) {
        PyErr_Format(PyExc_ValueError, "Array circle must have 1 dimensions, got %d",
                     PyArray_NDIM((PyArrayObject*)py_circle));
        return NULL;
    }

    if (PyArray_TYPE((PyArrayObject*)py_circle) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "Array circle has wrong dtype");
        return NULL;
    }

    PyArrayObject *circle_data_array = (PyArrayObject*)py_circle;
    if (!PyArray_IS_F_CONTIGUOUS(circle_data_array)) {
        circle_data_array = (PyArrayObject*)PyArray_FromArray(
            circle_data_array, NULL, NPY_ARRAY_F_CONTIGUOUS);
        if (circle_data_array == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert array to Fortran order");
            return NULL;
        }
    }

    int* circle_data = (int*)PyArray_DATA(circle_data_array);


    /* Call Fortran subroutine */
    extern void f90wrap_m_circle__long_line_brief_(void*, float*);
    f90wrap_m_circle__long_line_brief_(circle_data, &radius);

    Py_RETURN_NONE;
}



/* Method table */
static PyMethodDef _docstring_direct_methods[] = {
    {"wrap_t_circle_create", (PyCFunction)wrap_t_circle_create, METH_VARARGS, wrap_t_circle_create__doc__},
    {"wrap_t_circle_destroy", (PyCFunction)wrap_t_circle_destroy, METH_VARARGS, wrap_t_circle_destroy__doc__},
    {"wrap_m_circle__details_doc", (PyCFunction)wrap_m_circle__details_doc, METH_VARARGS, wrap_m_circle__details_doc__doc__},
    {"wrap_m_circle__details_with_parenthesis", (PyCFunction)wrap_m_circle__details_with_parenthesis, METH_VARARGS, wrap_m_circle__details_with_parenthesis__doc__},
    {"wrap_m_circle__output_1", (PyCFunction)wrap_m_circle__output_1, METH_VARARGS, wrap_m_circle__output_1__doc__},
    {"wrap_m_circle__long_line_brief", (PyCFunction)wrap_m_circle__long_line_brief, METH_VARARGS, wrap_m_circle__long_line_brief__doc__},
    {"wrap_m_circle__function_2", (PyCFunction)wrap_m_circle__function_2, METH_VARARGS, wrap_m_circle__function_2__doc__},
    {"wrap_m_circle__multiline_details", (PyCFunction)wrap_m_circle__multiline_details, METH_VARARGS, wrap_m_circle__multiline_details__doc__},
    {"wrap_m_circle__construct_circle", (PyCFunction)wrap_m_circle__construct_circle, METH_VARARGS, wrap_m_circle__construct_circle__doc__},
    {"wrap_m_circle__empty_lines_details", (PyCFunction)wrap_m_circle__empty_lines_details, METH_VARARGS, wrap_m_circle__empty_lines_details__doc__},
    {"wrap_m_circle__doc_inside", (PyCFunction)wrap_m_circle__doc_inside, METH_VARARGS, wrap_m_circle__doc_inside__doc__},
    {"wrap_m_circle__no_direction", (PyCFunction)wrap_m_circle__no_direction, METH_VARARGS, wrap_m_circle__no_direction__doc__},
    {"wrap_m_circle__construct_circle_more_doc", (PyCFunction)wrap_m_circle__construct_circle_more_doc, METH_VARARGS, wrap_m_circle__construct_circle_more_doc__doc__},
    {"wrap_m_circle__incomplete_doc_sub", (PyCFunction)wrap_m_circle__incomplete_doc_sub, METH_VARARGS, wrap_m_circle__incomplete_doc_sub__doc__},
    {"wrap_f90wrap_t_circle__get__radius", (PyCFunction)wrap_f90wrap_t_circle__get__radius, METH_VARARGS, wrap_f90wrap_t_circle__get__radius__doc__},
    {"wrap_f90wrap_t_circle__set__radius", (PyCFunction)wrap_f90wrap_t_circle__set__radius, METH_VARARGS, wrap_f90wrap_t_circle__set__radius__doc__},
    {"wrap_f90wrap_m_circle__t_circle_initialise", (PyCFunction)wrap_f90wrap_m_circle__t_circle_initialise, METH_VARARGS, wrap_f90wrap_m_circle__t_circle_initialise__doc__},
    {"wrap_f90wrap_m_circle__t_circle_finalise", (PyCFunction)wrap_f90wrap_m_circle__t_circle_finalise, METH_VARARGS, wrap_f90wrap_m_circle__t_circle_finalise__doc__},
    {"wrap_f90wrap_m_circle__output_1", (PyCFunction)wrap_f90wrap_m_circle__output_1, METH_VARARGS, wrap_f90wrap_m_circle__output_1__doc__},
    {"wrap_f90wrap_m_circle__construct_circle_more_doc", (PyCFunction)wrap_f90wrap_m_circle__construct_circle_more_doc, METH_VARARGS, wrap_f90wrap_m_circle__construct_circle_more_doc__doc__},
    {"wrap_f90wrap_m_circle__details_doc", (PyCFunction)wrap_f90wrap_m_circle__details_doc, METH_VARARGS, wrap_f90wrap_m_circle__details_doc__doc__},
    {"wrap_f90wrap_m_circle__no_direction", (PyCFunction)wrap_f90wrap_m_circle__no_direction, METH_VARARGS, wrap_f90wrap_m_circle__no_direction__doc__},
    {"wrap_f90wrap_m_circle__details_with_parenthesis", (PyCFunction)wrap_f90wrap_m_circle__details_with_parenthesis, METH_VARARGS, wrap_f90wrap_m_circle__details_with_parenthesis__doc__},
    {"wrap_f90wrap_m_circle__multiline_details", (PyCFunction)wrap_f90wrap_m_circle__multiline_details, METH_VARARGS, wrap_f90wrap_m_circle__multiline_details__doc__},
    {"wrap_f90wrap_m_circle__construct_circle", (PyCFunction)wrap_f90wrap_m_circle__construct_circle, METH_VARARGS, wrap_f90wrap_m_circle__construct_circle__doc__},
    {"wrap_f90wrap_m_circle__empty_lines_details", (PyCFunction)wrap_f90wrap_m_circle__empty_lines_details, METH_VARARGS, wrap_f90wrap_m_circle__empty_lines_details__doc__},
    {"wrap_f90wrap_m_circle__incomplete_doc_sub", (PyCFunction)wrap_f90wrap_m_circle__incomplete_doc_sub, METH_VARARGS, wrap_f90wrap_m_circle__incomplete_doc_sub__doc__},
    {"wrap_f90wrap_m_circle__doc_inside", (PyCFunction)wrap_f90wrap_m_circle__doc_inside, METH_VARARGS, wrap_f90wrap_m_circle__doc_inside__doc__},
    {"wrap_f90wrap_m_circle__function_2", (PyCFunction)wrap_f90wrap_m_circle__function_2, METH_VARARGS, wrap_f90wrap_m_circle__function_2__doc__},
    {"wrap_f90wrap_m_circle__long_line_brief", (PyCFunction)wrap_f90wrap_m_circle__long_line_brief, METH_VARARGS, wrap_f90wrap_m_circle__long_line_brief__doc__},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* Module definition */
static struct PyModuleDef _docstring_direct_module = {
    PyModuleDef_HEAD_INIT,
    "_docstring_direct",
    "Fortran module _docstring_direct wrapped with f90wrap",
    -1,
    _docstring_direct_methods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit__docstring_direct(void) {
    PyObject *module;

    /* Import NumPy C API */
    import_array();

    /* Initialize t_circle type */
    if (PyType_Ready(&t_circleType) < 0) {
        return NULL;
    }

    /* Create module */
    module = PyModule_Create(&_docstring_direct_module);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&t_circleType);
    if (PyModule_AddObject(module, "t_circle", (PyObject *)&t_circleType) < 0) {
        Py_DECREF(&t_circleType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
