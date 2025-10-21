#include <Python.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define F90WRAP_F_SYMBOL(name) name##_

void f90wrap_abort_(char *message, int len_message)
{
    /* Acquire GIL since we're calling Python C-API from Fortran */
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    if (message == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "f90wrap_abort called");
        PyGILState_Release(gstate);
        return;
    }
    while (len_message > 0 && message[len_message - 1] == ' ') {
        --len_message;
    }
    if (len_message <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "f90wrap_abort called");
        PyGILState_Release(gstate);
        return;
    }
    PyObject* unicode = PyUnicode_FromStringAndSize(message, len_message);
    if (unicode == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "f90wrap_abort called");
        PyGILState_Release(gstate);
        return;
    }
    PyErr_SetObject(PyExc_RuntimeError, unicode);
    Py_DECREF(unicode);
    PyGILState_Release(gstate);
}

void f90wrap_abort__(char *message, int len_message)
{
    f90wrap_abort_(message, len_message);
}

/* External f90wrap helper functions */
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_routine4)(int* f90wrap_n0, float* x1, int* x2, int* x3, int* x4, \
    int* x5, float* x6);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__get__m)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__get__n)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2__array__y)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array__array_getitem__items)(int* dummy_this, int* \
    index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array__array_setitem__items)(int* dummy_this, int* \
    index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array__array_len__items)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array__array_getitem__items)(int* dummy_this, int* \
    index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array__array_setitem__items)(int* dummy_this, int* \
    index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array__array_len__items)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array__array_getitem__items)(int* dummy_this, int* \
    index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array__array_setitem__items)(int* dummy_this, int* \
    index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array__array_len__items)(int* dummy_this, int* length);

static PyObject* wrap_test_module_test_routine4(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_x1 = NULL;
    PyObject* py_x2 = NULL;
    PyObject* py_x3 = NULL;
    PyObject* py_x4 = NULL;
    PyObject* py_x5 = NULL;
    PyObject* py_x6 = NULL;
    float x6_val = 0;
    PyArrayObject* x6_scalar_arr = NULL;
    int x6_scalar_copyback = 0;
    int x6_scalar_is_array = 0;
    static char *kwlist[] = {"x1", "x2", "x3", "x4", "x5", "x6", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOO", kwlist, &py_x1, &py_x2, &py_x3, &py_x4, &py_x5, &py_x6)) {
        return NULL;
    }
    
    PyArrayObject* x1_arr = NULL;
    float* x1 = NULL;
    /* Extract x1 array data */
    if (!PyArray_Check(py_x1)) {
        PyErr_SetString(PyExc_TypeError, "Argument x1 must be a NumPy array");
        return NULL;
    }
    x1_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_x1, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (x1_arr == NULL) {
        return NULL;
    }
    x1 = (float*)PyArray_DATA(x1_arr);
    int n0_x1 = (int)PyArray_DIM(x1_arr, 0);
    f90wrap_n0_val = n0_x1;
    
    PyObject* x2_handle_obj = NULL;
    PyObject* x2_sequence = NULL;
    Py_ssize_t x2_handle_len = 0;
    if (PyObject_HasAttrString(py_x2, "_handle")) {
        x2_handle_obj = PyObject_GetAttrString(py_x2, "_handle");
        if (x2_handle_obj == NULL) {
            return NULL;
        }
        x2_sequence = PySequence_Fast(x2_handle_obj, "Failed to access handle sequence");
        if (x2_sequence == NULL) {
            Py_DECREF(x2_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_x2)) {
        x2_sequence = PySequence_Fast(py_x2, "Argument x2 must be a handle sequence");
        if (x2_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument x2 must be a Fortran derived-type instance");
        return NULL;
    }
    x2_handle_len = PySequence_Fast_GET_SIZE(x2_sequence);
    if (x2_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument x2 has an invalid handle length");
        Py_DECREF(x2_sequence);
        if (x2_handle_obj) Py_DECREF(x2_handle_obj);
        return NULL;
    }
    int* x2 = (int*)malloc(sizeof(int) * x2_handle_len);
    if (x2 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(x2_sequence);
        if (x2_handle_obj) Py_DECREF(x2_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < x2_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(x2_sequence, i);
        if (item == NULL) {
            free(x2);
            Py_DECREF(x2_sequence);
            if (x2_handle_obj) Py_DECREF(x2_handle_obj);
            return NULL;
        }
        x2[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(x2);
            Py_DECREF(x2_sequence);
            if (x2_handle_obj) Py_DECREF(x2_handle_obj);
            return NULL;
        }
    }
    (void)x2_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* x3_handle_obj = NULL;
    PyObject* x3_sequence = NULL;
    Py_ssize_t x3_handle_len = 0;
    if (PyObject_HasAttrString(py_x3, "_handle")) {
        x3_handle_obj = PyObject_GetAttrString(py_x3, "_handle");
        if (x3_handle_obj == NULL) {
            return NULL;
        }
        x3_sequence = PySequence_Fast(x3_handle_obj, "Failed to access handle sequence");
        if (x3_sequence == NULL) {
            Py_DECREF(x3_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_x3)) {
        x3_sequence = PySequence_Fast(py_x3, "Argument x3 must be a handle sequence");
        if (x3_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument x3 must be a Fortran derived-type instance");
        return NULL;
    }
    x3_handle_len = PySequence_Fast_GET_SIZE(x3_sequence);
    if (x3_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument x3 has an invalid handle length");
        Py_DECREF(x3_sequence);
        if (x3_handle_obj) Py_DECREF(x3_handle_obj);
        return NULL;
    }
    int* x3 = (int*)malloc(sizeof(int) * x3_handle_len);
    if (x3 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(x3_sequence);
        if (x3_handle_obj) Py_DECREF(x3_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < x3_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(x3_sequence, i);
        if (item == NULL) {
            free(x3);
            Py_DECREF(x3_sequence);
            if (x3_handle_obj) Py_DECREF(x3_handle_obj);
            return NULL;
        }
        x3[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(x3);
            Py_DECREF(x3_sequence);
            if (x3_handle_obj) Py_DECREF(x3_handle_obj);
            return NULL;
        }
    }
    (void)x3_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* x4_handle_obj = NULL;
    PyObject* x4_sequence = NULL;
    Py_ssize_t x4_handle_len = 0;
    if (PyObject_HasAttrString(py_x4, "_handle")) {
        x4_handle_obj = PyObject_GetAttrString(py_x4, "_handle");
        if (x4_handle_obj == NULL) {
            return NULL;
        }
        x4_sequence = PySequence_Fast(x4_handle_obj, "Failed to access handle sequence");
        if (x4_sequence == NULL) {
            Py_DECREF(x4_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_x4)) {
        x4_sequence = PySequence_Fast(py_x4, "Argument x4 must be a handle sequence");
        if (x4_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument x4 must be a Fortran derived-type instance");
        return NULL;
    }
    x4_handle_len = PySequence_Fast_GET_SIZE(x4_sequence);
    if (x4_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument x4 has an invalid handle length");
        Py_DECREF(x4_sequence);
        if (x4_handle_obj) Py_DECREF(x4_handle_obj);
        return NULL;
    }
    int* x4 = (int*)malloc(sizeof(int) * x4_handle_len);
    if (x4 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(x4_sequence);
        if (x4_handle_obj) Py_DECREF(x4_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < x4_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(x4_sequence, i);
        if (item == NULL) {
            free(x4);
            Py_DECREF(x4_sequence);
            if (x4_handle_obj) Py_DECREF(x4_handle_obj);
            return NULL;
        }
        x4[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(x4);
            Py_DECREF(x4_sequence);
            if (x4_handle_obj) Py_DECREF(x4_handle_obj);
            return NULL;
        }
    }
    (void)x4_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* x5_handle_obj = NULL;
    PyObject* x5_sequence = NULL;
    Py_ssize_t x5_handle_len = 0;
    if (PyObject_HasAttrString(py_x5, "_handle")) {
        x5_handle_obj = PyObject_GetAttrString(py_x5, "_handle");
        if (x5_handle_obj == NULL) {
            return NULL;
        }
        x5_sequence = PySequence_Fast(x5_handle_obj, "Failed to access handle sequence");
        if (x5_sequence == NULL) {
            Py_DECREF(x5_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_x5)) {
        x5_sequence = PySequence_Fast(py_x5, "Argument x5 must be a handle sequence");
        if (x5_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument x5 must be a Fortran derived-type instance");
        return NULL;
    }
    x5_handle_len = PySequence_Fast_GET_SIZE(x5_sequence);
    if (x5_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument x5 has an invalid handle length");
        Py_DECREF(x5_sequence);
        if (x5_handle_obj) Py_DECREF(x5_handle_obj);
        return NULL;
    }
    int* x5 = (int*)malloc(sizeof(int) * x5_handle_len);
    if (x5 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(x5_sequence);
        if (x5_handle_obj) Py_DECREF(x5_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < x5_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(x5_sequence, i);
        if (item == NULL) {
            free(x5);
            Py_DECREF(x5_sequence);
            if (x5_handle_obj) Py_DECREF(x5_handle_obj);
            return NULL;
        }
        x5[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(x5);
            Py_DECREF(x5_sequence);
            if (x5_handle_obj) Py_DECREF(x5_handle_obj);
            return NULL;
        }
    }
    (void)x5_handle_len;  /* suppress unused warnings when unchanged */
    
    float* x6 = &x6_val;
    if (PyArray_Check(py_x6)) {
        x6_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_x6, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (x6_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(x6_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument x6 must have exactly one element");
            Py_DECREF(x6_scalar_arr);
            return NULL;
        }
        x6_scalar_is_array = 1;
        x6 = (float*)PyArray_DATA(x6_scalar_arr);
        x6_val = x6[0];
        if (PyArray_DATA(x6_scalar_arr) != PyArray_DATA((PyArrayObject*)py_x6) || PyArray_TYPE(x6_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_x6)) {
            x6_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_x6)) {
        x6_val = (float)PyFloat_AsDouble(py_x6);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument x6 must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_routine4)(&f90wrap_n0_val, x1, x2, x3, x4, x5, x6);
    if (PyErr_Occurred()) {
        Py_XDECREF(x1_arr);
        if (x2_sequence) Py_DECREF(x2_sequence);
        if (x2_handle_obj) Py_DECREF(x2_handle_obj);
        free(x2);
        if (x3_sequence) Py_DECREF(x3_sequence);
        if (x3_handle_obj) Py_DECREF(x3_handle_obj);
        free(x3);
        if (x4_sequence) Py_DECREF(x4_sequence);
        if (x4_handle_obj) Py_DECREF(x4_handle_obj);
        free(x4);
        if (x5_sequence) Py_DECREF(x5_sequence);
        if (x5_handle_obj) Py_DECREF(x5_handle_obj);
        free(x5);
        return NULL;
    }
    
    if (x6_scalar_is_array) {
        if (x6_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_x6, x6_scalar_arr) < 0) {
                Py_DECREF(x6_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(x6_scalar_arr);
    }
    Py_DECREF(x1_arr);
    if (x2_sequence) {
        Py_DECREF(x2_sequence);
    }
    if (x2_handle_obj) {
        Py_DECREF(x2_handle_obj);
    }
    free(x2);
    if (x3_sequence) {
        Py_DECREF(x3_sequence);
    }
    if (x3_handle_obj) {
        Py_DECREF(x3_handle_obj);
    }
    free(x3);
    if (x4_sequence) {
        Py_DECREF(x4_sequence);
    }
    if (x4_handle_obj) {
        Py_DECREF(x4_handle_obj);
    }
    free(x4);
    if (x5_sequence) {
        Py_DECREF(x5_sequence);
    }
    if (x5_handle_obj) {
        Py_DECREF(x5_handle_obj);
    }
    free(x5);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_module_test_type2_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_initialise)(this);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_this_obj = PyList_New(4);
    if (py_this_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)this[i]);
        if (item == NULL) {
            Py_DECREF(py_this_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_this_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_this_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_this_obj != NULL) return py_this_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_this_obj != NULL) Py_DECREF(py_this_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_this_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_this_obj);
    }
    return result_tuple;
}

static PyObject* wrap_test_module_test_type2_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    static char *kwlist[] = {"this", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_this)) {
        return NULL;
    }
    
    PyObject* this_handle_obj = NULL;
    PyObject* this_sequence = NULL;
    Py_ssize_t this_handle_len = 0;
    if (PyObject_HasAttrString(py_this, "_handle")) {
        this_handle_obj = PyObject_GetAttrString(py_this, "_handle");
        if (this_handle_obj == NULL) {
            return NULL;
        }
        this_sequence = PySequence_Fast(this_handle_obj, "Failed to access handle sequence");
        if (this_sequence == NULL) {
            Py_DECREF(this_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_this)) {
        this_sequence = PySequence_Fast(py_this, "Argument this must be a handle sequence");
        if (this_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument this must be a Fortran derived-type instance");
        return NULL;
    }
    this_handle_len = PySequence_Fast_GET_SIZE(this_sequence);
    if (this_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument this has an invalid handle length");
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    int* this = (int*)malloc(sizeof(int) * this_handle_len);
    if (this == NULL) {
        PyErr_NoMemory();
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < this_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(this_sequence, i);
        if (item == NULL) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
        this[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
    }
    (void)this_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_module_test_type2_xn_array_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array_initialise)(this);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_this_obj = PyList_New(4);
    if (py_this_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)this[i]);
        if (item == NULL) {
            Py_DECREF(py_this_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_this_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_this_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_this_obj != NULL) return py_this_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_this_obj != NULL) Py_DECREF(py_this_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_this_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_this_obj);
    }
    return result_tuple;
}

static PyObject* wrap_test_module_test_type2_xn_array_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    static char *kwlist[] = {"this", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_this)) {
        return NULL;
    }
    
    PyObject* this_handle_obj = NULL;
    PyObject* this_sequence = NULL;
    Py_ssize_t this_handle_len = 0;
    if (PyObject_HasAttrString(py_this, "_handle")) {
        this_handle_obj = PyObject_GetAttrString(py_this, "_handle");
        if (this_handle_obj == NULL) {
            return NULL;
        }
        this_sequence = PySequence_Fast(this_handle_obj, "Failed to access handle sequence");
        if (this_sequence == NULL) {
            Py_DECREF(this_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_this)) {
        this_sequence = PySequence_Fast(py_this, "Argument this must be a handle sequence");
        if (this_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument this must be a Fortran derived-type instance");
        return NULL;
    }
    this_handle_len = PySequence_Fast_GET_SIZE(this_sequence);
    if (this_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument this has an invalid handle length");
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    int* this = (int*)malloc(sizeof(int) * this_handle_len);
    if (this == NULL) {
        PyErr_NoMemory();
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < this_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(this_sequence, i);
        if (item == NULL) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
        this[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
    }
    (void)this_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_module_test_type2_xm_array_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array_initialise)(this);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_this_obj = PyList_New(4);
    if (py_this_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)this[i]);
        if (item == NULL) {
            Py_DECREF(py_this_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_this_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_this_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_this_obj != NULL) return py_this_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_this_obj != NULL) Py_DECREF(py_this_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_this_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_this_obj);
    }
    return result_tuple;
}

static PyObject* wrap_test_module_test_type2_xm_array_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    static char *kwlist[] = {"this", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_this)) {
        return NULL;
    }
    
    PyObject* this_handle_obj = NULL;
    PyObject* this_sequence = NULL;
    Py_ssize_t this_handle_len = 0;
    if (PyObject_HasAttrString(py_this, "_handle")) {
        this_handle_obj = PyObject_GetAttrString(py_this, "_handle");
        if (this_handle_obj == NULL) {
            return NULL;
        }
        this_sequence = PySequence_Fast(this_handle_obj, "Failed to access handle sequence");
        if (this_sequence == NULL) {
            Py_DECREF(this_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_this)) {
        this_sequence = PySequence_Fast(py_this, "Argument this must be a handle sequence");
        if (this_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument this must be a Fortran derived-type instance");
        return NULL;
    }
    this_handle_len = PySequence_Fast_GET_SIZE(this_sequence);
    if (this_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument this has an invalid handle length");
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    int* this = (int*)malloc(sizeof(int) * this_handle_len);
    if (this == NULL) {
        PyErr_NoMemory();
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < this_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(this_sequence, i);
        if (item == NULL) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
        this[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
    }
    (void)this_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_module_test_type2_x5_array_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array_initialise)(this);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_this_obj = PyList_New(4);
    if (py_this_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)this[i]);
        if (item == NULL) {
            Py_DECREF(py_this_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_this_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_this_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_this_obj != NULL) return py_this_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_this_obj != NULL) Py_DECREF(py_this_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_this_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_this_obj);
    }
    return result_tuple;
}

static PyObject* wrap_test_module_test_type2_x5_array_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    static char *kwlist[] = {"this", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_this)) {
        return NULL;
    }
    
    PyObject* this_handle_obj = NULL;
    PyObject* this_sequence = NULL;
    Py_ssize_t this_handle_len = 0;
    if (PyObject_HasAttrString(py_this, "_handle")) {
        this_handle_obj = PyObject_GetAttrString(py_this, "_handle");
        if (this_handle_obj == NULL) {
            return NULL;
        }
        this_sequence = PySequence_Fast(this_handle_obj, "Failed to access handle sequence");
        if (this_sequence == NULL) {
            Py_DECREF(this_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_this)) {
        this_sequence = PySequence_Fast(py_this, "Argument this must be a handle sequence");
        if (this_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument this must be a Fortran derived-type instance");
        return NULL;
    }
    this_handle_len = PySequence_Fast_GET_SIZE(this_sequence);
    if (this_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument this has an invalid handle length");
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    int* this = (int*)malloc(sizeof(int) * this_handle_len);
    if (this == NULL) {
        PyErr_NoMemory();
        Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < this_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(this_sequence, i);
        if (item == NULL) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
        this[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(this);
            Py_DECREF(this_sequence);
            if (this_handle_obj) Py_DECREF(this_handle_obj);
            return NULL;
        }
    }
    (void)this_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_module_helper_get_m(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_test_module__get__m)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_test_module_helper_get_n(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_test_module__get__n)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_test_module__test_type2_helper_array_y(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* dummy_handle = Py_None;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &dummy_handle)) {
        return NULL;
    }
    
    int dummy_this[4] = {0, 0, 0, 0};
    if (dummy_handle != Py_None) {
        PyObject* handle_sequence = PySequence_Fast(dummy_handle, "Handle must be a sequence");
        if (handle_sequence == NULL) {
            return NULL;
        }
        Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
        if (handle_len != 4) {
            Py_DECREF(handle_sequence);
            PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
            return NULL;
        }
        for (int i = 0; i < 4; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
            if (item == NULL) {
                Py_DECREF(handle_sequence);
                return NULL;
            }
            dummy_this[i] = (int)PyLong_AsLong(item);
            if (PyErr_Occurred()) {
                Py_DECREF(handle_sequence);
                return NULL;
            }
        }
        Py_DECREF(handle_sequence);
    }
    int nd = 0;
    int dtype = 0;
    int dshape[10] = {0};
    long long handle = 0;
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2__array__y)(dummy_this, &nd, &dtype, dshape, &handle);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (nd < 0 || nd > 10) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensionality");
        return NULL;
    }
    PyObject* shape_tuple = PyTuple_New(nd);
    if (shape_tuple == NULL) {
        return NULL;
    }
    for (int i = 0; i < nd; ++i) {
        PyObject* dim = PyLong_FromLong((long)dshape[i]);
        if (dim == NULL) {
            Py_DECREF(shape_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(shape_tuple, i, dim);
    }
    PyObject* result = PyTuple_New(4);
    if (result == NULL) {
        Py_DECREF(shape_tuple);
        return NULL;
    }
    PyObject* nd_obj = PyLong_FromLong((long)nd);
    if (nd_obj == NULL) {
        Py_DECREF(shape_tuple);
        Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, nd_obj);
    PyObject* dtype_obj = PyLong_FromLong((long)dtype);
    if (dtype_obj == NULL) {
        Py_DECREF(shape_tuple);
        Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 1, dtype_obj);
    PyTuple_SET_ITEM(result, 2, shape_tuple);
    shape_tuple = NULL;
    PyObject* handle_obj = PyLong_FromLongLong(handle);
    if (handle_obj == NULL) {
        Py_DECREF(result);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 3, handle_obj);
    return result;
}

static PyObject* wrap_test_module__test_type2_xn_array_helper_array_getitem_items(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    static char *kwlist[] = {"handle", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_parent, &index)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    int handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array__array_getitem__items)(parent_handle, &index, handle);
    if (PyErr_Occurred()) {
        Py_DECREF(parent_sequence);
        return NULL;
    }
    Py_DECREF(parent_sequence);
    PyObject* result = PyList_New(4);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)handle[i]);
        if (item == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* wrap_test_module__test_type2_xn_array_helper_array_setitem_items(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "index", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiO", kwlist, &py_parent, &index, &py_value)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    PyObject* value_handle_obj = NULL;
    PyObject* value_sequence = NULL;
    Py_ssize_t value_handle_len = 0;
    if (PyObject_HasAttrString(py_value, "_handle")) {
        value_handle_obj = PyObject_GetAttrString(py_value, "_handle");
        if (value_handle_obj == NULL) { return NULL; }
        value_sequence = PySequence_Fast(value_handle_obj, "Failed to access handle sequence");
        if (value_sequence == NULL) { Py_DECREF(value_handle_obj); return NULL; }
    } else if (PySequence_Check(py_value)) {
        value_sequence = PySequence_Fast(py_value, "Argument value must be a handle sequence");
        if (value_sequence == NULL) { return NULL; }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument value must be a Fortran derived-type instance");
        return NULL;
    }
    value_handle_len = PySequence_Fast_GET_SIZE(value_sequence);
    if (value_handle_len != 4) {
        Py_DECREF(parent_sequence);
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    Py_DECREF(parent_sequence);
    int* value = (int*)malloc(sizeof(int) * 4);
    if (value == NULL) {
        PyErr_NoMemory();
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);
        if (item == NULL) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
        value[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
    }
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array__array_setitem__items)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_module__test_type2_xn_array_helper_array_len_items(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_parent;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_parent)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    int length = 0;
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xn_array__array_len__items)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_test_module__test_type2_xm_array_helper_array_getitem_items(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    static char *kwlist[] = {"handle", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_parent, &index)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    int handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array__array_getitem__items)(parent_handle, &index, handle);
    if (PyErr_Occurred()) {
        Py_DECREF(parent_sequence);
        return NULL;
    }
    Py_DECREF(parent_sequence);
    PyObject* result = PyList_New(4);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)handle[i]);
        if (item == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* wrap_test_module__test_type2_xm_array_helper_array_setitem_items(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "index", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiO", kwlist, &py_parent, &index, &py_value)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    PyObject* value_handle_obj = NULL;
    PyObject* value_sequence = NULL;
    Py_ssize_t value_handle_len = 0;
    if (PyObject_HasAttrString(py_value, "_handle")) {
        value_handle_obj = PyObject_GetAttrString(py_value, "_handle");
        if (value_handle_obj == NULL) { return NULL; }
        value_sequence = PySequence_Fast(value_handle_obj, "Failed to access handle sequence");
        if (value_sequence == NULL) { Py_DECREF(value_handle_obj); return NULL; }
    } else if (PySequence_Check(py_value)) {
        value_sequence = PySequence_Fast(py_value, "Argument value must be a handle sequence");
        if (value_sequence == NULL) { return NULL; }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument value must be a Fortran derived-type instance");
        return NULL;
    }
    value_handle_len = PySequence_Fast_GET_SIZE(value_sequence);
    if (value_handle_len != 4) {
        Py_DECREF(parent_sequence);
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    Py_DECREF(parent_sequence);
    int* value = (int*)malloc(sizeof(int) * 4);
    if (value == NULL) {
        PyErr_NoMemory();
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);
        if (item == NULL) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
        value[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
    }
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array__array_setitem__items)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_module__test_type2_xm_array_helper_array_len_items(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_parent;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_parent)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    int length = 0;
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_xm_array__array_len__items)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_test_module__test_type2_x5_array_helper_array_getitem_items(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    static char *kwlist[] = {"handle", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_parent, &index)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    int handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array__array_getitem__items)(parent_handle, &index, handle);
    if (PyErr_Occurred()) {
        Py_DECREF(parent_sequence);
        return NULL;
    }
    Py_DECREF(parent_sequence);
    PyObject* result = PyList_New(4);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)handle[i]);
        if (item == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* wrap_test_module__test_type2_x5_array_helper_array_setitem_items(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "index", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiO", kwlist, &py_parent, &index, &py_value)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    PyObject* value_handle_obj = NULL;
    PyObject* value_sequence = NULL;
    Py_ssize_t value_handle_len = 0;
    if (PyObject_HasAttrString(py_value, "_handle")) {
        value_handle_obj = PyObject_GetAttrString(py_value, "_handle");
        if (value_handle_obj == NULL) { return NULL; }
        value_sequence = PySequence_Fast(value_handle_obj, "Failed to access handle sequence");
        if (value_sequence == NULL) { Py_DECREF(value_handle_obj); return NULL; }
    } else if (PySequence_Check(py_value)) {
        value_sequence = PySequence_Fast(py_value, "Argument value must be a handle sequence");
        if (value_sequence == NULL) { return NULL; }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument value must be a Fortran derived-type instance");
        return NULL;
    }
    value_handle_len = PySequence_Fast_GET_SIZE(value_sequence);
    if (value_handle_len != 4) {
        Py_DECREF(parent_sequence);
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    Py_DECREF(parent_sequence);
    int* value = (int*)malloc(sizeof(int) * 4);
    if (value == NULL) {
        PyErr_NoMemory();
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);
        if (item == NULL) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
        value[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
    }
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array__array_setitem__items)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_module__test_type2_x5_array_helper_array_len_items(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_parent;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_parent)) {
        return NULL;
    }
    PyObject* parent_sequence = PySequence_Fast(py_parent, "Handle must be a sequence");
    if (parent_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t parent_len = PySequence_Fast_GET_SIZE(parent_sequence);
    if (parent_len != 4) {
        Py_DECREF(parent_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int parent_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(parent_sequence, i);
        if (item == NULL) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
        parent_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(parent_sequence);
            return NULL;
        }
    }
    int length = 0;
    F90WRAP_F_SYMBOL(f90wrap_test_module__test_type2_x5_array__array_len__items)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

/* Method table for _test_python module */
static PyMethodDef _test_python_methods[] = {
    {"f90wrap_test_module__test_routine4", (PyCFunction)wrap_test_module_test_routine4, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for test_routine4"},
    {"f90wrap_test_module__test_type2_initialise", (PyCFunction)wrap_test_module_test_type2_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for test_type2"},
    {"f90wrap_test_module__test_type2_finalise", (PyCFunction)wrap_test_module_test_type2_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for test_type2"},
    {"f90wrap_test_module__test_type2_xn_array_initialise", (PyCFunction)wrap_test_module_test_type2_xn_array_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for test_type2_xn_array"},
    {"f90wrap_test_module__test_type2_xn_array_finalise", (PyCFunction)wrap_test_module_test_type2_xn_array_finalise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated destructor for test_type2_xn_array"},
    {"f90wrap_test_module__test_type2_xm_array_initialise", (PyCFunction)wrap_test_module_test_type2_xm_array_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for test_type2_xm_array"},
    {"f90wrap_test_module__test_type2_xm_array_finalise", (PyCFunction)wrap_test_module_test_type2_xm_array_finalise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated destructor for test_type2_xm_array"},
    {"f90wrap_test_module__test_type2_x5_array_initialise", (PyCFunction)wrap_test_module_test_type2_x5_array_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for test_type2_x5_array"},
    {"f90wrap_test_module__test_type2_x5_array_finalise", (PyCFunction)wrap_test_module_test_type2_x5_array_finalise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated destructor for test_type2_x5_array"},
    {"f90wrap_test_module__get__m", (PyCFunction)wrap_test_module_helper_get_m, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for m"},
    {"f90wrap_test_module__get__n", (PyCFunction)wrap_test_module_helper_get_n, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for n"},
    {"f90wrap_test_module__test_type2__array__y", (PyCFunction)wrap_test_module__test_type2_helper_array_y, METH_VARARGS | \
        METH_KEYWORDS, "Array helper for y"},
    {"f90wrap_test_module__test_type2_xn_array__array_getitem__items", \
        (PyCFunction)wrap_test_module__test_type2_xn_array_helper_array_getitem_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {"f90wrap_test_module__test_type2_xn_array__array_setitem__items", \
        (PyCFunction)wrap_test_module__test_type2_xn_array_helper_array_setitem_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {"f90wrap_test_module__test_type2_xn_array__array_len__items", \
        (PyCFunction)wrap_test_module__test_type2_xn_array_helper_array_len_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {"f90wrap_test_module__test_type2_xm_array__array_getitem__items", \
        (PyCFunction)wrap_test_module__test_type2_xm_array_helper_array_getitem_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {"f90wrap_test_module__test_type2_xm_array__array_setitem__items", \
        (PyCFunction)wrap_test_module__test_type2_xm_array_helper_array_setitem_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {"f90wrap_test_module__test_type2_xm_array__array_len__items", \
        (PyCFunction)wrap_test_module__test_type2_xm_array_helper_array_len_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {"f90wrap_test_module__test_type2_x5_array__array_getitem__items", \
        (PyCFunction)wrap_test_module__test_type2_x5_array_helper_array_getitem_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {"f90wrap_test_module__test_type2_x5_array__array_setitem__items", \
        (PyCFunction)wrap_test_module__test_type2_x5_array_helper_array_setitem_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {"f90wrap_test_module__test_type2_x5_array__array_len__items", \
        (PyCFunction)wrap_test_module__test_type2_x5_array_helper_array_len_items, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for items"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _test_pythonmodule = {
    PyModuleDef_HEAD_INIT,
    "test_python",
    "Direct-C wrapper for _test_python module",
    -1,
    _test_python_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__test_python(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_test_pythonmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
