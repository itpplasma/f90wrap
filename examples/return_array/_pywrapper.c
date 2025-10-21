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
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__array_init)(int* in_array, int* in_size);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__array_2d_init)(int* in_array, int* in_size_x, int* in_size_y);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__array_wrapper_init)(int* in_wrapper, int* in_size);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__array_free)(int* in_array);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_scalar)(int* in_array, float* ret_return_scalar);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_hard_coded_1d)(float* ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_hard_coded_2d)(float* ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_member)(int* f90wrap_n0, int* in_array, float* ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_member_2d)(int* f90wrap_n0, int* f90wrap_n1, int* in_array, \
    float* ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_member_wrapper)(int* f90wrap_n0, int* in_wrapper, float* \
    ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_input)(int* f90wrap_n0, int* in_len, float* ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_input_2d)(int* f90wrap_n0, int* f90wrap_n1, int* in_len_x, \
    int* in_len_y, float* ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_size)(int* f90wrap_n0, int* f90wrap_n1, float* in_array, \
    float* ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_size_2d_in)(int* f90wrap_n0, int* f90wrap_n1, int* f90wrap_n2, \
    float* in_array, float* ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_size_2d_out)(int* f90wrap_n0, int* f90wrap_n1, int* \
    f90wrap_n2, int* f90wrap_n3, int* f90wrap_n4, int* f90wrap_n5, float* in_array_1, float* in_array_2, float* \
    ret_retval);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__return_derived_type_value)(int* f90wrap_n0, int* f90wrap_n1, int* this, \
    int* size_2d, float* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_double_wrapper_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_double_wrapper_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_value_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_value_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper__get__a_size)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper__set__a_size)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper__array__a_data)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__get__a_size_x)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__set__a_size_x)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__get__a_size_y)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__set__a_size_y)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__array__a_data)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_double_wrapper__get__array_wrapper)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_double_wrapper__set__array_wrapper)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_value__get__value)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_value__set__value)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d__get__x)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d__set__x)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d__get__y)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d__set__y)(int* handle, int* value);

static PyObject* wrap_m_test_array_init(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_array = NULL;
    PyObject* py_in_size = NULL;
    int in_size_val = 0;
    PyArrayObject* in_size_scalar_arr = NULL;
    int in_size_scalar_copyback = 0;
    int in_size_scalar_is_array = 0;
    static char *kwlist[] = {"in_array", "in_size", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_in_array, &py_in_size)) {
        return NULL;
    }
    
    PyObject* in_array_handle_obj = NULL;
    PyObject* in_array_sequence = NULL;
    Py_ssize_t in_array_handle_len = 0;
    if (PyObject_HasAttrString(py_in_array, "_handle")) {
        in_array_handle_obj = PyObject_GetAttrString(py_in_array, "_handle");
        if (in_array_handle_obj == NULL) {
            return NULL;
        }
        in_array_sequence = PySequence_Fast(in_array_handle_obj, "Failed to access handle sequence");
        if (in_array_sequence == NULL) {
            Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_in_array)) {
        in_array_sequence = PySequence_Fast(py_in_array, "Argument in_array must be a handle sequence");
        if (in_array_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_array must be a Fortran derived-type instance");
        return NULL;
    }
    in_array_handle_len = PySequence_Fast_GET_SIZE(in_array_sequence);
    if (in_array_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument in_array has an invalid handle length");
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    int* in_array = (int*)malloc(sizeof(int) * in_array_handle_len);
    if (in_array == NULL) {
        PyErr_NoMemory();
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < in_array_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(in_array_sequence, i);
        if (item == NULL) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
        in_array[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    }
    (void)in_array_handle_len;  /* suppress unused warnings when unchanged */
    
    int* in_size = &in_size_val;
    if (PyArray_Check(py_in_size)) {
        in_size_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_size, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_size_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_size_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_size must have exactly one element");
            Py_DECREF(in_size_scalar_arr);
            return NULL;
        }
        in_size_scalar_is_array = 1;
        in_size = (int*)PyArray_DATA(in_size_scalar_arr);
        in_size_val = in_size[0];
        if (PyArray_DATA(in_size_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_size) || PyArray_TYPE(in_size_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in_size)) {
            in_size_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_size)) {
        in_size_val = (int)PyLong_AsLong(py_in_size);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_size must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__array_init)(in_array, in_size);
    if (PyErr_Occurred()) {
        if (in_array_sequence) Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        free(in_array);
        return NULL;
    }
    
    if (in_size_scalar_is_array) {
        if (in_size_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_size, in_size_scalar_arr) < 0) {
                Py_DECREF(in_size_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_size_scalar_arr);
    }
    if (in_array_sequence) {
        Py_DECREF(in_array_sequence);
    }
    if (in_array_handle_obj) {
        Py_DECREF(in_array_handle_obj);
    }
    free(in_array);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test_array_2d_init(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_array = NULL;
    PyObject* py_in_size_x = NULL;
    int in_size_x_val = 0;
    PyArrayObject* in_size_x_scalar_arr = NULL;
    int in_size_x_scalar_copyback = 0;
    int in_size_x_scalar_is_array = 0;
    PyObject* py_in_size_y = NULL;
    int in_size_y_val = 0;
    PyArrayObject* in_size_y_scalar_arr = NULL;
    int in_size_y_scalar_copyback = 0;
    int in_size_y_scalar_is_array = 0;
    static char *kwlist[] = {"in_array", "in_size_x", "in_size_y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_in_array, &py_in_size_x, &py_in_size_y)) {
        return NULL;
    }
    
    PyObject* in_array_handle_obj = NULL;
    PyObject* in_array_sequence = NULL;
    Py_ssize_t in_array_handle_len = 0;
    if (PyObject_HasAttrString(py_in_array, "_handle")) {
        in_array_handle_obj = PyObject_GetAttrString(py_in_array, "_handle");
        if (in_array_handle_obj == NULL) {
            return NULL;
        }
        in_array_sequence = PySequence_Fast(in_array_handle_obj, "Failed to access handle sequence");
        if (in_array_sequence == NULL) {
            Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_in_array)) {
        in_array_sequence = PySequence_Fast(py_in_array, "Argument in_array must be a handle sequence");
        if (in_array_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_array must be a Fortran derived-type instance");
        return NULL;
    }
    in_array_handle_len = PySequence_Fast_GET_SIZE(in_array_sequence);
    if (in_array_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument in_array has an invalid handle length");
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    int* in_array = (int*)malloc(sizeof(int) * in_array_handle_len);
    if (in_array == NULL) {
        PyErr_NoMemory();
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < in_array_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(in_array_sequence, i);
        if (item == NULL) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
        in_array[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    }
    (void)in_array_handle_len;  /* suppress unused warnings when unchanged */
    
    int* in_size_x = &in_size_x_val;
    if (PyArray_Check(py_in_size_x)) {
        in_size_x_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_size_x, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_size_x_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_size_x_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_size_x must have exactly one element");
            Py_DECREF(in_size_x_scalar_arr);
            return NULL;
        }
        in_size_x_scalar_is_array = 1;
        in_size_x = (int*)PyArray_DATA(in_size_x_scalar_arr);
        in_size_x_val = in_size_x[0];
        if (PyArray_DATA(in_size_x_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_size_x) || \
            PyArray_TYPE(in_size_x_scalar_arr) != PyArray_TYPE((PyArrayObject*)py_in_size_x)) {
            in_size_x_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_size_x)) {
        in_size_x_val = (int)PyLong_AsLong(py_in_size_x);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_size_x must be a scalar number or NumPy array");
        return NULL;
    }
    int* in_size_y = &in_size_y_val;
    if (PyArray_Check(py_in_size_y)) {
        in_size_y_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_size_y, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_size_y_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_size_y_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_size_y must have exactly one element");
            Py_DECREF(in_size_y_scalar_arr);
            return NULL;
        }
        in_size_y_scalar_is_array = 1;
        in_size_y = (int*)PyArray_DATA(in_size_y_scalar_arr);
        in_size_y_val = in_size_y[0];
        if (PyArray_DATA(in_size_y_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_size_y) || \
            PyArray_TYPE(in_size_y_scalar_arr) != PyArray_TYPE((PyArrayObject*)py_in_size_y)) {
            in_size_y_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_size_y)) {
        in_size_y_val = (int)PyLong_AsLong(py_in_size_y);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_size_y must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__array_2d_init)(in_array, in_size_x, in_size_y);
    if (PyErr_Occurred()) {
        if (in_array_sequence) Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        free(in_array);
        return NULL;
    }
    
    if (in_size_x_scalar_is_array) {
        if (in_size_x_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_size_x, in_size_x_scalar_arr) < 0) {
                Py_DECREF(in_size_x_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_size_x_scalar_arr);
    }
    if (in_size_y_scalar_is_array) {
        if (in_size_y_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_size_y, in_size_y_scalar_arr) < 0) {
                Py_DECREF(in_size_y_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_size_y_scalar_arr);
    }
    if (in_array_sequence) {
        Py_DECREF(in_array_sequence);
    }
    if (in_array_handle_obj) {
        Py_DECREF(in_array_handle_obj);
    }
    free(in_array);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test_array_wrapper_init(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_wrapper = NULL;
    PyObject* py_in_size = NULL;
    int in_size_val = 0;
    PyArrayObject* in_size_scalar_arr = NULL;
    int in_size_scalar_copyback = 0;
    int in_size_scalar_is_array = 0;
    static char *kwlist[] = {"in_wrapper", "in_size", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_in_wrapper, &py_in_size)) {
        return NULL;
    }
    
    PyObject* in_wrapper_handle_obj = NULL;
    PyObject* in_wrapper_sequence = NULL;
    Py_ssize_t in_wrapper_handle_len = 0;
    if (PyObject_HasAttrString(py_in_wrapper, "_handle")) {
        in_wrapper_handle_obj = PyObject_GetAttrString(py_in_wrapper, "_handle");
        if (in_wrapper_handle_obj == NULL) {
            return NULL;
        }
        in_wrapper_sequence = PySequence_Fast(in_wrapper_handle_obj, "Failed to access handle sequence");
        if (in_wrapper_sequence == NULL) {
            Py_DECREF(in_wrapper_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_in_wrapper)) {
        in_wrapper_sequence = PySequence_Fast(py_in_wrapper, "Argument in_wrapper must be a handle sequence");
        if (in_wrapper_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_wrapper must be a Fortran derived-type instance");
        return NULL;
    }
    in_wrapper_handle_len = PySequence_Fast_GET_SIZE(in_wrapper_sequence);
    if (in_wrapper_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument in_wrapper has an invalid handle length");
        Py_DECREF(in_wrapper_sequence);
        if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
        return NULL;
    }
    int* in_wrapper = (int*)malloc(sizeof(int) * in_wrapper_handle_len);
    if (in_wrapper == NULL) {
        PyErr_NoMemory();
        Py_DECREF(in_wrapper_sequence);
        if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < in_wrapper_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(in_wrapper_sequence, i);
        if (item == NULL) {
            free(in_wrapper);
            Py_DECREF(in_wrapper_sequence);
            if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
            return NULL;
        }
        in_wrapper[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(in_wrapper);
            Py_DECREF(in_wrapper_sequence);
            if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
            return NULL;
        }
    }
    (void)in_wrapper_handle_len;  /* suppress unused warnings when unchanged */
    
    int* in_size = &in_size_val;
    if (PyArray_Check(py_in_size)) {
        in_size_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_size, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_size_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_size_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_size must have exactly one element");
            Py_DECREF(in_size_scalar_arr);
            return NULL;
        }
        in_size_scalar_is_array = 1;
        in_size = (int*)PyArray_DATA(in_size_scalar_arr);
        in_size_val = in_size[0];
        if (PyArray_DATA(in_size_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_size) || PyArray_TYPE(in_size_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in_size)) {
            in_size_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_size)) {
        in_size_val = (int)PyLong_AsLong(py_in_size);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_size must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__array_wrapper_init)(in_wrapper, in_size);
    if (PyErr_Occurred()) {
        if (in_wrapper_sequence) Py_DECREF(in_wrapper_sequence);
        if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
        free(in_wrapper);
        return NULL;
    }
    
    if (in_size_scalar_is_array) {
        if (in_size_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_size, in_size_scalar_arr) < 0) {
                Py_DECREF(in_size_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_size_scalar_arr);
    }
    if (in_wrapper_sequence) {
        Py_DECREF(in_wrapper_sequence);
    }
    if (in_wrapper_handle_obj) {
        Py_DECREF(in_wrapper_handle_obj);
    }
    free(in_wrapper);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test_array_free(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_array = NULL;
    static char *kwlist[] = {"in_array", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_in_array)) {
        return NULL;
    }
    
    PyObject* in_array_handle_obj = NULL;
    PyObject* in_array_sequence = NULL;
    Py_ssize_t in_array_handle_len = 0;
    if (PyObject_HasAttrString(py_in_array, "_handle")) {
        in_array_handle_obj = PyObject_GetAttrString(py_in_array, "_handle");
        if (in_array_handle_obj == NULL) {
            return NULL;
        }
        in_array_sequence = PySequence_Fast(in_array_handle_obj, "Failed to access handle sequence");
        if (in_array_sequence == NULL) {
            Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_in_array)) {
        in_array_sequence = PySequence_Fast(py_in_array, "Argument in_array must be a handle sequence");
        if (in_array_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_array must be a Fortran derived-type instance");
        return NULL;
    }
    in_array_handle_len = PySequence_Fast_GET_SIZE(in_array_sequence);
    if (in_array_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument in_array has an invalid handle length");
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    int* in_array = (int*)malloc(sizeof(int) * in_array_handle_len);
    if (in_array == NULL) {
        PyErr_NoMemory();
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < in_array_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(in_array_sequence, i);
        if (item == NULL) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
        in_array[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    }
    (void)in_array_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__array_free)(in_array);
    if (PyErr_Occurred()) {
        if (in_array_sequence) Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        free(in_array);
        return NULL;
    }
    
    if (in_array_sequence) {
        Py_DECREF(in_array_sequence);
    }
    if (in_array_handle_obj) {
        Py_DECREF(in_array_handle_obj);
    }
    free(in_array);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test_return_scalar(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_array = NULL;
    float ret_return_scalar_val = 0;
    static char *kwlist[] = {"in_array", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_in_array)) {
        return NULL;
    }
    
    PyObject* in_array_handle_obj = NULL;
    PyObject* in_array_sequence = NULL;
    Py_ssize_t in_array_handle_len = 0;
    if (PyObject_HasAttrString(py_in_array, "_handle")) {
        in_array_handle_obj = PyObject_GetAttrString(py_in_array, "_handle");
        if (in_array_handle_obj == NULL) {
            return NULL;
        }
        in_array_sequence = PySequence_Fast(in_array_handle_obj, "Failed to access handle sequence");
        if (in_array_sequence == NULL) {
            Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_in_array)) {
        in_array_sequence = PySequence_Fast(py_in_array, "Argument in_array must be a handle sequence");
        if (in_array_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_array must be a Fortran derived-type instance");
        return NULL;
    }
    in_array_handle_len = PySequence_Fast_GET_SIZE(in_array_sequence);
    if (in_array_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument in_array has an invalid handle length");
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    int* in_array = (int*)malloc(sizeof(int) * in_array_handle_len);
    if (in_array == NULL) {
        PyErr_NoMemory();
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < in_array_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(in_array_sequence, i);
        if (item == NULL) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
        in_array[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    }
    (void)in_array_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_scalar)(in_array, &ret_return_scalar_val);
    if (PyErr_Occurred()) {
        if (in_array_sequence) Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        free(in_array);
        return NULL;
    }
    
    PyObject* py_ret_return_scalar_obj = Py_BuildValue("d", ret_return_scalar_val);
    if (py_ret_return_scalar_obj == NULL) {
        return NULL;
    }
    if (in_array_sequence) {
        Py_DECREF(in_array_sequence);
    }
    if (in_array_handle_obj) {
        Py_DECREF(in_array_handle_obj);
    }
    free(in_array);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_return_scalar_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_return_scalar_obj != NULL) return py_ret_return_scalar_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_return_scalar_obj != NULL) Py_DECREF(py_ret_return_scalar_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_return_scalar_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_return_scalar_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_hard_coded_1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(10);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    npy_intp ret_retval_dims[1] = {ret_retval_dim_0};
    py_ret_retval_arr = PyArray_SimpleNew(1, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_hard_coded_1d)(ret_retval);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_hard_coded_2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(5);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    npy_intp ret_retval_dim_1 = (npy_intp)(6);
    if (ret_retval_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    npy_intp ret_retval_dims[2] = {ret_retval_dim_0, ret_retval_dim_1};
    py_ret_retval_arr = PyArray_SimpleNew(2, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_hard_coded_2d)(ret_retval);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_array_member(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_in_array = NULL;
    static char *kwlist[] = {"f90wrap_n0", "in_array", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iO", kwlist, &f90wrap_n0_val, &py_in_array)) {
        return NULL;
    }
    
    PyObject* in_array_handle_obj = NULL;
    PyObject* in_array_sequence = NULL;
    Py_ssize_t in_array_handle_len = 0;
    if (PyObject_HasAttrString(py_in_array, "_handle")) {
        in_array_handle_obj = PyObject_GetAttrString(py_in_array, "_handle");
        if (in_array_handle_obj == NULL) {
            return NULL;
        }
        in_array_sequence = PySequence_Fast(in_array_handle_obj, "Failed to access handle sequence");
        if (in_array_sequence == NULL) {
            Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_in_array)) {
        in_array_sequence = PySequence_Fast(py_in_array, "Argument in_array must be a handle sequence");
        if (in_array_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_array must be a Fortran derived-type instance");
        return NULL;
    }
    in_array_handle_len = PySequence_Fast_GET_SIZE(in_array_sequence);
    if (in_array_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument in_array has an invalid handle length");
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    int* in_array = (int*)malloc(sizeof(int) * in_array_handle_len);
    if (in_array == NULL) {
        PyErr_NoMemory();
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < in_array_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(in_array_sequence, i);
        if (item == NULL) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
        in_array[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    }
    (void)in_array_handle_len;  /* suppress unused warnings when unchanged */
    
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(f90wrap_n0_val);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n0_val = (int)ret_retval_dim_0;
    npy_intp ret_retval_dims[1] = {ret_retval_dim_0};
    py_ret_retval_arr = PyArray_SimpleNew(1, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_member)(&f90wrap_n0_val, in_array, ret_retval);
    if (PyErr_Occurred()) {
        if (in_array_sequence) Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        free(in_array);
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    if (in_array_sequence) {
        Py_DECREF(in_array_sequence);
    }
    if (in_array_handle_obj) {
        Py_DECREF(in_array_handle_obj);
    }
    free(in_array);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_array_member_2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_in_array = NULL;
    static char *kwlist[] = {"f90wrap_n0", "f90wrap_n1", "in_array", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiO", kwlist, &f90wrap_n0_val, &f90wrap_n1_val, &py_in_array)) {
        return NULL;
    }
    
    PyObject* in_array_handle_obj = NULL;
    PyObject* in_array_sequence = NULL;
    Py_ssize_t in_array_handle_len = 0;
    if (PyObject_HasAttrString(py_in_array, "_handle")) {
        in_array_handle_obj = PyObject_GetAttrString(py_in_array, "_handle");
        if (in_array_handle_obj == NULL) {
            return NULL;
        }
        in_array_sequence = PySequence_Fast(in_array_handle_obj, "Failed to access handle sequence");
        if (in_array_sequence == NULL) {
            Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_in_array)) {
        in_array_sequence = PySequence_Fast(py_in_array, "Argument in_array must be a handle sequence");
        if (in_array_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_array must be a Fortran derived-type instance");
        return NULL;
    }
    in_array_handle_len = PySequence_Fast_GET_SIZE(in_array_sequence);
    if (in_array_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument in_array has an invalid handle length");
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    int* in_array = (int*)malloc(sizeof(int) * in_array_handle_len);
    if (in_array == NULL) {
        PyErr_NoMemory();
        Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < in_array_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(in_array_sequence, i);
        if (item == NULL) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
        in_array[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(in_array);
            Py_DECREF(in_array_sequence);
            if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
            return NULL;
        }
    }
    (void)in_array_handle_len;  /* suppress unused warnings when unchanged */
    
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(f90wrap_n0_val);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n0_val = (int)ret_retval_dim_0;
    npy_intp ret_retval_dim_1 = (npy_intp)(f90wrap_n1_val);
    if (ret_retval_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_retval_dim_1;
    npy_intp ret_retval_dims[2] = {ret_retval_dim_0, ret_retval_dim_1};
    py_ret_retval_arr = PyArray_SimpleNew(2, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_member_2d)(&f90wrap_n0_val, &f90wrap_n1_val, in_array, ret_retval);
    if (PyErr_Occurred()) {
        if (in_array_sequence) Py_DECREF(in_array_sequence);
        if (in_array_handle_obj) Py_DECREF(in_array_handle_obj);
        free(in_array);
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    if (in_array_sequence) {
        Py_DECREF(in_array_sequence);
    }
    if (in_array_handle_obj) {
        Py_DECREF(in_array_handle_obj);
    }
    free(in_array);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_array_member_wrapper(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_in_wrapper = NULL;
    static char *kwlist[] = {"f90wrap_n0", "in_wrapper", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iO", kwlist, &f90wrap_n0_val, &py_in_wrapper)) {
        return NULL;
    }
    
    PyObject* in_wrapper_handle_obj = NULL;
    PyObject* in_wrapper_sequence = NULL;
    Py_ssize_t in_wrapper_handle_len = 0;
    if (PyObject_HasAttrString(py_in_wrapper, "_handle")) {
        in_wrapper_handle_obj = PyObject_GetAttrString(py_in_wrapper, "_handle");
        if (in_wrapper_handle_obj == NULL) {
            return NULL;
        }
        in_wrapper_sequence = PySequence_Fast(in_wrapper_handle_obj, "Failed to access handle sequence");
        if (in_wrapper_sequence == NULL) {
            Py_DECREF(in_wrapper_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_in_wrapper)) {
        in_wrapper_sequence = PySequence_Fast(py_in_wrapper, "Argument in_wrapper must be a handle sequence");
        if (in_wrapper_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_wrapper must be a Fortran derived-type instance");
        return NULL;
    }
    in_wrapper_handle_len = PySequence_Fast_GET_SIZE(in_wrapper_sequence);
    if (in_wrapper_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument in_wrapper has an invalid handle length");
        Py_DECREF(in_wrapper_sequence);
        if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
        return NULL;
    }
    int* in_wrapper = (int*)malloc(sizeof(int) * in_wrapper_handle_len);
    if (in_wrapper == NULL) {
        PyErr_NoMemory();
        Py_DECREF(in_wrapper_sequence);
        if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < in_wrapper_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(in_wrapper_sequence, i);
        if (item == NULL) {
            free(in_wrapper);
            Py_DECREF(in_wrapper_sequence);
            if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
            return NULL;
        }
        in_wrapper[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(in_wrapper);
            Py_DECREF(in_wrapper_sequence);
            if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
            return NULL;
        }
    }
    (void)in_wrapper_handle_len;  /* suppress unused warnings when unchanged */
    
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(f90wrap_n0_val);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n0_val = (int)ret_retval_dim_0;
    npy_intp ret_retval_dims[1] = {ret_retval_dim_0};
    py_ret_retval_arr = PyArray_SimpleNew(1, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_member_wrapper)(&f90wrap_n0_val, in_wrapper, ret_retval);
    if (PyErr_Occurred()) {
        if (in_wrapper_sequence) Py_DECREF(in_wrapper_sequence);
        if (in_wrapper_handle_obj) Py_DECREF(in_wrapper_handle_obj);
        free(in_wrapper);
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    if (in_wrapper_sequence) {
        Py_DECREF(in_wrapper_sequence);
    }
    if (in_wrapper_handle_obj) {
        Py_DECREF(in_wrapper_handle_obj);
    }
    free(in_wrapper);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_array_input(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_in_len = NULL;
    int in_len_val = 0;
    PyArrayObject* in_len_scalar_arr = NULL;
    int in_len_scalar_copyback = 0;
    int in_len_scalar_is_array = 0;
    static char *kwlist[] = {"f90wrap_n0", "in_len", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iO", kwlist, &f90wrap_n0_val, &py_in_len)) {
        return NULL;
    }
    
    int* in_len = &in_len_val;
    if (PyArray_Check(py_in_len)) {
        in_len_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_len, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_len_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_len_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_len must have exactly one element");
            Py_DECREF(in_len_scalar_arr);
            return NULL;
        }
        in_len_scalar_is_array = 1;
        in_len = (int*)PyArray_DATA(in_len_scalar_arr);
        in_len_val = in_len[0];
        if (PyArray_DATA(in_len_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_len) || PyArray_TYPE(in_len_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in_len)) {
            in_len_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_len)) {
        in_len_val = (int)PyLong_AsLong(py_in_len);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_len must be a scalar number or NumPy array");
        return NULL;
    }
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(f90wrap_n0_val);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n0_val = (int)ret_retval_dim_0;
    npy_intp ret_retval_dims[1] = {ret_retval_dim_0};
    py_ret_retval_arr = PyArray_SimpleNew(1, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_input)(&f90wrap_n0_val, in_len, ret_retval);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    if (in_len_scalar_is_array) {
        if (in_len_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_len, in_len_scalar_arr) < 0) {
                Py_DECREF(in_len_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_len_scalar_arr);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_array_input_2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_in_len_x = NULL;
    int in_len_x_val = 0;
    PyArrayObject* in_len_x_scalar_arr = NULL;
    int in_len_x_scalar_copyback = 0;
    int in_len_x_scalar_is_array = 0;
    PyObject* py_in_len_y = NULL;
    int in_len_y_val = 0;
    PyArrayObject* in_len_y_scalar_arr = NULL;
    int in_len_y_scalar_copyback = 0;
    int in_len_y_scalar_is_array = 0;
    static char *kwlist[] = {"f90wrap_n0", "f90wrap_n1", "in_len_x", "in_len_y", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiOO", kwlist, &f90wrap_n0_val, &f90wrap_n1_val, &py_in_len_x, \
        &py_in_len_y)) {
        return NULL;
    }
    
    int* in_len_x = &in_len_x_val;
    if (PyArray_Check(py_in_len_x)) {
        in_len_x_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_len_x, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_len_x_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_len_x_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_len_x must have exactly one element");
            Py_DECREF(in_len_x_scalar_arr);
            return NULL;
        }
        in_len_x_scalar_is_array = 1;
        in_len_x = (int*)PyArray_DATA(in_len_x_scalar_arr);
        in_len_x_val = in_len_x[0];
        if (PyArray_DATA(in_len_x_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_len_x) || PyArray_TYPE(in_len_x_scalar_arr) \
            != PyArray_TYPE((PyArrayObject*)py_in_len_x)) {
            in_len_x_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_len_x)) {
        in_len_x_val = (int)PyLong_AsLong(py_in_len_x);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_len_x must be a scalar number or NumPy array");
        return NULL;
    }
    int* in_len_y = &in_len_y_val;
    if (PyArray_Check(py_in_len_y)) {
        in_len_y_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_len_y, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_len_y_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_len_y_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_len_y must have exactly one element");
            Py_DECREF(in_len_y_scalar_arr);
            return NULL;
        }
        in_len_y_scalar_is_array = 1;
        in_len_y = (int*)PyArray_DATA(in_len_y_scalar_arr);
        in_len_y_val = in_len_y[0];
        if (PyArray_DATA(in_len_y_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_len_y) || PyArray_TYPE(in_len_y_scalar_arr) \
            != PyArray_TYPE((PyArrayObject*)py_in_len_y)) {
            in_len_y_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_len_y)) {
        in_len_y_val = (int)PyLong_AsLong(py_in_len_y);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_len_y must be a scalar number or NumPy array");
        return NULL;
    }
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(f90wrap_n0_val);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n0_val = (int)ret_retval_dim_0;
    npy_intp ret_retval_dim_1 = (npy_intp)(f90wrap_n1_val);
    if (ret_retval_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_retval_dim_1;
    npy_intp ret_retval_dims[2] = {ret_retval_dim_0, ret_retval_dim_1};
    py_ret_retval_arr = PyArray_SimpleNew(2, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_input_2d)(&f90wrap_n0_val, &f90wrap_n1_val, in_len_x, in_len_y, \
        ret_retval);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    if (in_len_x_scalar_is_array) {
        if (in_len_x_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_len_x, in_len_x_scalar_arr) < 0) {
                Py_DECREF(in_len_x_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_len_x_scalar_arr);
    }
    if (in_len_y_scalar_is_array) {
        if (in_len_y_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_len_y, in_len_y_scalar_arr) < 0) {
                Py_DECREF(in_len_y_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_len_y_scalar_arr);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_array_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_in_array = NULL;
    static char *kwlist[] = {"f90wrap_n0", "f90wrap_n1", "in_array", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiO", kwlist, &f90wrap_n0_val, &f90wrap_n1_val, &py_in_array)) {
        return NULL;
    }
    
    PyArrayObject* in_array_arr = NULL;
    float* in_array = NULL;
    /* Extract in_array array data */
    if (!PyArray_Check(py_in_array)) {
        PyErr_SetString(PyExc_TypeError, "Argument in_array must be a NumPy array");
        return NULL;
    }
    in_array_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_in_array, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (in_array_arr == NULL) {
        return NULL;
    }
    in_array = (float*)PyArray_DATA(in_array_arr);
    int n0_in_array = (int)PyArray_DIM(in_array_arr, 0);
    f90wrap_n0_val = n0_in_array;
    
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(f90wrap_n1_val);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_retval_dim_0;
    npy_intp ret_retval_dims[1] = {ret_retval_dim_0};
    py_ret_retval_arr = PyArray_SimpleNew(1, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_size)(&f90wrap_n0_val, &f90wrap_n1_val, in_array, ret_retval);
    if (PyErr_Occurred()) {
        Py_XDECREF(in_array_arr);
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    Py_DECREF(in_array_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_array_size_2d_in(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    int f90wrap_n2_val = 0;
    PyObject* py_in_array = NULL;
    static char *kwlist[] = {"f90wrap_n0", "f90wrap_n1", "f90wrap_n2", "in_array", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiO", kwlist, &f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, \
        &py_in_array)) {
        return NULL;
    }
    
    PyArrayObject* in_array_arr = NULL;
    float* in_array = NULL;
    /* Extract in_array array data */
    if (!PyArray_Check(py_in_array)) {
        PyErr_SetString(PyExc_TypeError, "Argument in_array must be a NumPy array");
        return NULL;
    }
    in_array_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_in_array, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (in_array_arr == NULL) {
        return NULL;
    }
    in_array = (float*)PyArray_DATA(in_array_arr);
    int n0_in_array = (int)PyArray_DIM(in_array_arr, 0);
    int n1_in_array = (int)PyArray_DIM(in_array_arr, 1);
    f90wrap_n0_val = n0_in_array;
    f90wrap_n1_val = n1_in_array;
    
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(f90wrap_n2_val);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n2_val = (int)ret_retval_dim_0;
    npy_intp ret_retval_dims[1] = {ret_retval_dim_0};
    py_ret_retval_arr = PyArray_SimpleNew(1, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_size_2d_in)(&f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, in_array, \
        ret_retval);
    if (PyErr_Occurred()) {
        Py_XDECREF(in_array_arr);
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    Py_DECREF(in_array_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_array_size_2d_out(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    int f90wrap_n2_val = 0;
    int f90wrap_n3_val = 0;
    int f90wrap_n4_val = 0;
    int f90wrap_n5_val = 0;
    PyObject* py_in_array_1 = NULL;
    PyObject* py_in_array_2 = NULL;
    static char *kwlist[] = {"f90wrap_n0", "f90wrap_n1", "f90wrap_n2", "f90wrap_n3", "f90wrap_n4", "f90wrap_n5", \
        "in_array_1", "in_array_2", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiiiiOO", kwlist, &f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, \
        &f90wrap_n3_val, &f90wrap_n4_val, &f90wrap_n5_val, &py_in_array_1, &py_in_array_2)) {
        return NULL;
    }
    
    PyArrayObject* in_array_1_arr = NULL;
    float* in_array_1 = NULL;
    /* Extract in_array_1 array data */
    if (!PyArray_Check(py_in_array_1)) {
        PyErr_SetString(PyExc_TypeError, "Argument in_array_1 must be a NumPy array");
        return NULL;
    }
    in_array_1_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_in_array_1, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (in_array_1_arr == NULL) {
        return NULL;
    }
    in_array_1 = (float*)PyArray_DATA(in_array_1_arr);
    int n0_in_array_1 = (int)PyArray_DIM(in_array_1_arr, 0);
    int n1_in_array_1 = (int)PyArray_DIM(in_array_1_arr, 1);
    f90wrap_n0_val = n0_in_array_1;
    f90wrap_n1_val = n1_in_array_1;
    
    PyArrayObject* in_array_2_arr = NULL;
    float* in_array_2 = NULL;
    /* Extract in_array_2 array data */
    if (!PyArray_Check(py_in_array_2)) {
        PyErr_SetString(PyExc_TypeError, "Argument in_array_2 must be a NumPy array");
        return NULL;
    }
    in_array_2_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_in_array_2, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (in_array_2_arr == NULL) {
        return NULL;
    }
    in_array_2 = (float*)PyArray_DATA(in_array_2_arr);
    int n0_in_array_2 = (int)PyArray_DIM(in_array_2_arr, 0);
    int n1_in_array_2 = (int)PyArray_DIM(in_array_2_arr, 1);
    f90wrap_n2_val = n0_in_array_2;
    f90wrap_n3_val = n1_in_array_2;
    
    PyArrayObject* ret_retval_arr = NULL;
    PyObject* py_ret_retval_arr = NULL;
    float* ret_retval = NULL;
    npy_intp ret_retval_dim_0 = (npy_intp)(f90wrap_n4_val);
    if (ret_retval_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n4_val = (int)ret_retval_dim_0;
    npy_intp ret_retval_dim_1 = (npy_intp)(f90wrap_n5_val);
    if (ret_retval_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_retval must be positive");
        return NULL;
    }
    f90wrap_n5_val = (int)ret_retval_dim_1;
    npy_intp ret_retval_dims[2] = {ret_retval_dim_0, ret_retval_dim_1};
    py_ret_retval_arr = PyArray_SimpleNew(2, ret_retval_dims, NPY_FLOAT32);
    if (py_ret_retval_arr == NULL) {
        return NULL;
    }
    ret_retval_arr = (PyArrayObject*)py_ret_retval_arr;
    ret_retval = (float*)PyArray_DATA(ret_retval_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_array_size_2d_out)(&f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, \
        &f90wrap_n3_val, &f90wrap_n4_val, &f90wrap_n5_val, in_array_1, in_array_2, ret_retval);
    if (PyErr_Occurred()) {
        Py_XDECREF(in_array_1_arr);
        Py_XDECREF(in_array_2_arr);
        Py_XDECREF(py_ret_retval_arr);
        return NULL;
    }
    
    Py_DECREF(in_array_1_arr);
    Py_DECREF(in_array_2_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_retval_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_retval_arr != NULL) return py_ret_retval_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_retval_arr != NULL) Py_DECREF(py_ret_retval_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_retval_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_retval_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_return_derived_type_value(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_this = NULL;
    PyObject* py_size_2d = NULL;
    static char *kwlist[] = {"f90wrap_n0", "f90wrap_n1", "this", "size_2d", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiOO", kwlist, &f90wrap_n0_val, &f90wrap_n1_val, &py_this, \
        &py_size_2d)) {
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
    
    PyObject* size_2d_handle_obj = NULL;
    PyObject* size_2d_sequence = NULL;
    Py_ssize_t size_2d_handle_len = 0;
    if (PyObject_HasAttrString(py_size_2d, "_handle")) {
        size_2d_handle_obj = PyObject_GetAttrString(py_size_2d, "_handle");
        if (size_2d_handle_obj == NULL) {
            return NULL;
        }
        size_2d_sequence = PySequence_Fast(size_2d_handle_obj, "Failed to access handle sequence");
        if (size_2d_sequence == NULL) {
            Py_DECREF(size_2d_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_size_2d)) {
        size_2d_sequence = PySequence_Fast(py_size_2d, "Argument size_2d must be a handle sequence");
        if (size_2d_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument size_2d must be a Fortran derived-type instance");
        return NULL;
    }
    size_2d_handle_len = PySequence_Fast_GET_SIZE(size_2d_sequence);
    if (size_2d_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument size_2d has an invalid handle length");
        Py_DECREF(size_2d_sequence);
        if (size_2d_handle_obj) Py_DECREF(size_2d_handle_obj);
        return NULL;
    }
    int* size_2d = (int*)malloc(sizeof(int) * size_2d_handle_len);
    if (size_2d == NULL) {
        PyErr_NoMemory();
        Py_DECREF(size_2d_sequence);
        if (size_2d_handle_obj) Py_DECREF(size_2d_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < size_2d_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(size_2d_sequence, i);
        if (item == NULL) {
            free(size_2d);
            Py_DECREF(size_2d_sequence);
            if (size_2d_handle_obj) Py_DECREF(size_2d_handle_obj);
            return NULL;
        }
        size_2d[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(size_2d);
            Py_DECREF(size_2d_sequence);
            if (size_2d_handle_obj) Py_DECREF(size_2d_handle_obj);
            return NULL;
        }
    }
    (void)size_2d_handle_len;  /* suppress unused warnings when unchanged */
    
    PyArrayObject* ret_output_arr = NULL;
    PyObject* py_ret_output_arr = NULL;
    float* ret_output = NULL;
    npy_intp ret_output_dim_0 = (npy_intp)(f90wrap_n0_val);
    if (ret_output_dim_0 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_output must be positive");
        return NULL;
    }
    f90wrap_n0_val = (int)ret_output_dim_0;
    npy_intp ret_output_dim_1 = (npy_intp)(f90wrap_n1_val);
    if (ret_output_dim_1 <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dimension for ret_output must be positive");
        return NULL;
    }
    f90wrap_n1_val = (int)ret_output_dim_1;
    npy_intp ret_output_dims[2] = {ret_output_dim_0, ret_output_dim_1};
    py_ret_output_arr = PyArray_SimpleNew(2, ret_output_dims, NPY_FLOAT32);
    if (py_ret_output_arr == NULL) {
        return NULL;
    }
    ret_output_arr = (PyArrayObject*)py_ret_output_arr;
    ret_output = (float*)PyArray_DATA(ret_output_arr);
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__return_derived_type_value)(&f90wrap_n0_val, &f90wrap_n1_val, this, size_2d, \
        ret_output);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        if (size_2d_sequence) Py_DECREF(size_2d_sequence);
        if (size_2d_handle_obj) Py_DECREF(size_2d_handle_obj);
        free(size_2d);
        Py_XDECREF(py_ret_output_arr);
        return NULL;
    }
    
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    if (size_2d_sequence) {
        Py_DECREF(size_2d_sequence);
    }
    if (size_2d_handle_obj) {
        Py_DECREF(size_2d_handle_obj);
    }
    free(size_2d);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_arr != NULL) return py_ret_output_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_arr != NULL) Py_DECREF(py_ret_output_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_arr);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_t_array_wrapper_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper_initialise)(this);
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

static PyObject* wrap_m_test_t_array_wrapper_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test_t_array_2d_wrapper_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper_initialise)(this);
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

static PyObject* wrap_m_test_t_array_2d_wrapper_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test_t_array_double_wrapper_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_double_wrapper_initialise)(this);
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

static PyObject* wrap_m_test_t_array_double_wrapper_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_double_wrapper_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test_t_value_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_value_initialise)(this);
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

static PyObject* wrap_m_test_t_value_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_value_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test_t_size_2d_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d_initialise)(this);
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

static PyObject* wrap_m_test_t_size_2d_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test__t_array_wrapper_helper_get_a_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_handle)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value;
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper__get__a_size)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_test__t_array_wrapper_helper_set_a_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "a_size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_handle, &value)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper__set__a_size)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test__t_array_wrapper_helper_array_a_data(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_wrapper__array__a_data)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_m_test__t_array_2d_wrapper_helper_get_a_size_x(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_handle)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value;
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__get__a_size_x)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_test__t_array_2d_wrapper_helper_set_a_size_x(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "a_size_x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_handle, &value)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__set__a_size_x)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test__t_array_2d_wrapper_helper_get_a_size_y(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_handle)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value;
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__get__a_size_y)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_test__t_array_2d_wrapper_helper_set_a_size_y(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "a_size_y", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_handle, &value)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__set__a_size_y)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test__t_array_2d_wrapper_helper_array_a_data(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_2d_wrapper__array__a_data)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_m_test__t_array_double_wrapper_helper_get_derived_array_wrapper(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_handle)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int handle_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        handle_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value_handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_double_wrapper__get__array_wrapper)(handle_handle, value_handle);
    PyObject* result = PyList_New(4);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)value_handle[i]);
        if (item == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* wrap_m_test__t_array_double_wrapper_helper_set_derived_array_wrapper(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent = Py_None;
    PyObject* py_value = Py_None;
    static char *kwlist[] = {"handle", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_parent, &py_value)) {
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
    Py_DECREF(parent_sequence);
    int value_handle[4] = {0};
    PyObject* value_sequence = PySequence_Fast(py_value, "Value must be a sequence");
    if (value_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t value_len = PySequence_Fast_GET_SIZE(value_sequence);
    if (value_len != 4) {
        Py_DECREF(value_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);
        value_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(value_sequence);
            return NULL;
        }
    }
    Py_DECREF(value_sequence);
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_array_double_wrapper__set__array_wrapper)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test__t_value_helper_get_value(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_handle)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    float value;
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_value__get__value)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_test__t_value_helper_set_value(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od", kwlist, &py_handle, &value)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    float fortran_value = (float)value;
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_value__set__value)(this_handle, &fortran_value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test__t_size_2d_helper_get_x(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_handle)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value;
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d__get__x)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_test__t_size_2d_helper_set_x(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_handle, &value)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d__set__x)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_test__t_size_2d_helper_get_y(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_handle)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    int value;
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d__get__y)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_test__t_size_2d_helper_set_y(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "y", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_handle, &value)) {
        return NULL;
    }
    PyObject* handle_sequence = PySequence_Fast(py_handle, "Handle must be a sequence");
    if (handle_sequence == NULL) {
        return NULL;
    }
    Py_ssize_t handle_len = PySequence_Fast_GET_SIZE(handle_sequence);
    if (handle_len != 4) {
        Py_DECREF(handle_sequence);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    int this_handle[4] = {0};
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(handle_sequence, i);
        if (item == NULL) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
        this_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(handle_sequence);
            return NULL;
        }
    }
    Py_DECREF(handle_sequence);
    F90WRAP_F_SYMBOL(f90wrap_m_test__t_size_2d__set__y)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_test__array_init", (PyCFunction)wrap_m_test_array_init, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        array_init"},
    {"f90wrap_m_test__array_2d_init", (PyCFunction)wrap_m_test_array_2d_init, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        array_2d_init"},
    {"f90wrap_m_test__array_wrapper_init", (PyCFunction)wrap_m_test_array_wrapper_init, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for array_wrapper_init"},
    {"f90wrap_m_test__array_free", (PyCFunction)wrap_m_test_array_free, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        array_free"},
    {"f90wrap_m_test__return_scalar", (PyCFunction)wrap_m_test_return_scalar, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        return_scalar"},
    {"f90wrap_m_test__return_hard_coded_1d", (PyCFunction)wrap_m_test_return_hard_coded_1d, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for return_hard_coded_1d"},
    {"f90wrap_m_test__return_hard_coded_2d", (PyCFunction)wrap_m_test_return_hard_coded_2d, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for return_hard_coded_2d"},
    {"f90wrap_m_test__return_array_member", (PyCFunction)wrap_m_test_return_array_member, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for return_array_member"},
    {"f90wrap_m_test__return_array_member_2d", (PyCFunction)wrap_m_test_return_array_member_2d, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_array_member_2d"},
    {"f90wrap_m_test__return_array_member_wrapper", (PyCFunction)wrap_m_test_return_array_member_wrapper, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_array_member_wrapper"},
    {"f90wrap_m_test__return_array_input", (PyCFunction)wrap_m_test_return_array_input, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for return_array_input"},
    {"f90wrap_m_test__return_array_input_2d", (PyCFunction)wrap_m_test_return_array_input_2d, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for return_array_input_2d"},
    {"f90wrap_m_test__return_array_size", (PyCFunction)wrap_m_test_return_array_size, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for return_array_size"},
    {"f90wrap_m_test__return_array_size_2d_in", (PyCFunction)wrap_m_test_return_array_size_2d_in, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_array_size_2d_in"},
    {"f90wrap_m_test__return_array_size_2d_out", (PyCFunction)wrap_m_test_return_array_size_2d_out, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_array_size_2d_out"},
    {"f90wrap_m_test__return_derived_type_value", (PyCFunction)wrap_m_test_return_derived_type_value, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_derived_type_value"},
    {"f90wrap_m_test__t_array_wrapper_initialise", (PyCFunction)wrap_m_test_t_array_wrapper_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for t_array_wrapper"},
    {"f90wrap_m_test__t_array_wrapper_finalise", (PyCFunction)wrap_m_test_t_array_wrapper_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for t_array_wrapper"},
    {"f90wrap_m_test__t_array_2d_wrapper_initialise", (PyCFunction)wrap_m_test_t_array_2d_wrapper_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for t_array_2d_wrapper"},
    {"f90wrap_m_test__t_array_2d_wrapper_finalise", (PyCFunction)wrap_m_test_t_array_2d_wrapper_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for t_array_2d_wrapper"},
    {"f90wrap_m_test__t_array_double_wrapper_initialise", (PyCFunction)wrap_m_test_t_array_double_wrapper_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for t_array_double_wrapper"},
    {"f90wrap_m_test__t_array_double_wrapper_finalise", (PyCFunction)wrap_m_test_t_array_double_wrapper_finalise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated destructor for t_array_double_wrapper"},
    {"f90wrap_m_test__t_value_initialise", (PyCFunction)wrap_m_test_t_value_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for t_value"},
    {"f90wrap_m_test__t_value_finalise", (PyCFunction)wrap_m_test_t_value_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for t_value"},
    {"f90wrap_m_test__t_size_2d_initialise", (PyCFunction)wrap_m_test_t_size_2d_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for t_size_2d"},
    {"f90wrap_m_test__t_size_2d_finalise", (PyCFunction)wrap_m_test_t_size_2d_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for t_size_2d"},
    {"f90wrap_m_test__t_array_wrapper__get__a_size", (PyCFunction)wrap_m_test__t_array_wrapper_helper_get_a_size, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a_size"},
    {"f90wrap_m_test__t_array_wrapper__set__a_size", (PyCFunction)wrap_m_test__t_array_wrapper_helper_set_a_size, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a_size"},
    {"f90wrap_m_test__t_array_wrapper__array__a_data", (PyCFunction)wrap_m_test__t_array_wrapper_helper_array_a_data, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for a_data"},
    {"f90wrap_m_test__t_array_2d_wrapper__get__a_size_x", (PyCFunction)wrap_m_test__t_array_2d_wrapper_helper_get_a_size_x, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a_size_x"},
    {"f90wrap_m_test__t_array_2d_wrapper__set__a_size_x", (PyCFunction)wrap_m_test__t_array_2d_wrapper_helper_set_a_size_x, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a_size_x"},
    {"f90wrap_m_test__t_array_2d_wrapper__get__a_size_y", (PyCFunction)wrap_m_test__t_array_2d_wrapper_helper_get_a_size_y, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a_size_y"},
    {"f90wrap_m_test__t_array_2d_wrapper__set__a_size_y", (PyCFunction)wrap_m_test__t_array_2d_wrapper_helper_set_a_size_y, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a_size_y"},
    {"f90wrap_m_test__t_array_2d_wrapper__array__a_data", (PyCFunction)wrap_m_test__t_array_2d_wrapper_helper_array_a_data, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for a_data"},
    {"f90wrap_m_test__t_array_double_wrapper__get__array_wrapper", \
        (PyCFunction)wrap_m_test__t_array_double_wrapper_helper_get_derived_array_wrapper, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for array_wrapper"},
    {"f90wrap_m_test__t_array_double_wrapper__set__array_wrapper", \
        (PyCFunction)wrap_m_test__t_array_double_wrapper_helper_set_derived_array_wrapper, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for array_wrapper"},
    {"f90wrap_m_test__t_value__get__value", (PyCFunction)wrap_m_test__t_value_helper_get_value, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for value"},
    {"f90wrap_m_test__t_value__set__value", (PyCFunction)wrap_m_test__t_value_helper_set_value, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for value"},
    {"f90wrap_m_test__t_size_2d__get__x", (PyCFunction)wrap_m_test__t_size_2d_helper_get_x, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for x"},
    {"f90wrap_m_test__t_size_2d__set__x", (PyCFunction)wrap_m_test__t_size_2d_helper_set_x, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for x"},
    {"f90wrap_m_test__t_size_2d__get__y", (PyCFunction)wrap_m_test__t_size_2d_helper_get_y, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for y"},
    {"f90wrap_m_test__t_size_2d__set__y", (PyCFunction)wrap_m_test__t_size_2d_helper_set_y, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for y"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _pywrappermodule = {
    PyModuleDef_HEAD_INIT,
    "pywrapper",
    "Direct-C wrapper for _pywrapper module",
    -1,
    _pywrapper_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__pywrapper(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_pywrappermodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
