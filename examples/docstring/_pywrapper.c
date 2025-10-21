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
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__construct_circle)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__construct_circle_more_doc)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__no_direction)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__incomplete_doc_sub)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__doc_inside)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__output_1)(float* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__function_2)(char* input, int* ret_function_2, int input_len);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__details_doc)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__details_with_parenthesis)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__multiline_details)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__empty_lines_details)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__long_line_brief)(int* circle, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__t_circle_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__t_circle_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__t_circle__get__radius)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_circle__t_circle__set__radius)(int* handle, float* value);

static PyObject* wrap_m_circle_construct_circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__construct_circle)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_construct_circle_more_doc(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__construct_circle_more_doc)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_no_direction(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__no_direction)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_incomplete_doc_sub(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__incomplete_doc_sub)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_doc_inside(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__doc_inside)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_output_1(PyObject* self, PyObject* args, PyObject* kwargs)
{
    float output_val = 0;
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__output_1)(&output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_output_obj = Py_BuildValue("d", output_val);
    if (py_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_output_obj != NULL) return py_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_output_obj != NULL) Py_DECREF(py_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_circle_function_2(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    int ret_function_2_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    int input_len = 0;
    char* input = NULL;
    int input_is_array = 0;
    if (py_input == Py_None) {
        PyErr_SetString(PyExc_TypeError, "Argument input cannot be None");
        return NULL;
    } else {
        PyObject* input_bytes = NULL;
        if (PyArray_Check(py_input)) {
            /* Handle numpy array - extract buffer for in-place modification */
            PyArrayObject* input_arr = (PyArrayObject*)py_input;
            if (PyArray_TYPE(input_arr) != NPY_STRING) {
                PyErr_SetString(PyExc_TypeError, "Argument input must be a string array");
                return NULL;
            }
            input_len = (int)PyArray_ITEMSIZE(input_arr);
            input = (char*)PyArray_DATA(input_arr);
            input_is_array = 1;
        } else if (PyBytes_Check(py_input)) {
            input_bytes = py_input;
            Py_INCREF(input_bytes);
        } else if (PyUnicode_Check(py_input)) {
            input_bytes = PyUnicode_AsUTF8String(py_input);
            if (input_bytes == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument input must be str, bytes, or numpy array");
            return NULL;
        }
        if (input_bytes != NULL) {
            input_len = (int)PyBytes_GET_SIZE(input_bytes);
            input = (char*)malloc((size_t)input_len + 1);
            if (input == NULL) {
                Py_DECREF(input_bytes);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(input, PyBytes_AS_STRING(input_bytes), (size_t)input_len);
            input[input_len] = '\0';
            Py_DECREF(input_bytes);
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__function_2)(input, &ret_function_2_val, input_len);
    if (PyErr_Occurred()) {
        if (!input_is_array) free(input);
        return NULL;
    }
    
    PyObject* py_ret_function_2_obj = Py_BuildValue("i", ret_function_2_val);
    if (py_ret_function_2_obj == NULL) {
        return NULL;
    }
    if (!input_is_array) free(input);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_function_2_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_function_2_obj != NULL) return py_ret_function_2_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_function_2_obj != NULL) Py_DECREF(py_ret_function_2_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_function_2_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_function_2_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_circle_details_doc(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__details_doc)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_details_with_parenthesis(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__details_with_parenthesis)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_multiline_details(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__multiline_details)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_empty_lines_details(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__empty_lines_details)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_long_line_brief(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_circle = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"circle", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_radius)) {
        return NULL;
    }
    
    PyObject* circle_handle_obj = NULL;
    PyObject* circle_sequence = NULL;
    Py_ssize_t circle_handle_len = 0;
    if (PyObject_HasAttrString(py_circle, "_handle")) {
        circle_handle_obj = PyObject_GetAttrString(py_circle, "_handle");
        if (circle_handle_obj == NULL) {
            return NULL;
        }
        circle_sequence = PySequence_Fast(circle_handle_obj, "Failed to access handle sequence");
        if (circle_sequence == NULL) {
            Py_DECREF(circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_circle)) {
        circle_sequence = PySequence_Fast(py_circle, "Argument circle must be a handle sequence");
        if (circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument circle must be a Fortran derived-type instance");
        return NULL;
    }
    circle_handle_len = PySequence_Fast_GET_SIZE(circle_sequence);
    if (circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument circle has an invalid handle length");
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    int* circle = (int*)malloc(sizeof(int) * circle_handle_len);
    if (circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(circle_sequence, i);
        if (item == NULL) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
        circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(circle);
            Py_DECREF(circle_sequence);
            if (circle_handle_obj) Py_DECREF(circle_handle_obj);
            return NULL;
        }
    }
    (void)circle_handle_len;  /* suppress unused warnings when unchanged */
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__long_line_brief)(circle, radius);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle_t_circle_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_circle__t_circle_initialise)(this);
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

static PyObject* wrap_m_circle_t_circle_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_circle__t_circle_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_circle__t_circle_helper_get_radius(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_circle__t_circle__get__radius)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_circle__t_circle_helper_set_radius(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "radius", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_circle__t_circle__set__radius)(this_handle, &fortran_value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_circle__construct_circle", (PyCFunction)wrap_m_circle_construct_circle, METH_VARARGS | METH_KEYWORDS, \
        "Initialize circle\n"},
    {"f90wrap_m_circle__construct_circle_more_doc", (PyCFunction)wrap_m_circle_construct_circle_more_doc, METH_VARARGS | \
        METH_KEYWORDS, "Initialize circle with more doc\n"},
    {"f90wrap_m_circle__no_direction", (PyCFunction)wrap_m_circle_no_direction, METH_VARARGS | METH_KEYWORDS, "Without \
        direction\n"},
    {"f90wrap_m_circle__incomplete_doc_sub", (PyCFunction)wrap_m_circle_incomplete_doc_sub, METH_VARARGS | METH_KEYWORDS, \
        "Incomplete doc\n"},
    {"f90wrap_m_circle__doc_inside", (PyCFunction)wrap_m_circle_doc_inside, METH_VARARGS | METH_KEYWORDS, \
        "==========================================================================="},
    {"f90wrap_m_circle__output_1", (PyCFunction)wrap_m_circle_output_1, METH_VARARGS | METH_KEYWORDS, "subroutine output_1 \
        outputs 1\n"},
    {"f90wrap_m_circle__function_2", (PyCFunction)wrap_m_circle_function_2, METH_VARARGS | METH_KEYWORDS, "this is a \
        function\n"},
    {"f90wrap_m_circle__details_doc", (PyCFunction)wrap_m_circle_details_doc, METH_VARARGS | METH_KEYWORDS, "Initialize \
        circle\n"},
    {"f90wrap_m_circle__details_with_parenthesis", (PyCFunction)wrap_m_circle_details_with_parenthesis, METH_VARARGS | \
        METH_KEYWORDS, "Initialize circle\n"},
    {"f90wrap_m_circle__multiline_details", (PyCFunction)wrap_m_circle_multiline_details, METH_VARARGS | METH_KEYWORDS, \
        "Initialize circle\n"},
    {"f90wrap_m_circle__empty_lines_details", (PyCFunction)wrap_m_circle_empty_lines_details, METH_VARARGS | METH_KEYWORDS, \
        "Initialize circle\n"},
    {"f90wrap_m_circle__long_line_brief", (PyCFunction)wrap_m_circle_long_line_brief, METH_VARARGS | METH_KEYWORDS, "This is \
        a very long brief that takes up a lot of space and contains lots of information, it should probably be wrapped to \
        the next line, but we will continue regardless\n"},
    {"f90wrap_m_circle__t_circle_initialise", (PyCFunction)wrap_m_circle_t_circle_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for t_circle"},
    {"f90wrap_m_circle__t_circle_finalise", (PyCFunction)wrap_m_circle_t_circle_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for t_circle"},
    {"f90wrap_m_circle__t_circle__get__radius", (PyCFunction)wrap_m_circle__t_circle_helper_get_radius, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for radius"},
    {"f90wrap_m_circle__t_circle__set__radius", (PyCFunction)wrap_m_circle__t_circle_helper_set_radius, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for radius"},
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
