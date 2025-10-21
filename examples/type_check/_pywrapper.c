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
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__is_circle_circle)(int* f90wrap_n0, int* circle, int* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__is_circle_square)(int* f90wrap_n0, int* square, int* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_int32_0d)(int* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_int64_0d)(long long* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_real32_0d)(float* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_real64_0d)(double* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_int_1d)(int* f90wrap_n0, int* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_int_2d)(int* f90wrap_n0, int* f90wrap_n1, int* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_real)(int* f90wrap_n0, float* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_double)(int* f90wrap_n0, double* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_bool)(int* f90wrap_n0, int* output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__optional_scalar_real)(int* f90wrap_n0, float* output, float* \
    opt_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__optional_scalar_int)(int* f90wrap_n0, int* output, int* opt_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_int8)(short* input, int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_int16)(short* input, int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_int32)(int* input, int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_int64)(long long* input, int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_real32)(float* input, int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_real64)(double* input, int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_array_int64)(int* f90wrap_n0, long long* input, int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_array_real64)(int* f90wrap_n0, double* input, int* ret_output);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_square_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_square_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_circle_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_circle_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_square__get__length)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_square__set__length)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_circle__get__radius)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_circle__set__radius)(int* handle, float* value);

static PyObject* wrap_m_type_test_is_circle_circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_circle = NULL;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"circle", "output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_circle, &py_output)) {
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
    
    PyArrayObject* output_arr = NULL;
    int* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (int*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n0_val = n0_output;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__is_circle_circle)(&f90wrap_n0_val, circle, output);
    if (PyErr_Occurred()) {
        if (circle_sequence) Py_DECREF(circle_sequence);
        if (circle_handle_obj) Py_DECREF(circle_handle_obj);
        free(circle);
        Py_XDECREF(output_arr);
        return NULL;
    }
    
    Py_DECREF(output_arr);
    if (circle_sequence) {
        Py_DECREF(circle_sequence);
    }
    if (circle_handle_obj) {
        Py_DECREF(circle_handle_obj);
    }
    free(circle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test_is_circle_square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_square = NULL;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"square", "output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_square, &py_output)) {
        return NULL;
    }
    
    PyObject* square_handle_obj = NULL;
    PyObject* square_sequence = NULL;
    Py_ssize_t square_handle_len = 0;
    if (PyObject_HasAttrString(py_square, "_handle")) {
        square_handle_obj = PyObject_GetAttrString(py_square, "_handle");
        if (square_handle_obj == NULL) {
            return NULL;
        }
        square_sequence = PySequence_Fast(square_handle_obj, "Failed to access handle sequence");
        if (square_sequence == NULL) {
            Py_DECREF(square_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_square)) {
        square_sequence = PySequence_Fast(py_square, "Argument square must be a handle sequence");
        if (square_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument square must be a Fortran derived-type instance");
        return NULL;
    }
    square_handle_len = PySequence_Fast_GET_SIZE(square_sequence);
    if (square_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument square has an invalid handle length");
        Py_DECREF(square_sequence);
        if (square_handle_obj) Py_DECREF(square_handle_obj);
        return NULL;
    }
    int* square = (int*)malloc(sizeof(int) * square_handle_len);
    if (square == NULL) {
        PyErr_NoMemory();
        Py_DECREF(square_sequence);
        if (square_handle_obj) Py_DECREF(square_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < square_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(square_sequence, i);
        if (item == NULL) {
            free(square);
            Py_DECREF(square_sequence);
            if (square_handle_obj) Py_DECREF(square_handle_obj);
            return NULL;
        }
        square[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(square);
            Py_DECREF(square_sequence);
            if (square_handle_obj) Py_DECREF(square_handle_obj);
            return NULL;
        }
    }
    (void)square_handle_len;  /* suppress unused warnings when unchanged */
    
    PyArrayObject* output_arr = NULL;
    int* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (int*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n0_val = n0_output;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__is_circle_square)(&f90wrap_n0_val, square, output);
    if (PyErr_Occurred()) {
        if (square_sequence) Py_DECREF(square_sequence);
        if (square_handle_obj) Py_DECREF(square_handle_obj);
        free(square);
        Py_XDECREF(output_arr);
        return NULL;
    }
    
    Py_DECREF(output_arr);
    if (square_sequence) {
        Py_DECREF(square_sequence);
    }
    if (square_handle_obj) {
        Py_DECREF(square_handle_obj);
    }
    free(square);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test_write_array_int32_0d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_output = NULL;
    int output_val = 0;
    PyArrayObject* output_scalar_arr = NULL;
    int output_scalar_copyback = 0;
    int output_scalar_is_array = 0;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    int* output = &output_val;
    if (PyArray_Check(py_output)) {
        output_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_output, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (output_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(output_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument output must have exactly one element");
            Py_DECREF(output_scalar_arr);
            return NULL;
        }
        output_scalar_is_array = 1;
        output = (int*)PyArray_DATA(output_scalar_arr);
        output_val = output[0];
        if (PyArray_DATA(output_scalar_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_output)) {
            output_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_output)) {
        output_val = (int)PyLong_AsLong(py_output);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_int32_0d)(output);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (output_scalar_is_array) {
        if (output_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_output, output_scalar_arr) < 0) {
                Py_DECREF(output_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(output_scalar_arr);
    }
    PyObject* py_output_obj = Py_BuildValue("i", output_val);
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

static PyObject* wrap_m_type_test_write_array_int64_0d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_output = NULL;
    long long output_val = 0;
    PyArrayObject* output_scalar_arr = NULL;
    int output_scalar_copyback = 0;
    int output_scalar_is_array = 0;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    long long* output = &output_val;
    if (PyArray_Check(py_output)) {
        output_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_output, NPY_INT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (output_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(output_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument output must have exactly one element");
            Py_DECREF(output_scalar_arr);
            return NULL;
        }
        output_scalar_is_array = 1;
        output = (long long*)PyArray_DATA(output_scalar_arr);
        output_val = output[0];
        if (PyArray_DATA(output_scalar_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_output)) {
            output_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_output)) {
        output_val = (long long)PyLong_AsLong(py_output);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_int64_0d)(output);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (output_scalar_is_array) {
        if (output_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_output, output_scalar_arr) < 0) {
                Py_DECREF(output_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(output_scalar_arr);
    }
    PyObject* py_output_obj = Py_BuildValue("i", output_val);
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

static PyObject* wrap_m_type_test_write_array_real32_0d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_output = NULL;
    float output_val = 0;
    PyArrayObject* output_scalar_arr = NULL;
    int output_scalar_copyback = 0;
    int output_scalar_is_array = 0;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    float* output = &output_val;
    if (PyArray_Check(py_output)) {
        output_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_output, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (output_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(output_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument output must have exactly one element");
            Py_DECREF(output_scalar_arr);
            return NULL;
        }
        output_scalar_is_array = 1;
        output = (float*)PyArray_DATA(output_scalar_arr);
        output_val = output[0];
        if (PyArray_DATA(output_scalar_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_output)) {
            output_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_output)) {
        output_val = (float)PyFloat_AsDouble(py_output);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_real32_0d)(output);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (output_scalar_is_array) {
        if (output_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_output, output_scalar_arr) < 0) {
                Py_DECREF(output_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(output_scalar_arr);
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

static PyObject* wrap_m_type_test_write_array_real64_0d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_output = NULL;
    double output_val = 0;
    PyArrayObject* output_scalar_arr = NULL;
    int output_scalar_copyback = 0;
    int output_scalar_is_array = 0;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    double* output = &output_val;
    if (PyArray_Check(py_output)) {
        output_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_output, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (output_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(output_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument output must have exactly one element");
            Py_DECREF(output_scalar_arr);
            return NULL;
        }
        output_scalar_is_array = 1;
        output = (double*)PyArray_DATA(output_scalar_arr);
        output_val = output[0];
        if (PyArray_DATA(output_scalar_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_output)) {
            output_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_output)) {
        output_val = (double)PyFloat_AsDouble(py_output);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_real64_0d)(output);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (output_scalar_is_array) {
        if (output_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_output, output_scalar_arr) < 0) {
                Py_DECREF(output_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(output_scalar_arr);
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

static PyObject* wrap_m_type_test_write_array_int_1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    PyArrayObject* output_arr = NULL;
    int* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (int*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n0_val = n0_output;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_int_1d)(&f90wrap_n0_val, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(output_arr);
        return NULL;
    }
    
    Py_DECREF(output_arr);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test_write_array_int_2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    PyArrayObject* output_arr = NULL;
    int* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (int*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    int n1_output = (int)PyArray_DIM(output_arr, 1);
    f90wrap_n0_val = n0_output;
    f90wrap_n1_val = n1_output;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_int_2d)(&f90wrap_n0_val, &f90wrap_n1_val, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(output_arr);
        return NULL;
    }
    
    Py_DECREF(output_arr);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test_write_array_real(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    PyArrayObject* output_arr = NULL;
    float* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (float*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n0_val = n0_output;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_real)(&f90wrap_n0_val, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(output_arr);
        return NULL;
    }
    
    Py_DECREF(output_arr);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test_write_array_double(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    PyArrayObject* output_arr = NULL;
    double* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (double*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n0_val = n0_output;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_double)(&f90wrap_n0_val, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(output_arr);
        return NULL;
    }
    
    Py_DECREF(output_arr);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test_write_array_bool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_output = NULL;
    static char *kwlist[] = {"output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_output)) {
        return NULL;
    }
    
    PyArrayObject* output_arr = NULL;
    int* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_BOOL, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (int*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n0_val = n0_output;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__write_array_bool)(&f90wrap_n0_val, output);
    if (PyErr_Occurred()) {
        Py_XDECREF(output_arr);
        return NULL;
    }
    
    Py_DECREF(output_arr);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test_optional_scalar_real(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_output = NULL;
    PyObject* py_opt_output = Py_None;
    float opt_output_val = 0;
    PyArrayObject* opt_output_scalar_arr = NULL;
    int opt_output_scalar_copyback = 0;
    int opt_output_scalar_is_array = 0;
    static char *kwlist[] = {"output", "opt_output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &py_output, &py_opt_output)) {
        return NULL;
    }
    
    PyArrayObject* output_arr = NULL;
    PyObject* py_output_arr = NULL;
    int output_needs_copyback = 0;
    float* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (float*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n0_val = n0_output;
    Py_INCREF(py_output);
    py_output_arr = py_output;
    if (PyArray_DATA(output_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_arr) != \
        PyArray_TYPE((PyArrayObject*)py_output)) {
        output_needs_copyback = 1;
    }
    
    float* opt_output = &opt_output_val;
    if (py_opt_output == Py_None) {
        opt_output_val = 0;
    } else {
        if (PyArray_Check(py_opt_output)) {
            opt_output_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
                py_opt_output, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
            if (opt_output_scalar_arr == NULL) {
                return NULL;
            }
            if (PyArray_SIZE(opt_output_scalar_arr) != 1) {
                PyErr_SetString(PyExc_ValueError, "Argument opt_output must have exactly one element");
                Py_DECREF(opt_output_scalar_arr);
                return NULL;
            }
            opt_output_scalar_is_array = 1;
            opt_output = (float*)PyArray_DATA(opt_output_scalar_arr);
            opt_output_val = opt_output[0];
            if (PyArray_DATA(opt_output_scalar_arr) != PyArray_DATA((PyArrayObject*)py_opt_output) || \
                PyArray_TYPE(opt_output_scalar_arr) != PyArray_TYPE((PyArrayObject*)py_opt_output)) {
                opt_output_scalar_copyback = 1;
            }
        } else if (PyNumber_Check(py_opt_output)) {
            opt_output_val = (float)PyFloat_AsDouble(py_opt_output);
            if (PyErr_Occurred()) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument opt_output must be a scalar number or NumPy array");
            return NULL;
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__optional_scalar_real)(&f90wrap_n0_val, output, opt_output);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_output_arr);
        return NULL;
    }
    
    if (opt_output_scalar_is_array) {
        if (opt_output_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_opt_output, opt_output_scalar_arr) < 0) {
                Py_DECREF(opt_output_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(opt_output_scalar_arr);
    }
    if (output_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_output, output_arr) < 0) {
            Py_DECREF(output_arr);
            Py_DECREF(py_output_arr);
            return NULL;
        }
    }
    Py_DECREF(output_arr);
    PyObject* py_opt_output_obj = Py_BuildValue("d", opt_output_val);
    if (py_opt_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_output_arr != NULL) result_count++;
    if (py_opt_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_output_arr != NULL) return py_output_arr;
        if (py_opt_output_obj != NULL) return py_opt_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_output_arr != NULL) Py_DECREF(py_output_arr);
        if (py_opt_output_obj != NULL) Py_DECREF(py_opt_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_output_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_output_arr);
    }
    if (py_opt_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_opt_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_optional_scalar_int(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_output = NULL;
    PyObject* py_opt_output = Py_None;
    int opt_output_val = 0;
    PyArrayObject* opt_output_scalar_arr = NULL;
    int opt_output_scalar_copyback = 0;
    int opt_output_scalar_is_array = 0;
    static char *kwlist[] = {"output", "opt_output", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &py_output, &py_opt_output)) {
        return NULL;
    }
    
    PyArrayObject* output_arr = NULL;
    PyObject* py_output_arr = NULL;
    int output_needs_copyback = 0;
    int* output = NULL;
    /* Extract output array data */
    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_TypeError, "Argument output must be a NumPy array");
        return NULL;
    }
    output_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_output, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (output_arr == NULL) {
        return NULL;
    }
    output = (int*)PyArray_DATA(output_arr);
    int n0_output = (int)PyArray_DIM(output_arr, 0);
    f90wrap_n0_val = n0_output;
    Py_INCREF(py_output);
    py_output_arr = py_output;
    if (PyArray_DATA(output_arr) != PyArray_DATA((PyArrayObject*)py_output) || PyArray_TYPE(output_arr) != \
        PyArray_TYPE((PyArrayObject*)py_output)) {
        output_needs_copyback = 1;
    }
    
    int* opt_output = &opt_output_val;
    if (py_opt_output == Py_None) {
        opt_output_val = 0;
    } else {
        if (PyArray_Check(py_opt_output)) {
            opt_output_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
                py_opt_output, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
            if (opt_output_scalar_arr == NULL) {
                return NULL;
            }
            if (PyArray_SIZE(opt_output_scalar_arr) != 1) {
                PyErr_SetString(PyExc_ValueError, "Argument opt_output must have exactly one element");
                Py_DECREF(opt_output_scalar_arr);
                return NULL;
            }
            opt_output_scalar_is_array = 1;
            opt_output = (int*)PyArray_DATA(opt_output_scalar_arr);
            opt_output_val = opt_output[0];
            if (PyArray_DATA(opt_output_scalar_arr) != PyArray_DATA((PyArrayObject*)py_opt_output) || \
                PyArray_TYPE(opt_output_scalar_arr) != PyArray_TYPE((PyArrayObject*)py_opt_output)) {
                opt_output_scalar_copyback = 1;
            }
        } else if (PyNumber_Check(py_opt_output)) {
            opt_output_val = (int)PyLong_AsLong(py_opt_output);
            if (PyErr_Occurred()) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument opt_output must be a scalar number or NumPy array");
            return NULL;
        }
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__optional_scalar_int)(&f90wrap_n0_val, output, opt_output);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_output_arr);
        return NULL;
    }
    
    if (opt_output_scalar_is_array) {
        if (opt_output_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_opt_output, opt_output_scalar_arr) < 0) {
                Py_DECREF(opt_output_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(opt_output_scalar_arr);
    }
    if (output_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_output, output_arr) < 0) {
            Py_DECREF(output_arr);
            Py_DECREF(py_output_arr);
            return NULL;
        }
    }
    Py_DECREF(output_arr);
    PyObject* py_opt_output_obj = Py_BuildValue("i", opt_output_val);
    if (py_opt_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_output_arr != NULL) result_count++;
    if (py_opt_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_output_arr != NULL) return py_output_arr;
        if (py_opt_output_obj != NULL) return py_opt_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_output_arr != NULL) Py_DECREF(py_output_arr);
        if (py_opt_output_obj != NULL) Py_DECREF(py_opt_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_output_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_output_arr);
    }
    if (py_opt_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_opt_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_in_scalar_int8(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    short input_val = 0;
    PyArrayObject* input_scalar_arr = NULL;
    int input_scalar_copyback = 0;
    int input_scalar_is_array = 0;
    int ret_output_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    short* input = &input_val;
    if (PyArray_Check(py_input)) {
        input_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_input, NPY_INT16, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (input_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(input_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument input must have exactly one element");
            Py_DECREF(input_scalar_arr);
            return NULL;
        }
        input_scalar_is_array = 1;
        input = (short*)PyArray_DATA(input_scalar_arr);
        input_val = input[0];
        if (PyArray_DATA(input_scalar_arr) != PyArray_DATA((PyArrayObject*)py_input) || PyArray_TYPE(input_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_input)) {
            input_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_input)) {
        input_val = (short)PyLong_AsLong(py_input);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_int8)(input, &ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (input_scalar_is_array) {
        if (input_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_input, input_scalar_arr) < 0) {
                Py_DECREF(input_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(input_scalar_arr);
    }
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_in_scalar_int16(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    short input_val = 0;
    PyArrayObject* input_scalar_arr = NULL;
    int input_scalar_copyback = 0;
    int input_scalar_is_array = 0;
    int ret_output_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    short* input = &input_val;
    if (PyArray_Check(py_input)) {
        input_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_input, NPY_INT16, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (input_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(input_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument input must have exactly one element");
            Py_DECREF(input_scalar_arr);
            return NULL;
        }
        input_scalar_is_array = 1;
        input = (short*)PyArray_DATA(input_scalar_arr);
        input_val = input[0];
        if (PyArray_DATA(input_scalar_arr) != PyArray_DATA((PyArrayObject*)py_input) || PyArray_TYPE(input_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_input)) {
            input_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_input)) {
        input_val = (short)PyLong_AsLong(py_input);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_int16)(input, &ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (input_scalar_is_array) {
        if (input_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_input, input_scalar_arr) < 0) {
                Py_DECREF(input_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(input_scalar_arr);
    }
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_in_scalar_int32(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    int input_val = 0;
    PyArrayObject* input_scalar_arr = NULL;
    int input_scalar_copyback = 0;
    int input_scalar_is_array = 0;
    int ret_output_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    int* input = &input_val;
    if (PyArray_Check(py_input)) {
        input_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_input, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (input_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(input_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument input must have exactly one element");
            Py_DECREF(input_scalar_arr);
            return NULL;
        }
        input_scalar_is_array = 1;
        input = (int*)PyArray_DATA(input_scalar_arr);
        input_val = input[0];
        if (PyArray_DATA(input_scalar_arr) != PyArray_DATA((PyArrayObject*)py_input) || PyArray_TYPE(input_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_input)) {
            input_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_input)) {
        input_val = (int)PyLong_AsLong(py_input);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_int32)(input, &ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (input_scalar_is_array) {
        if (input_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_input, input_scalar_arr) < 0) {
                Py_DECREF(input_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(input_scalar_arr);
    }
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_in_scalar_int64(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    long long input_val = 0;
    PyArrayObject* input_scalar_arr = NULL;
    int input_scalar_copyback = 0;
    int input_scalar_is_array = 0;
    int ret_output_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    long long* input = &input_val;
    if (PyArray_Check(py_input)) {
        input_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_input, NPY_INT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (input_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(input_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument input must have exactly one element");
            Py_DECREF(input_scalar_arr);
            return NULL;
        }
        input_scalar_is_array = 1;
        input = (long long*)PyArray_DATA(input_scalar_arr);
        input_val = input[0];
        if (PyArray_DATA(input_scalar_arr) != PyArray_DATA((PyArrayObject*)py_input) || PyArray_TYPE(input_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_input)) {
            input_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_input)) {
        input_val = (long long)PyLong_AsLong(py_input);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_int64)(input, &ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (input_scalar_is_array) {
        if (input_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_input, input_scalar_arr) < 0) {
                Py_DECREF(input_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(input_scalar_arr);
    }
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_in_scalar_real32(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    float input_val = 0;
    PyArrayObject* input_scalar_arr = NULL;
    int input_scalar_copyback = 0;
    int input_scalar_is_array = 0;
    int ret_output_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    float* input = &input_val;
    if (PyArray_Check(py_input)) {
        input_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_input, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (input_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(input_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument input must have exactly one element");
            Py_DECREF(input_scalar_arr);
            return NULL;
        }
        input_scalar_is_array = 1;
        input = (float*)PyArray_DATA(input_scalar_arr);
        input_val = input[0];
        if (PyArray_DATA(input_scalar_arr) != PyArray_DATA((PyArrayObject*)py_input) || PyArray_TYPE(input_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_input)) {
            input_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_input)) {
        input_val = (float)PyFloat_AsDouble(py_input);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_real32)(input, &ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (input_scalar_is_array) {
        if (input_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_input, input_scalar_arr) < 0) {
                Py_DECREF(input_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(input_scalar_arr);
    }
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_in_scalar_real64(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_input = NULL;
    double input_val = 0;
    PyArrayObject* input_scalar_arr = NULL;
    int input_scalar_copyback = 0;
    int input_scalar_is_array = 0;
    int ret_output_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    double* input = &input_val;
    if (PyArray_Check(py_input)) {
        input_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_input, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (input_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(input_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument input must have exactly one element");
            Py_DECREF(input_scalar_arr);
            return NULL;
        }
        input_scalar_is_array = 1;
        input = (double*)PyArray_DATA(input_scalar_arr);
        input_val = input[0];
        if (PyArray_DATA(input_scalar_arr) != PyArray_DATA((PyArrayObject*)py_input) || PyArray_TYPE(input_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_input)) {
            input_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_input)) {
        input_val = (double)PyFloat_AsDouble(py_input);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_scalar_real64)(input, &ret_output_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (input_scalar_is_array) {
        if (input_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_input, input_scalar_arr) < 0) {
                Py_DECREF(input_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(input_scalar_arr);
    }
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_in_array_int64(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_input = NULL;
    int ret_output_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    PyArrayObject* input_arr = NULL;
    long long* input = NULL;
    /* Extract input array data */
    if (!PyArray_Check(py_input)) {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a NumPy array");
        return NULL;
    }
    input_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_input, NPY_INT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (input_arr == NULL) {
        return NULL;
    }
    input = (long long*)PyArray_DATA(input_arr);
    int n0_input = (int)PyArray_DIM(input_arr, 0);
    f90wrap_n0_val = n0_input;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_array_int64)(&f90wrap_n0_val, input, &ret_output_val);
    if (PyErr_Occurred()) {
        Py_XDECREF(input_arr);
        return NULL;
    }
    
    Py_DECREF(input_arr);
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_in_array_real64(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_input = NULL;
    int ret_output_val = 0;
    static char *kwlist[] = {"input", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_input)) {
        return NULL;
    }
    
    PyArrayObject* input_arr = NULL;
    double* input = NULL;
    /* Extract input array data */
    if (!PyArray_Check(py_input)) {
        PyErr_SetString(PyExc_TypeError, "Argument input must be a NumPy array");
        return NULL;
    }
    input_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_input, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (input_arr == NULL) {
        return NULL;
    }
    input = (double*)PyArray_DATA(input_arr);
    int n0_input = (int)PyArray_DIM(input_arr, 0);
    f90wrap_n0_val = n0_input;
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__in_array_real64)(&f90wrap_n0_val, input, &ret_output_val);
    if (PyErr_Occurred()) {
        Py_XDECREF(input_arr);
        return NULL;
    }
    
    Py_DECREF(input_arr);
    PyObject* py_ret_output_obj = Py_BuildValue("i", ret_output_val);
    if (py_ret_output_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_output_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_output_obj != NULL) return py_ret_output_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_output_obj != NULL) Py_DECREF(py_ret_output_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_output_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_output_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_type_test_t_square_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_square_initialise)(this);
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

static PyObject* wrap_m_type_test_t_square_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_square_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test_t_circle_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_circle_initialise)(this);
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

static PyObject* wrap_m_type_test_t_circle_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_circle_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test__t_square_helper_get_length(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_square__get__length)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_type_test__t_square_helper_set_length(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "length", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_square__set__length)(this_handle, &fortran_value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_type_test__t_circle_helper_get_radius(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_circle__get__radius)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_type_test__t_circle_helper_set_radius(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_type_test__t_circle__set__radius)(this_handle, &fortran_value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_type_test__is_circle_circle", (PyCFunction)wrap_m_type_test_is_circle_circle, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for is_circle_circle"},
    {"f90wrap_m_type_test__is_circle_square", (PyCFunction)wrap_m_type_test_is_circle_square, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for is_circle_square"},
    {"f90wrap_m_type_test__write_array_int32_0d", (PyCFunction)wrap_m_type_test_write_array_int32_0d, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for write_array_int32_0d"},
    {"f90wrap_m_type_test__write_array_int64_0d", (PyCFunction)wrap_m_type_test_write_array_int64_0d, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for write_array_int64_0d"},
    {"f90wrap_m_type_test__write_array_real32_0d", (PyCFunction)wrap_m_type_test_write_array_real32_0d, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for write_array_real32_0d"},
    {"f90wrap_m_type_test__write_array_real64_0d", (PyCFunction)wrap_m_type_test_write_array_real64_0d, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for write_array_real64_0d"},
    {"f90wrap_m_type_test__write_array_int_1d", (PyCFunction)wrap_m_type_test_write_array_int_1d, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for write_array_int_1d"},
    {"f90wrap_m_type_test__write_array_int_2d", (PyCFunction)wrap_m_type_test_write_array_int_2d, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for write_array_int_2d"},
    {"f90wrap_m_type_test__write_array_real", (PyCFunction)wrap_m_type_test_write_array_real, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for write_array_real"},
    {"f90wrap_m_type_test__write_array_double", (PyCFunction)wrap_m_type_test_write_array_double, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for write_array_double"},
    {"f90wrap_m_type_test__write_array_bool", (PyCFunction)wrap_m_type_test_write_array_bool, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for write_array_bool"},
    {"f90wrap_m_type_test__optional_scalar_real", (PyCFunction)wrap_m_type_test_optional_scalar_real, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for optional_scalar_real"},
    {"f90wrap_m_type_test__optional_scalar_int", (PyCFunction)wrap_m_type_test_optional_scalar_int, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for optional_scalar_int"},
    {"f90wrap_m_type_test__in_scalar_int8", (PyCFunction)wrap_m_type_test_in_scalar_int8, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for in_scalar_int8"},
    {"f90wrap_m_type_test__in_scalar_int16", (PyCFunction)wrap_m_type_test_in_scalar_int16, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for in_scalar_int16"},
    {"f90wrap_m_type_test__in_scalar_int32", (PyCFunction)wrap_m_type_test_in_scalar_int32, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for in_scalar_int32"},
    {"f90wrap_m_type_test__in_scalar_int64", (PyCFunction)wrap_m_type_test_in_scalar_int64, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for in_scalar_int64"},
    {"f90wrap_m_type_test__in_scalar_real32", (PyCFunction)wrap_m_type_test_in_scalar_real32, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for in_scalar_real32"},
    {"f90wrap_m_type_test__in_scalar_real64", (PyCFunction)wrap_m_type_test_in_scalar_real64, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for in_scalar_real64"},
    {"f90wrap_m_type_test__in_array_int64", (PyCFunction)wrap_m_type_test_in_array_int64, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for in_array_int64"},
    {"f90wrap_m_type_test__in_array_real64", (PyCFunction)wrap_m_type_test_in_array_real64, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for in_array_real64"},
    {"f90wrap_m_type_test__t_square_initialise", (PyCFunction)wrap_m_type_test_t_square_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for t_square"},
    {"f90wrap_m_type_test__t_square_finalise", (PyCFunction)wrap_m_type_test_t_square_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for t_square"},
    {"f90wrap_m_type_test__t_circle_initialise", (PyCFunction)wrap_m_type_test_t_circle_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for t_circle"},
    {"f90wrap_m_type_test__t_circle_finalise", (PyCFunction)wrap_m_type_test_t_circle_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for t_circle"},
    {"f90wrap_m_type_test__t_square__get__length", (PyCFunction)wrap_m_type_test__t_square_helper_get_length, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for length"},
    {"f90wrap_m_type_test__t_square__set__length", (PyCFunction)wrap_m_type_test__t_square_helper_set_length, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for length"},
    {"f90wrap_m_type_test__t_circle__get__radius", (PyCFunction)wrap_m_type_test__t_circle_helper_get_radius, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for radius"},
    {"f90wrap_m_type_test__t_circle__set__radius", (PyCFunction)wrap_m_type_test__t_circle_helper_set_radius, METH_VARARGS | \
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
