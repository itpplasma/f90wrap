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
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__test_real)(float* in_real, int* ret_out_int);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__test_real4)(float* in_real, int* ret_out_int);
extern void F90WRAP_F_SYMBOL(f90wrap_m_test__test_real8)(double* in_real, int* ret_out_int);

static PyObject* wrap_m_test_test_real(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_real = NULL;
    float in_real_val = 0;
    PyArrayObject* in_real_scalar_arr = NULL;
    int in_real_scalar_copyback = 0;
    int in_real_scalar_is_array = 0;
    int ret_out_int_val = 0;
    static char *kwlist[] = {"in_real", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_in_real)) {
        return NULL;
    }
    
    float* in_real = &in_real_val;
    if (PyArray_Check(py_in_real)) {
        in_real_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_real, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_real_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_real_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_real must have exactly one element");
            Py_DECREF(in_real_scalar_arr);
            return NULL;
        }
        in_real_scalar_is_array = 1;
        in_real = (float*)PyArray_DATA(in_real_scalar_arr);
        in_real_val = in_real[0];
        if (PyArray_DATA(in_real_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_real) || PyArray_TYPE(in_real_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in_real)) {
            in_real_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_real)) {
        in_real_val = (float)PyFloat_AsDouble(py_in_real);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_real must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__test_real)(in_real, &ret_out_int_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (in_real_scalar_is_array) {
        if (in_real_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_real, in_real_scalar_arr) < 0) {
                Py_DECREF(in_real_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_real_scalar_arr);
    }
    PyObject* py_ret_out_int_obj = Py_BuildValue("i", ret_out_int_val);
    if (py_ret_out_int_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_out_int_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_out_int_obj != NULL) return py_ret_out_int_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_out_int_obj != NULL) Py_DECREF(py_ret_out_int_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_out_int_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_out_int_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_test_real4(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_real = NULL;
    float in_real_val = 0;
    PyArrayObject* in_real_scalar_arr = NULL;
    int in_real_scalar_copyback = 0;
    int in_real_scalar_is_array = 0;
    int ret_out_int_val = 0;
    static char *kwlist[] = {"in_real", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_in_real)) {
        return NULL;
    }
    
    float* in_real = &in_real_val;
    if (PyArray_Check(py_in_real)) {
        in_real_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_real, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_real_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_real_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_real must have exactly one element");
            Py_DECREF(in_real_scalar_arr);
            return NULL;
        }
        in_real_scalar_is_array = 1;
        in_real = (float*)PyArray_DATA(in_real_scalar_arr);
        in_real_val = in_real[0];
        if (PyArray_DATA(in_real_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_real) || PyArray_TYPE(in_real_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in_real)) {
            in_real_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_real)) {
        in_real_val = (float)PyFloat_AsDouble(py_in_real);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_real must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__test_real4)(in_real, &ret_out_int_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (in_real_scalar_is_array) {
        if (in_real_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_real, in_real_scalar_arr) < 0) {
                Py_DECREF(in_real_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_real_scalar_arr);
    }
    PyObject* py_ret_out_int_obj = Py_BuildValue("i", ret_out_int_val);
    if (py_ret_out_int_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_out_int_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_out_int_obj != NULL) return py_ret_out_int_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_out_int_obj != NULL) Py_DECREF(py_ret_out_int_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_out_int_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_out_int_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_test_test_real8(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_real = NULL;
    double in_real_val = 0;
    PyArrayObject* in_real_scalar_arr = NULL;
    int in_real_scalar_copyback = 0;
    int in_real_scalar_is_array = 0;
    int ret_out_int_val = 0;
    static char *kwlist[] = {"in_real", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_in_real)) {
        return NULL;
    }
    
    double* in_real = &in_real_val;
    if (PyArray_Check(py_in_real)) {
        in_real_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_real, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in_real_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in_real_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_real must have exactly one element");
            Py_DECREF(in_real_scalar_arr);
            return NULL;
        }
        in_real_scalar_is_array = 1;
        in_real = (double*)PyArray_DATA(in_real_scalar_arr);
        in_real_val = in_real[0];
        if (PyArray_DATA(in_real_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_real) || PyArray_TYPE(in_real_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in_real)) {
            in_real_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_real)) {
        in_real_val = (double)PyFloat_AsDouble(py_in_real);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_real must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_test__test_real8)(in_real, &ret_out_int_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (in_real_scalar_is_array) {
        if (in_real_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_real, in_real_scalar_arr) < 0) {
                Py_DECREF(in_real_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in_real_scalar_arr);
    }
    PyObject* py_ret_out_int_obj = Py_BuildValue("i", ret_out_int_val);
    if (py_ret_out_int_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_out_int_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_out_int_obj != NULL) return py_ret_out_int_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_out_int_obj != NULL) Py_DECREF(py_ret_out_int_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_out_int_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_out_int_obj);
    }
    return result_tuple;
}

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_test__test_real", (PyCFunction)wrap_m_test_test_real, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        test_real"},
    {"f90wrap_m_test__test_real4", (PyCFunction)wrap_m_test_test_real4, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        test_real4"},
    {"f90wrap_m_test__test_real8", (PyCFunction)wrap_m_test_test_real8, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        test_real8"},
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
