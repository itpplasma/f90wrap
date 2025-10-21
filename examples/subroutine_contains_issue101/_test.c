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
extern void F90WRAP_F_SYMBOL(f90wrap_routine_member_procedures)(int* in1, int* in2, int* out1, int* out2);
extern void F90WRAP_F_SYMBOL(f90wrap_routine_member_procedures2)(int* in1, int* in2, int* out1, int* out2);
extern void F90WRAP_F_SYMBOL(f90wrap_function_member_procedures)(int* in1, int* in2, int* out1, int* out2, int* \
    ret_out3);

static PyObject* wrap__test_routine_member_procedures(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in1 = NULL;
    int in1_val = 0;
    PyArrayObject* in1_scalar_arr = NULL;
    int in1_scalar_copyback = 0;
    int in1_scalar_is_array = 0;
    PyObject* py_in2 = NULL;
    int in2_val = 0;
    PyArrayObject* in2_scalar_arr = NULL;
    int in2_scalar_copyback = 0;
    int in2_scalar_is_array = 0;
    int out1_val = 0;
    int out2_val = 0;
    static char *kwlist[] = {"in1", "in2", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_in1, &py_in2)) {
        return NULL;
    }
    
    int* in1 = &in1_val;
    if (PyArray_Check(py_in1)) {
        in1_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in1, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in1_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in1_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in1 must have exactly one element");
            Py_DECREF(in1_scalar_arr);
            return NULL;
        }
        in1_scalar_is_array = 1;
        in1 = (int*)PyArray_DATA(in1_scalar_arr);
        in1_val = in1[0];
        if (PyArray_DATA(in1_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in1) || PyArray_TYPE(in1_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in1)) {
            in1_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in1)) {
        in1_val = (int)PyLong_AsLong(py_in1);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in1 must be a scalar number or NumPy array");
        return NULL;
    }
    int* in2 = &in2_val;
    if (PyArray_Check(py_in2)) {
        in2_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in2, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in2_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in2_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in2 must have exactly one element");
            Py_DECREF(in2_scalar_arr);
            return NULL;
        }
        in2_scalar_is_array = 1;
        in2 = (int*)PyArray_DATA(in2_scalar_arr);
        in2_val = in2[0];
        if (PyArray_DATA(in2_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in2) || PyArray_TYPE(in2_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in2)) {
            in2_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in2)) {
        in2_val = (int)PyLong_AsLong(py_in2);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in2 must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_routine_member_procedures)(in1, in2, &out1_val, &out2_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (in1_scalar_is_array) {
        if (in1_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in1, in1_scalar_arr) < 0) {
                Py_DECREF(in1_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in1_scalar_arr);
    }
    if (in2_scalar_is_array) {
        if (in2_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in2, in2_scalar_arr) < 0) {
                Py_DECREF(in2_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in2_scalar_arr);
    }
    PyObject* py_out1_obj = Py_BuildValue("i", out1_val);
    if (py_out1_obj == NULL) {
        return NULL;
    }
    PyObject* py_out2_obj = Py_BuildValue("i", out2_val);
    if (py_out2_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_out1_obj != NULL) result_count++;
    if (py_out2_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_out1_obj != NULL) return py_out1_obj;
        if (py_out2_obj != NULL) return py_out2_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_out1_obj != NULL) Py_DECREF(py_out1_obj);
        if (py_out2_obj != NULL) Py_DECREF(py_out2_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_out1_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_out1_obj);
    }
    if (py_out2_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_out2_obj);
    }
    return result_tuple;
}

static PyObject* wrap__test_routine_member_procedures2(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in1 = NULL;
    int in1_val = 0;
    PyArrayObject* in1_scalar_arr = NULL;
    int in1_scalar_copyback = 0;
    int in1_scalar_is_array = 0;
    PyObject* py_in2 = NULL;
    int in2_val = 0;
    PyArrayObject* in2_scalar_arr = NULL;
    int in2_scalar_copyback = 0;
    int in2_scalar_is_array = 0;
    int out1_val = 0;
    int out2_val = 0;
    static char *kwlist[] = {"in1", "in2", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_in1, &py_in2)) {
        return NULL;
    }
    
    int* in1 = &in1_val;
    if (PyArray_Check(py_in1)) {
        in1_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in1, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in1_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in1_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in1 must have exactly one element");
            Py_DECREF(in1_scalar_arr);
            return NULL;
        }
        in1_scalar_is_array = 1;
        in1 = (int*)PyArray_DATA(in1_scalar_arr);
        in1_val = in1[0];
        if (PyArray_DATA(in1_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in1) || PyArray_TYPE(in1_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in1)) {
            in1_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in1)) {
        in1_val = (int)PyLong_AsLong(py_in1);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in1 must be a scalar number or NumPy array");
        return NULL;
    }
    int* in2 = &in2_val;
    if (PyArray_Check(py_in2)) {
        in2_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in2, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in2_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in2_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in2 must have exactly one element");
            Py_DECREF(in2_scalar_arr);
            return NULL;
        }
        in2_scalar_is_array = 1;
        in2 = (int*)PyArray_DATA(in2_scalar_arr);
        in2_val = in2[0];
        if (PyArray_DATA(in2_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in2) || PyArray_TYPE(in2_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in2)) {
            in2_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in2)) {
        in2_val = (int)PyLong_AsLong(py_in2);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in2 must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_routine_member_procedures2)(in1, in2, &out1_val, &out2_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (in1_scalar_is_array) {
        if (in1_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in1, in1_scalar_arr) < 0) {
                Py_DECREF(in1_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in1_scalar_arr);
    }
    if (in2_scalar_is_array) {
        if (in2_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in2, in2_scalar_arr) < 0) {
                Py_DECREF(in2_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in2_scalar_arr);
    }
    PyObject* py_out1_obj = Py_BuildValue("i", out1_val);
    if (py_out1_obj == NULL) {
        return NULL;
    }
    PyObject* py_out2_obj = Py_BuildValue("i", out2_val);
    if (py_out2_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_out1_obj != NULL) result_count++;
    if (py_out2_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_out1_obj != NULL) return py_out1_obj;
        if (py_out2_obj != NULL) return py_out2_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_out1_obj != NULL) Py_DECREF(py_out1_obj);
        if (py_out2_obj != NULL) Py_DECREF(py_out2_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_out1_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_out1_obj);
    }
    if (py_out2_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_out2_obj);
    }
    return result_tuple;
}

static PyObject* wrap__test_function_member_procedures(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in1 = NULL;
    int in1_val = 0;
    PyArrayObject* in1_scalar_arr = NULL;
    int in1_scalar_copyback = 0;
    int in1_scalar_is_array = 0;
    PyObject* py_in2 = NULL;
    int in2_val = 0;
    PyArrayObject* in2_scalar_arr = NULL;
    int in2_scalar_copyback = 0;
    int in2_scalar_is_array = 0;
    int out1_val = 0;
    int out2_val = 0;
    int ret_out3_val = 0;
    static char *kwlist[] = {"in1", "in2", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_in1, &py_in2)) {
        return NULL;
    }
    
    int* in1 = &in1_val;
    if (PyArray_Check(py_in1)) {
        in1_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in1, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in1_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in1_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in1 must have exactly one element");
            Py_DECREF(in1_scalar_arr);
            return NULL;
        }
        in1_scalar_is_array = 1;
        in1 = (int*)PyArray_DATA(in1_scalar_arr);
        in1_val = in1[0];
        if (PyArray_DATA(in1_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in1) || PyArray_TYPE(in1_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in1)) {
            in1_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in1)) {
        in1_val = (int)PyLong_AsLong(py_in1);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in1 must be a scalar number or NumPy array");
        return NULL;
    }
    int* in2 = &in2_val;
    if (PyArray_Check(py_in2)) {
        in2_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in2, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in2_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in2_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in2 must have exactly one element");
            Py_DECREF(in2_scalar_arr);
            return NULL;
        }
        in2_scalar_is_array = 1;
        in2 = (int*)PyArray_DATA(in2_scalar_arr);
        in2_val = in2[0];
        if (PyArray_DATA(in2_scalar_arr) != PyArray_DATA((PyArrayObject*)py_in2) || PyArray_TYPE(in2_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in2)) {
            in2_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in2)) {
        in2_val = (int)PyLong_AsLong(py_in2);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in2 must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_function_member_procedures)(in1, in2, &out1_val, &out2_val, &ret_out3_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (in1_scalar_is_array) {
        if (in1_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in1, in1_scalar_arr) < 0) {
                Py_DECREF(in1_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in1_scalar_arr);
    }
    if (in2_scalar_is_array) {
        if (in2_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in2, in2_scalar_arr) < 0) {
                Py_DECREF(in2_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in2_scalar_arr);
    }
    PyObject* py_out1_obj = Py_BuildValue("i", out1_val);
    if (py_out1_obj == NULL) {
        return NULL;
    }
    PyObject* py_out2_obj = Py_BuildValue("i", out2_val);
    if (py_out2_obj == NULL) {
        return NULL;
    }
    PyObject* py_ret_out3_obj = Py_BuildValue("i", ret_out3_val);
    if (py_ret_out3_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_out1_obj != NULL) result_count++;
    if (py_out2_obj != NULL) result_count++;
    if (py_ret_out3_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_out1_obj != NULL) return py_out1_obj;
        if (py_out2_obj != NULL) return py_out2_obj;
        if (py_ret_out3_obj != NULL) return py_ret_out3_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_out1_obj != NULL) Py_DECREF(py_out1_obj);
        if (py_out2_obj != NULL) Py_DECREF(py_out2_obj);
        if (py_ret_out3_obj != NULL) Py_DECREF(py_ret_out3_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_out1_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_out1_obj);
    }
    if (py_out2_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_out2_obj);
    }
    if (py_ret_out3_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_out3_obj);
    }
    return result_tuple;
}

/* Method table for _test module */
static PyMethodDef _test_methods[] = {
    {"f90wrap_routine_member_procedures", (PyCFunction)wrap__test_routine_member_procedures, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for routine_member_procedures"},
    {"f90wrap_routine_member_procedures2", (PyCFunction)wrap__test_routine_member_procedures2, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for routine_member_procedures2"},
    {"f90wrap_function_member_procedures", (PyCFunction)wrap__test_function_member_procedures, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for function_member_procedures"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _testmodule = {
    PyModuleDef_HEAD_INIT,
    "test",
    "Direct-C wrapper for _test module",
    -1,
    _test_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__test(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_testmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
