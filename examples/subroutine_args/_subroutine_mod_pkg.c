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
extern void F90WRAP_F_SYMBOL(f90wrap_subroutine_mod__routine_with_simple_args)(int* a, int* b, int* c, int* d);
extern void F90WRAP_F_SYMBOL(f90wrap_subroutine_mod__routine_with_multiline_args)(int* a, int* b, int* c, int* d);
extern void F90WRAP_F_SYMBOL(f90wrap_subroutine_mod__routine_with_commented_args)(int* a, int* b, int* c, int* d);
extern void F90WRAP_F_SYMBOL(f90wrap_subroutine_mod__routine_with_more_commented_args)(int* a, int* b, int* c, int* d);

static PyObject* wrap_subroutine_mod_routine_with_simple_args(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_a = NULL;
    int a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    PyObject* py_b = NULL;
    int b_val = 0;
    PyArrayObject* b_scalar_arr = NULL;
    int b_scalar_copyback = 0;
    int b_scalar_is_array = 0;
    int c_val = 0;
    int d_val = 0;
    static char *kwlist[] = {"a", "b", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_a, &py_b)) {
        return NULL;
    }
    
    int* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (int*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (int)PyLong_AsLong(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    int* b = &b_val;
    if (PyArray_Check(py_b)) {
        b_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_b, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (b_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(b_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument b must have exactly one element");
            Py_DECREF(b_scalar_arr);
            return NULL;
        }
        b_scalar_is_array = 1;
        b = (int*)PyArray_DATA(b_scalar_arr);
        b_val = b[0];
        if (PyArray_DATA(b_scalar_arr) != PyArray_DATA((PyArrayObject*)py_b) || PyArray_TYPE(b_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_b)) {
            b_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_b)) {
        b_val = (int)PyLong_AsLong(py_b);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument b must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_subroutine_mod__routine_with_simple_args)(a, b, &c_val, &d_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (a_scalar_is_array) {
        if (a_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_a, a_scalar_arr) < 0) {
                Py_DECREF(a_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(a_scalar_arr);
    }
    if (b_scalar_is_array) {
        if (b_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_b, b_scalar_arr) < 0) {
                Py_DECREF(b_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(b_scalar_arr);
    }
    PyObject* py_c_obj = Py_BuildValue("i", c_val);
    if (py_c_obj == NULL) {
        return NULL;
    }
    PyObject* py_d_obj = Py_BuildValue("i", d_val);
    if (py_d_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_c_obj != NULL) result_count++;
    if (py_d_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_c_obj != NULL) return py_c_obj;
        if (py_d_obj != NULL) return py_d_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_c_obj != NULL) Py_DECREF(py_c_obj);
        if (py_d_obj != NULL) Py_DECREF(py_d_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_c_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_c_obj);
    }
    if (py_d_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_d_obj);
    }
    return result_tuple;
}

static PyObject* wrap_subroutine_mod_routine_with_multiline_args(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_a = NULL;
    int a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    PyObject* py_b = NULL;
    int b_val = 0;
    PyArrayObject* b_scalar_arr = NULL;
    int b_scalar_copyback = 0;
    int b_scalar_is_array = 0;
    int c_val = 0;
    int d_val = 0;
    static char *kwlist[] = {"a", "b", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_a, &py_b)) {
        return NULL;
    }
    
    int* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (int*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (int)PyLong_AsLong(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    int* b = &b_val;
    if (PyArray_Check(py_b)) {
        b_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_b, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (b_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(b_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument b must have exactly one element");
            Py_DECREF(b_scalar_arr);
            return NULL;
        }
        b_scalar_is_array = 1;
        b = (int*)PyArray_DATA(b_scalar_arr);
        b_val = b[0];
        if (PyArray_DATA(b_scalar_arr) != PyArray_DATA((PyArrayObject*)py_b) || PyArray_TYPE(b_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_b)) {
            b_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_b)) {
        b_val = (int)PyLong_AsLong(py_b);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument b must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_subroutine_mod__routine_with_multiline_args)(a, b, &c_val, &d_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (a_scalar_is_array) {
        if (a_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_a, a_scalar_arr) < 0) {
                Py_DECREF(a_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(a_scalar_arr);
    }
    if (b_scalar_is_array) {
        if (b_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_b, b_scalar_arr) < 0) {
                Py_DECREF(b_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(b_scalar_arr);
    }
    PyObject* py_c_obj = Py_BuildValue("i", c_val);
    if (py_c_obj == NULL) {
        return NULL;
    }
    PyObject* py_d_obj = Py_BuildValue("i", d_val);
    if (py_d_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_c_obj != NULL) result_count++;
    if (py_d_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_c_obj != NULL) return py_c_obj;
        if (py_d_obj != NULL) return py_d_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_c_obj != NULL) Py_DECREF(py_c_obj);
        if (py_d_obj != NULL) Py_DECREF(py_d_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_c_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_c_obj);
    }
    if (py_d_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_d_obj);
    }
    return result_tuple;
}

static PyObject* wrap_subroutine_mod_routine_with_commented_args(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_a = NULL;
    int a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    PyObject* py_b = NULL;
    int b_val = 0;
    PyArrayObject* b_scalar_arr = NULL;
    int b_scalar_copyback = 0;
    int b_scalar_is_array = 0;
    int c_val = 0;
    int d_val = 0;
    static char *kwlist[] = {"a", "b", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_a, &py_b)) {
        return NULL;
    }
    
    int* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (int*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (int)PyLong_AsLong(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    int* b = &b_val;
    if (PyArray_Check(py_b)) {
        b_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_b, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (b_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(b_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument b must have exactly one element");
            Py_DECREF(b_scalar_arr);
            return NULL;
        }
        b_scalar_is_array = 1;
        b = (int*)PyArray_DATA(b_scalar_arr);
        b_val = b[0];
        if (PyArray_DATA(b_scalar_arr) != PyArray_DATA((PyArrayObject*)py_b) || PyArray_TYPE(b_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_b)) {
            b_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_b)) {
        b_val = (int)PyLong_AsLong(py_b);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument b must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_subroutine_mod__routine_with_commented_args)(a, b, &c_val, &d_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (a_scalar_is_array) {
        if (a_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_a, a_scalar_arr) < 0) {
                Py_DECREF(a_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(a_scalar_arr);
    }
    if (b_scalar_is_array) {
        if (b_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_b, b_scalar_arr) < 0) {
                Py_DECREF(b_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(b_scalar_arr);
    }
    PyObject* py_c_obj = Py_BuildValue("i", c_val);
    if (py_c_obj == NULL) {
        return NULL;
    }
    PyObject* py_d_obj = Py_BuildValue("i", d_val);
    if (py_d_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_c_obj != NULL) result_count++;
    if (py_d_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_c_obj != NULL) return py_c_obj;
        if (py_d_obj != NULL) return py_d_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_c_obj != NULL) Py_DECREF(py_c_obj);
        if (py_d_obj != NULL) Py_DECREF(py_d_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_c_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_c_obj);
    }
    if (py_d_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_d_obj);
    }
    return result_tuple;
}

static PyObject* wrap_subroutine_mod_routine_with_more_commented_args(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_a = NULL;
    int a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    PyObject* py_b = NULL;
    int b_val = 0;
    PyArrayObject* b_scalar_arr = NULL;
    int b_scalar_copyback = 0;
    int b_scalar_is_array = 0;
    int c_val = 0;
    int d_val = 0;
    static char *kwlist[] = {"a", "b", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_a, &py_b)) {
        return NULL;
    }
    
    int* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (int*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (int)PyLong_AsLong(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    int* b = &b_val;
    if (PyArray_Check(py_b)) {
        b_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_b, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (b_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(b_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument b must have exactly one element");
            Py_DECREF(b_scalar_arr);
            return NULL;
        }
        b_scalar_is_array = 1;
        b = (int*)PyArray_DATA(b_scalar_arr);
        b_val = b[0];
        if (PyArray_DATA(b_scalar_arr) != PyArray_DATA((PyArrayObject*)py_b) || PyArray_TYPE(b_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_b)) {
            b_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_b)) {
        b_val = (int)PyLong_AsLong(py_b);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument b must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_subroutine_mod__routine_with_more_commented_args)(a, b, &c_val, &d_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (a_scalar_is_array) {
        if (a_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_a, a_scalar_arr) < 0) {
                Py_DECREF(a_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(a_scalar_arr);
    }
    if (b_scalar_is_array) {
        if (b_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_b, b_scalar_arr) < 0) {
                Py_DECREF(b_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(b_scalar_arr);
    }
    PyObject* py_c_obj = Py_BuildValue("i", c_val);
    if (py_c_obj == NULL) {
        return NULL;
    }
    PyObject* py_d_obj = Py_BuildValue("i", d_val);
    if (py_d_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_c_obj != NULL) result_count++;
    if (py_d_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_c_obj != NULL) return py_c_obj;
        if (py_d_obj != NULL) return py_d_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_c_obj != NULL) Py_DECREF(py_c_obj);
        if (py_d_obj != NULL) Py_DECREF(py_d_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_c_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_c_obj);
    }
    if (py_d_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_d_obj);
    }
    return result_tuple;
}

/* Method table for _subroutine_mod_pkg module */
static PyMethodDef _subroutine_mod_pkg_methods[] = {
    {"f90wrap_subroutine_mod__routine_with_simple_args", (PyCFunction)wrap_subroutine_mod_routine_with_simple_args, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for routine_with_simple_args"},
    {"f90wrap_subroutine_mod__routine_with_multiline_args", (PyCFunction)wrap_subroutine_mod_routine_with_multiline_args, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for routine_with_multiline_args"},
    {"f90wrap_subroutine_mod__routine_with_commented_args", (PyCFunction)wrap_subroutine_mod_routine_with_commented_args, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for routine_with_commented_args"},
    {"f90wrap_subroutine_mod__routine_with_more_commented_args", \
        (PyCFunction)wrap_subroutine_mod_routine_with_more_commented_args, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        routine_with_more_commented_args"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _subroutine_mod_pkgmodule = {
    PyModuleDef_HEAD_INIT,
    "subroutine_mod_pkg",
    "Direct-C wrapper for _subroutine_mod_pkg module",
    -1,
    _subroutine_mod_pkg_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__subroutine_mod_pkg(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_subroutine_mod_pkgmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
