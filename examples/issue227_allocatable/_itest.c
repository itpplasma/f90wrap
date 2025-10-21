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
extern void F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type_func)(float* val, int* ret_out);
extern void F90WRAP_F_SYMBOL(f90wrap_alloc_output__noalloc_output_subroutine)(float* val, int* out);
extern void F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type__get__a)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type__set__a)(int* handle, float* value);

static PyObject* wrap_alloc_output_alloc_output_type_func(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_val = NULL;
    float val_val = 0;
    PyArrayObject* val_scalar_arr = NULL;
    int val_scalar_copyback = 0;
    int val_scalar_is_array = 0;
    static char *kwlist[] = {"val", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_val)) {
        return NULL;
    }
    
    float* val = &val_val;
    if (PyArray_Check(py_val)) {
        val_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_val, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (val_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(val_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument val must have exactly one element");
            Py_DECREF(val_scalar_arr);
            return NULL;
        }
        val_scalar_is_array = 1;
        val = (float*)PyArray_DATA(val_scalar_arr);
        val_val = val[0];
        if (PyArray_DATA(val_scalar_arr) != PyArray_DATA((PyArrayObject*)py_val) || PyArray_TYPE(val_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_val)) {
            val_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_val)) {
        val_val = (float)PyFloat_AsDouble(py_val);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val must be a scalar number or NumPy array");
        return NULL;
    }
    int ret_out[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type_func)(val, ret_out);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (val_scalar_is_array) {
        if (val_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_val, val_scalar_arr) < 0) {
                Py_DECREF(val_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(val_scalar_arr);
    }
    PyObject* py_ret_out_obj = PyList_New(4);
    if (py_ret_out_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_out[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_out_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_out_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_out_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_out_obj != NULL) return py_ret_out_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_out_obj != NULL) Py_DECREF(py_ret_out_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_out_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_out_obj);
    }
    return result_tuple;
}

static PyObject* wrap_alloc_output_noalloc_output_subroutine(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_val = NULL;
    float val_val = 0;
    PyArrayObject* val_scalar_arr = NULL;
    int val_scalar_copyback = 0;
    int val_scalar_is_array = 0;
    PyObject* py_out = NULL;
    static char *kwlist[] = {"val", "out", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_val, &py_out)) {
        return NULL;
    }
    
    float* val = &val_val;
    if (PyArray_Check(py_val)) {
        val_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_val, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (val_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(val_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument val must have exactly one element");
            Py_DECREF(val_scalar_arr);
            return NULL;
        }
        val_scalar_is_array = 1;
        val = (float*)PyArray_DATA(val_scalar_arr);
        val_val = val[0];
        if (PyArray_DATA(val_scalar_arr) != PyArray_DATA((PyArrayObject*)py_val) || PyArray_TYPE(val_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_val)) {
            val_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_val)) {
        val_val = (float)PyFloat_AsDouble(py_val);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val must be a scalar number or NumPy array");
        return NULL;
    }
    PyObject* out_handle_obj = NULL;
    PyObject* out_sequence = NULL;
    Py_ssize_t out_handle_len = 0;
    if (PyObject_HasAttrString(py_out, "_handle")) {
        out_handle_obj = PyObject_GetAttrString(py_out, "_handle");
        if (out_handle_obj == NULL) {
            return NULL;
        }
        out_sequence = PySequence_Fast(out_handle_obj, "Failed to access handle sequence");
        if (out_sequence == NULL) {
            Py_DECREF(out_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_out)) {
        out_sequence = PySequence_Fast(py_out, "Argument out must be a handle sequence");
        if (out_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument out must be a Fortran derived-type instance");
        return NULL;
    }
    out_handle_len = PySequence_Fast_GET_SIZE(out_sequence);
    if (out_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument out has an invalid handle length");
        Py_DECREF(out_sequence);
        if (out_handle_obj) Py_DECREF(out_handle_obj);
        return NULL;
    }
    int* out = (int*)malloc(sizeof(int) * out_handle_len);
    if (out == NULL) {
        PyErr_NoMemory();
        Py_DECREF(out_sequence);
        if (out_handle_obj) Py_DECREF(out_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < out_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(out_sequence, i);
        if (item == NULL) {
            free(out);
            Py_DECREF(out_sequence);
            if (out_handle_obj) Py_DECREF(out_handle_obj);
            return NULL;
        }
        out[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(out);
            Py_DECREF(out_sequence);
            if (out_handle_obj) Py_DECREF(out_handle_obj);
            return NULL;
        }
    }
    (void)out_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_alloc_output__noalloc_output_subroutine)(val, out);
    if (PyErr_Occurred()) {
        if (out_sequence) Py_DECREF(out_sequence);
        if (out_handle_obj) Py_DECREF(out_handle_obj);
        free(out);
        return NULL;
    }
    
    if (val_scalar_is_array) {
        if (val_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_val, val_scalar_arr) < 0) {
                Py_DECREF(val_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(val_scalar_arr);
    }
    if (out_sequence) {
        Py_DECREF(out_sequence);
    }
    if (out_handle_obj) {
        Py_DECREF(out_handle_obj);
    }
    free(out);
    Py_RETURN_NONE;
}

static PyObject* wrap_alloc_output_alloc_output_type_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type_initialise)(this);
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

static PyObject* wrap_alloc_output_alloc_output_type_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_alloc_output__alloc_output_type_helper_get_a(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type__get__a)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_alloc_output__alloc_output_type_helper_set_a(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "a", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_alloc_output__alloc_output_type__set__a)(this_handle, &fortran_value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _itest module */
static PyMethodDef _itest_methods[] = {
    {"f90wrap_alloc_output__alloc_output_type_func", (PyCFunction)wrap_alloc_output_alloc_output_type_func, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for alloc_output_type_func"},
    {"f90wrap_alloc_output__noalloc_output_subroutine", (PyCFunction)wrap_alloc_output_noalloc_output_subroutine, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for noalloc_output_subroutine"},
    {"f90wrap_alloc_output__alloc_output_type_initialise", (PyCFunction)wrap_alloc_output_alloc_output_type_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for alloc_output_type"},
    {"f90wrap_alloc_output__alloc_output_type_finalise", (PyCFunction)wrap_alloc_output_alloc_output_type_finalise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated destructor for alloc_output_type"},
    {"f90wrap_alloc_output__alloc_output_type__get__a", (PyCFunction)wrap_alloc_output__alloc_output_type_helper_get_a, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a"},
    {"f90wrap_alloc_output__alloc_output_type__set__a", (PyCFunction)wrap_alloc_output__alloc_output_type_helper_set_a, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _itestmodule = {
    PyModuleDef_HEAD_INIT,
    "itest",
    "Direct-C wrapper for _itest module",
    -1,
    _itest_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__itest(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_itestmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
