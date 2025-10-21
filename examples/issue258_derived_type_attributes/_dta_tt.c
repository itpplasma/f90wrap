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
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__get_outer_inner)(int* outer, int* ret_inner);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__new_inner)(int* ret_inner, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_inner_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__new_outer)(int* ret_node, int* value, int* inner);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_inner__get__value)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_inner__set__value)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer__get__value)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer__set__value)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer__get__inner)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer__set__inner)(int* handle, int* value);

static PyObject* wrap_dta_tt_get_outer_inner(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_outer = NULL;
    static char *kwlist[] = {"outer", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_outer)) {
        return NULL;
    }
    
    PyObject* outer_handle_obj = NULL;
    PyObject* outer_sequence = NULL;
    Py_ssize_t outer_handle_len = 0;
    if (PyObject_HasAttrString(py_outer, "_handle")) {
        outer_handle_obj = PyObject_GetAttrString(py_outer, "_handle");
        if (outer_handle_obj == NULL) {
            return NULL;
        }
        outer_sequence = PySequence_Fast(outer_handle_obj, "Failed to access handle sequence");
        if (outer_sequence == NULL) {
            Py_DECREF(outer_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_outer)) {
        outer_sequence = PySequence_Fast(py_outer, "Argument outer must be a handle sequence");
        if (outer_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument outer must be a Fortran derived-type instance");
        return NULL;
    }
    outer_handle_len = PySequence_Fast_GET_SIZE(outer_sequence);
    if (outer_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument outer has an invalid handle length");
        Py_DECREF(outer_sequence);
        if (outer_handle_obj) Py_DECREF(outer_handle_obj);
        return NULL;
    }
    int* outer = (int*)malloc(sizeof(int) * outer_handle_len);
    if (outer == NULL) {
        PyErr_NoMemory();
        Py_DECREF(outer_sequence);
        if (outer_handle_obj) Py_DECREF(outer_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < outer_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(outer_sequence, i);
        if (item == NULL) {
            free(outer);
            Py_DECREF(outer_sequence);
            if (outer_handle_obj) Py_DECREF(outer_handle_obj);
            return NULL;
        }
        outer[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(outer);
            Py_DECREF(outer_sequence);
            if (outer_handle_obj) Py_DECREF(outer_handle_obj);
            return NULL;
        }
    }
    (void)outer_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_inner[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__get_outer_inner)(outer, ret_inner);
    if (PyErr_Occurred()) {
        if (outer_sequence) Py_DECREF(outer_sequence);
        if (outer_handle_obj) Py_DECREF(outer_handle_obj);
        free(outer);
        return NULL;
    }
    
    PyObject* py_ret_inner_obj = PyList_New(4);
    if (py_ret_inner_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_inner[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_inner_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_inner_obj, i, item);
    }
    if (outer_sequence) {
        Py_DECREF(outer_sequence);
    }
    if (outer_handle_obj) {
        Py_DECREF(outer_handle_obj);
    }
    free(outer);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_inner_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_inner_obj != NULL) return py_ret_inner_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_inner_obj != NULL) Py_DECREF(py_ret_inner_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_inner_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_inner_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dta_tt_new_inner(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_value = NULL;
    int value_val = 0;
    PyArrayObject* value_scalar_arr = NULL;
    int value_scalar_copyback = 0;
    int value_scalar_is_array = 0;
    static char *kwlist[] = {"value", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_value)) {
        return NULL;
    }
    
    int ret_inner[4] = {0};
    int* value = &value_val;
    if (PyArray_Check(py_value)) {
        value_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_value, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (value_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(value_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument value must have exactly one element");
            Py_DECREF(value_scalar_arr);
            return NULL;
        }
        value_scalar_is_array = 1;
        value = (int*)PyArray_DATA(value_scalar_arr);
        value_val = value[0];
        if (PyArray_DATA(value_scalar_arr) != PyArray_DATA((PyArrayObject*)py_value) || PyArray_TYPE(value_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_value)) {
            value_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_value)) {
        value_val = (int)PyLong_AsLong(py_value);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument value must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__new_inner)(ret_inner, value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (value_scalar_is_array) {
        if (value_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_value, value_scalar_arr) < 0) {
                Py_DECREF(value_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(value_scalar_arr);
    }
    PyObject* py_ret_inner_obj = PyList_New(4);
    if (py_ret_inner_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_inner[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_inner_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_inner_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_inner_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_inner_obj != NULL) return py_ret_inner_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_inner_obj != NULL) Py_DECREF(py_ret_inner_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_inner_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_inner_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dta_tt_t_inner_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_inner_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_dta_tt_new_outer(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_value = NULL;
    int value_val = 0;
    PyArrayObject* value_scalar_arr = NULL;
    int value_scalar_copyback = 0;
    int value_scalar_is_array = 0;
    PyObject* py_inner = NULL;
    static char *kwlist[] = {"value", "inner", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_value, &py_inner)) {
        return NULL;
    }
    
    int ret_node[4] = {0};
    int* value = &value_val;
    if (PyArray_Check(py_value)) {
        value_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_value, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (value_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(value_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument value must have exactly one element");
            Py_DECREF(value_scalar_arr);
            return NULL;
        }
        value_scalar_is_array = 1;
        value = (int*)PyArray_DATA(value_scalar_arr);
        value_val = value[0];
        if (PyArray_DATA(value_scalar_arr) != PyArray_DATA((PyArrayObject*)py_value) || PyArray_TYPE(value_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_value)) {
            value_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_value)) {
        value_val = (int)PyLong_AsLong(py_value);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument value must be a scalar number or NumPy array");
        return NULL;
    }
    PyObject* inner_handle_obj = NULL;
    PyObject* inner_sequence = NULL;
    Py_ssize_t inner_handle_len = 0;
    if (PyObject_HasAttrString(py_inner, "_handle")) {
        inner_handle_obj = PyObject_GetAttrString(py_inner, "_handle");
        if (inner_handle_obj == NULL) {
            return NULL;
        }
        inner_sequence = PySequence_Fast(inner_handle_obj, "Failed to access handle sequence");
        if (inner_sequence == NULL) {
            Py_DECREF(inner_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_inner)) {
        inner_sequence = PySequence_Fast(py_inner, "Argument inner must be a handle sequence");
        if (inner_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument inner must be a Fortran derived-type instance");
        return NULL;
    }
    inner_handle_len = PySequence_Fast_GET_SIZE(inner_sequence);
    if (inner_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument inner has an invalid handle length");
        Py_DECREF(inner_sequence);
        if (inner_handle_obj) Py_DECREF(inner_handle_obj);
        return NULL;
    }
    int* inner = (int*)malloc(sizeof(int) * inner_handle_len);
    if (inner == NULL) {
        PyErr_NoMemory();
        Py_DECREF(inner_sequence);
        if (inner_handle_obj) Py_DECREF(inner_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < inner_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(inner_sequence, i);
        if (item == NULL) {
            free(inner);
            Py_DECREF(inner_sequence);
            if (inner_handle_obj) Py_DECREF(inner_handle_obj);
            return NULL;
        }
        inner[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(inner);
            Py_DECREF(inner_sequence);
            if (inner_handle_obj) Py_DECREF(inner_handle_obj);
            return NULL;
        }
    }
    (void)inner_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__new_outer)(ret_node, value, inner);
    if (PyErr_Occurred()) {
        if (inner_sequence) Py_DECREF(inner_sequence);
        if (inner_handle_obj) Py_DECREF(inner_handle_obj);
        free(inner);
        return NULL;
    }
    
    if (value_scalar_is_array) {
        if (value_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_value, value_scalar_arr) < 0) {
                Py_DECREF(value_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(value_scalar_arr);
    }
    PyObject* py_ret_node_obj = PyList_New(4);
    if (py_ret_node_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_node[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_node_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_node_obj, i, item);
    }
    if (inner_sequence) {
        Py_DECREF(inner_sequence);
    }
    if (inner_handle_obj) {
        Py_DECREF(inner_handle_obj);
    }
    free(inner);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_node_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_node_obj != NULL) return py_ret_node_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_node_obj != NULL) Py_DECREF(py_ret_node_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_node_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_node_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dta_tt_t_outer_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_dta_tt__t_inner_helper_get_value(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_inner__get__value)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_dta_tt__t_inner_helper_set_value(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "value", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_inner__set__value)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_dta_tt__t_outer_helper_get_value(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer__get__value)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_dta_tt__t_outer_helper_set_value(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "value", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer__set__value)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_dta_tt__t_outer_helper_get_derived_inner(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer__get__inner)(handle_handle, value_handle);
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

static PyObject* wrap_dta_tt__t_outer_helper_set_derived_inner(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dta_tt__t_outer__set__inner)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

/* Method table for _dta_tt module */
static PyMethodDef _dta_tt_methods[] = {
    {"f90wrap_dta_tt__get_outer_inner", (PyCFunction)wrap_dta_tt_get_outer_inner, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        get_outer_inner"},
    {"f90wrap_dta_tt__new_inner", (PyCFunction)wrap_dta_tt_new_inner, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        new_inner"},
    {"f90wrap_dta_tt__t_inner_finalise", (PyCFunction)wrap_dta_tt_t_inner_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for t_inner"},
    {"f90wrap_dta_tt__new_outer", (PyCFunction)wrap_dta_tt_new_outer, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        new_outer"},
    {"f90wrap_dta_tt__t_outer_finalise", (PyCFunction)wrap_dta_tt_t_outer_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for t_outer"},
    {"f90wrap_dta_tt__t_inner__get__value", (PyCFunction)wrap_dta_tt__t_inner_helper_get_value, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for value"},
    {"f90wrap_dta_tt__t_inner__set__value", (PyCFunction)wrap_dta_tt__t_inner_helper_set_value, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for value"},
    {"f90wrap_dta_tt__t_outer__get__value", (PyCFunction)wrap_dta_tt__t_outer_helper_get_value, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for value"},
    {"f90wrap_dta_tt__t_outer__set__value", (PyCFunction)wrap_dta_tt__t_outer_helper_set_value, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for value"},
    {"f90wrap_dta_tt__t_outer__get__inner", (PyCFunction)wrap_dta_tt__t_outer_helper_get_derived_inner, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for inner"},
    {"f90wrap_dta_tt__t_outer__set__inner", (PyCFunction)wrap_dta_tt__t_outer_helper_set_derived_inner, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for inner"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _dta_ttmodule = {
    PyModuleDef_HEAD_INIT,
    "dta_tt",
    "Direct-C wrapper for _dta_tt module",
    -1,
    _dta_tt_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__dta_tt(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_dta_ttmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
