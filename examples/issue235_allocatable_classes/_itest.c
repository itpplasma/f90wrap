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
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__myclass_t_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass_factory__myclass_create)(float* val, int* ret_myobject);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_create)(float* val, int* ret_self);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_t_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__get__create_count)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__set__create_count)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__get__destroy_count)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__set__destroy_count)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__myclass_t__get__val)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__myclass_t__set__val)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__get__create_count)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__set__create_count)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__get__destroy_count)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__set__destroy_count)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_t__get__val)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_t__set__val)(int* handle, float* value);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__get_val__binding__myclass_t)(int* self, float* val);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__set_val__binding__myclass_t)(int* self, float* val);
extern void F90WRAP_F_SYMBOL(f90wrap_myclass__myclass_destroy__binding__myclass_t)(int* self);
extern void F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_destroy__binding__mytype_t)(int* self);

static PyObject* wrap_myclass_myclass_t_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass__myclass_t_initialise)(this);
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

static PyObject* wrap_myclass_factory_myclass_create(PyObject* self, PyObject* args, PyObject* kwargs)
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
    int ret_myobject[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass_factory__myclass_create)(val, ret_myobject);
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
    PyObject* py_ret_myobject_obj = PyList_New(4);
    if (py_ret_myobject_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_myobject[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_myobject_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_myobject_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_myobject_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_myobject_obj != NULL) return py_ret_myobject_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_myobject_obj != NULL) Py_DECREF(py_ret_myobject_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_myobject_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_myobject_obj);
    }
    return result_tuple;
}

static PyObject* wrap_mytype_mytype_create(PyObject* self, PyObject* args, PyObject* kwargs)
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
    int ret_self[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_create)(val, ret_self);
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
    PyObject* py_ret_self_obj = PyList_New(4);
    if (py_ret_self_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_self[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_self_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_self_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_self_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_self_obj != NULL) return py_ret_self_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_self_obj != NULL) Py_DECREF(py_ret_self_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_self_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_self_obj);
    }
    return result_tuple;
}

static PyObject* wrap_mytype_mytype_t_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_t_initialise)(this);
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

static PyObject* wrap_myclass_helper_get_create_count(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_myclass__get__create_count)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_myclass_helper_set_create_count(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"create_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_myclass__set__create_count)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_myclass_helper_get_destroy_count(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_myclass__get__destroy_count)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_myclass_helper_set_destroy_count(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"destroy_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_myclass__set__destroy_count)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_myclass__myclass_t_helper_get_val(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_myclass__myclass_t__get__val)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_myclass__myclass_t_helper_set_val(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    float value;
    static char *kwlist[] = {"handle", "val", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_myclass__myclass_t__set__val)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_mytype_helper_get_create_count(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_mytype__get__create_count)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_mytype_helper_set_create_count(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"create_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_mytype__set__create_count)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_mytype_helper_get_destroy_count(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_mytype__get__destroy_count)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_mytype_helper_set_destroy_count(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"destroy_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_mytype__set__destroy_count)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_mytype__mytype_t_helper_get_val(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_t__get__val)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_mytype__mytype_t_helper_set_val(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    float value;
    static char *kwlist[] = {"handle", "val", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_t__set__val)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap__myclass__get_val__binding__myclass_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    float val_val = 0;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass__get_val__binding__myclass_t)(self_handle, &val_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_val_obj = Py_BuildValue("d", val_val);
    if (py_val_obj == NULL) {
        return NULL;
    }
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_val_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_val_obj != NULL) return py_val_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_val_obj != NULL) Py_DECREF(py_val_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_val_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_val_obj);
    }
    return result_tuple;
}

static PyObject* wrap__myclass__set_val__binding__myclass_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    PyObject* py_val = NULL;
    float val_val = 0;
    PyArrayObject* val_scalar_arr = NULL;
    int val_scalar_copyback = 0;
    int val_scalar_is_array = 0;
    static char *kwlist[] = {"self", "val", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_self, &py_val)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
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
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass__set_val__binding__myclass_t)(self_handle, val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
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
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap__myclass__myclass_destroy__binding__myclass_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_myclass__myclass_destroy__binding__myclass_t)(self_handle);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap__mytype__mytype_destroy__binding__mytype_t(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    static char *kwlist[] = {"self", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_self)) {
        return NULL;
    }
    
    PyObject* self_handle_obj = NULL;
    PyObject* self_sequence = NULL;
    Py_ssize_t self_handle_len = 0;
    if (PyObject_HasAttrString(py_self, "_handle")) {
        self_handle_obj = PyObject_GetAttrString(py_self, "_handle");
        if (self_handle_obj == NULL) {
            return NULL;
        }
        self_sequence = PySequence_Fast(self_handle_obj, "Failed to access handle sequence");
        if (self_sequence == NULL) {
            Py_DECREF(self_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_self)) {
        self_sequence = PySequence_Fast(py_self, "Argument self must be a handle sequence");
        if (self_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument self must be a Fortran derived-type instance");
        return NULL;
    }
    self_handle_len = PySequence_Fast_GET_SIZE(self_sequence);
    if (self_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument self has an invalid handle length");
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    int* self_handle = (int*)malloc(sizeof(int) * self_handle_len);
    if (self_handle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(self_sequence, i);
        if (item == NULL) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
        self_handle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(self_handle);
            Py_DECREF(self_sequence);
            if (self_handle_obj) Py_DECREF(self_handle_obj);
            return NULL;
        }
    }
    (void)self_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mytype__mytype_destroy__binding__mytype_t)(self_handle);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    Py_RETURN_NONE;
}

/* Method table for _itest module */
static PyMethodDef _itest_methods[] = {
    {"f90wrap_myclass__myclass_t_initialise", (PyCFunction)wrap_myclass_myclass_t_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for myclass_t"},
    {"f90wrap_myclass_factory__myclass_create", (PyCFunction)wrap_myclass_factory_myclass_create, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for myclass_create"},
    {"f90wrap_mytype__mytype_create", (PyCFunction)wrap_mytype_mytype_create, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        mytype_create"},
    {"f90wrap_mytype__mytype_t_initialise", (PyCFunction)wrap_mytype_mytype_t_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for mytype_t"},
    {"f90wrap_myclass__get__create_count", (PyCFunction)wrap_myclass_helper_get_create_count, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for create_count"},
    {"f90wrap_myclass__set__create_count", (PyCFunction)wrap_myclass_helper_set_create_count, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for create_count"},
    {"f90wrap_myclass__get__destroy_count", (PyCFunction)wrap_myclass_helper_get_destroy_count, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for destroy_count"},
    {"f90wrap_myclass__set__destroy_count", (PyCFunction)wrap_myclass_helper_set_destroy_count, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for destroy_count"},
    {"f90wrap_myclass__myclass_t__get__val", (PyCFunction)wrap_myclass__myclass_t_helper_get_val, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for val"},
    {"f90wrap_myclass__myclass_t__set__val", (PyCFunction)wrap_myclass__myclass_t_helper_set_val, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for val"},
    {"f90wrap_mytype__get__create_count", (PyCFunction)wrap_mytype_helper_get_create_count, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for create_count"},
    {"f90wrap_mytype__set__create_count", (PyCFunction)wrap_mytype_helper_set_create_count, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for create_count"},
    {"f90wrap_mytype__get__destroy_count", (PyCFunction)wrap_mytype_helper_get_destroy_count, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for destroy_count"},
    {"f90wrap_mytype__set__destroy_count", (PyCFunction)wrap_mytype_helper_set_destroy_count, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for destroy_count"},
    {"f90wrap_mytype__mytype_t__get__val", (PyCFunction)wrap_mytype__mytype_t_helper_get_val, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for val"},
    {"f90wrap_mytype__mytype_t__set__val", (PyCFunction)wrap_mytype__mytype_t_helper_set_val, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for val"},
    {"f90wrap_myclass__get_val__binding__myclass_t", (PyCFunction)wrap__myclass__get_val__binding__myclass_t, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for get_val"},
    {"f90wrap_myclass__set_val__binding__myclass_t", (PyCFunction)wrap__myclass__set_val__binding__myclass_t, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for set_val"},
    {"f90wrap_myclass__myclass_destroy__binding__myclass_t", \
        (PyCFunction)wrap__myclass__myclass_destroy__binding__myclass_t, METH_VARARGS | METH_KEYWORDS, "Binding alias for \
        myclass_destroy"},
    {"f90wrap_mytype__mytype_destroy__binding__mytype_t", (PyCFunction)wrap__mytype__mytype_destroy__binding__mytype_t, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for mytype_destroy"},
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
