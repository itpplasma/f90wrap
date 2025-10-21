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
extern void F90WRAP_F_SYMBOL(f90wrap_test__create)(int* self, int* n);
extern void F90WRAP_F_SYMBOL(f90wrap_test__asum)(int* self, float* ret_asum);
extern void F90WRAP_F_SYMBOL(f90wrap_test__atype_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test__atype_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test__btype_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test__btype_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_test__atype__array__array)(int* dummy_this, int* nd, int* dtype, int* dshape, long \
    long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test__btype__array__array)(int* dummy_this, int* nd, int* dtype, int* dshape, long \
    long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_test__p_create__binding__atype)(int* self, int* n);
extern void F90WRAP_F_SYMBOL(f90wrap_test__p_asum__binding__atype)(int* self, float* ret_asum_class);
extern void F90WRAP_F_SYMBOL(f90wrap_test__p_asum_2__binding__atype)(int* self, float* ret_asum_class);
extern void F90WRAP_F_SYMBOL(f90wrap_test__asum_class__binding__atype)(int* self, float* ret_asum_class);
extern void F90WRAP_F_SYMBOL(f90wrap_test__p_reset__binding__atype)(int* self, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_test__p_asum__binding__btype)(int* self, float* ret_bsum_class);

static PyObject* wrap_test_create(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_self = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"self", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_self, &py_n)) {
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
    
    int* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (int*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (int)PyLong_AsLong(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test__create)(self_handle, n);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    if (n_scalar_is_array) {
        if (n_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n, n_scalar_arr) < 0) {
                Py_DECREF(n_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_scalar_arr);
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

static PyObject* wrap_test_asum(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_self = NULL;
    float ret_asum_val = 0;
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
    F90WRAP_F_SYMBOL(f90wrap_test__asum)(self_handle, &ret_asum_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_ret_asum_obj = Py_BuildValue("d", ret_asum_val);
    if (py_ret_asum_obj == NULL) {
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
    if (py_ret_asum_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_asum_obj != NULL) return py_ret_asum_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_asum_obj != NULL) Py_DECREF(py_ret_asum_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_asum_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_asum_obj);
    }
    return result_tuple;
}

static PyObject* wrap_test_atype_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test__atype_initialise)(this);
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

static PyObject* wrap_test_atype_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_test__atype_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_test_btype_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test__btype_initialise)(this);
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

static PyObject* wrap_test_btype_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_test__btype_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_test__atype_helper_array_array(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_test__atype__array__array)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_test__btype_helper_array_array(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_test__btype__array__array)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap__test__p_create__binding__atype(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"self", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_self, &py_n)) {
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
    
    int* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (int*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (int)PyLong_AsLong(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test__p_create__binding__atype)(self_handle, n);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    if (n_scalar_is_array) {
        if (n_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n, n_scalar_arr) < 0) {
                Py_DECREF(n_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_scalar_arr);
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

static PyObject* wrap__test__p_asum__binding__atype(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    float ret_asum_class_val = 0;
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
    F90WRAP_F_SYMBOL(f90wrap_test__p_asum__binding__atype)(self_handle, &ret_asum_class_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_ret_asum_class_obj = Py_BuildValue("d", ret_asum_class_val);
    if (py_ret_asum_class_obj == NULL) {
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
    if (py_ret_asum_class_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_asum_class_obj != NULL) return py_ret_asum_class_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_asum_class_obj != NULL) Py_DECREF(py_ret_asum_class_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_asum_class_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_asum_class_obj);
    }
    return result_tuple;
}

static PyObject* wrap__test__p_asum_2__binding__atype(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    float ret_asum_class_val = 0;
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
    F90WRAP_F_SYMBOL(f90wrap_test__p_asum_2__binding__atype)(self_handle, &ret_asum_class_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_ret_asum_class_obj = Py_BuildValue("d", ret_asum_class_val);
    if (py_ret_asum_class_obj == NULL) {
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
    if (py_ret_asum_class_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_asum_class_obj != NULL) return py_ret_asum_class_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_asum_class_obj != NULL) Py_DECREF(py_ret_asum_class_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_asum_class_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_asum_class_obj);
    }
    return result_tuple;
}

static PyObject* wrap__test__asum_class__binding__atype(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    float ret_asum_class_val = 0;
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
    F90WRAP_F_SYMBOL(f90wrap_test__asum_class__binding__atype)(self_handle, &ret_asum_class_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_ret_asum_class_obj = Py_BuildValue("d", ret_asum_class_val);
    if (py_ret_asum_class_obj == NULL) {
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
    if (py_ret_asum_class_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_asum_class_obj != NULL) return py_ret_asum_class_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_asum_class_obj != NULL) Py_DECREF(py_ret_asum_class_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_asum_class_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_asum_class_obj);
    }
    return result_tuple;
}

static PyObject* wrap__test__p_reset__binding__atype(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    PyObject* py_value = NULL;
    int value_val = 0;
    PyArrayObject* value_scalar_arr = NULL;
    int value_scalar_copyback = 0;
    int value_scalar_is_array = 0;
    static char *kwlist[] = {"self", "value", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_self, &py_value)) {
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_test__p_reset__binding__atype)(self_handle, value);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
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
    if (self_sequence) {
        Py_DECREF(self_sequence);
    }
    if (self_handle_obj) {
        Py_DECREF(self_handle_obj);
    }
    free(self_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap__test__p_asum__binding__btype(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_self = NULL;
    float ret_bsum_class_val = 0;
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
    F90WRAP_F_SYMBOL(f90wrap_test__p_asum__binding__btype)(self_handle, &ret_bsum_class_val);
    if (PyErr_Occurred()) {
        if (self_sequence) Py_DECREF(self_sequence);
        if (self_handle_obj) Py_DECREF(self_handle_obj);
        free(self_handle);
        return NULL;
    }
    
    PyObject* py_ret_bsum_class_obj = Py_BuildValue("d", ret_bsum_class_val);
    if (py_ret_bsum_class_obj == NULL) {
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
    if (py_ret_bsum_class_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_bsum_class_obj != NULL) return py_ret_bsum_class_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_bsum_class_obj != NULL) Py_DECREF(py_ret_bsum_class_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_bsum_class_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_bsum_class_obj);
    }
    return result_tuple;
}

/* Method table for _library module */
static PyMethodDef _library_methods[] = {
    {"f90wrap_test__create", (PyCFunction)wrap_test_create, METH_VARARGS | METH_KEYWORDS, "Wrapper for create"},
    {"f90wrap_test__asum", (PyCFunction)wrap_test_asum, METH_VARARGS | METH_KEYWORDS, "Wrapper for asum"},
    {"f90wrap_test__atype_initialise", (PyCFunction)wrap_test_atype_initialise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated constructor for atype"},
    {"f90wrap_test__atype_finalise", (PyCFunction)wrap_test_atype_finalise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated destructor for atype"},
    {"f90wrap_test__btype_initialise", (PyCFunction)wrap_test_btype_initialise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated constructor for btype"},
    {"f90wrap_test__btype_finalise", (PyCFunction)wrap_test_btype_finalise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated destructor for btype"},
    {"f90wrap_test__atype__array__array", (PyCFunction)wrap_test__atype_helper_array_array, METH_VARARGS | METH_KEYWORDS, \
        "Array helper for array"},
    {"f90wrap_test__btype__array__array", (PyCFunction)wrap_test__btype_helper_array_array, METH_VARARGS | METH_KEYWORDS, \
        "Array helper for array"},
    {"f90wrap_test__p_create__binding__atype", (PyCFunction)wrap__test__p_create__binding__atype, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for p_create"},
    {"f90wrap_test__p_asum__binding__atype", (PyCFunction)wrap__test__p_asum__binding__atype, METH_VARARGS | METH_KEYWORDS, \
        "Binding alias for p_asum"},
    {"f90wrap_test__p_asum_2__binding__atype", (PyCFunction)wrap__test__p_asum_2__binding__atype, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for p_asum_2"},
    {"f90wrap_test__asum_class__binding__atype", (PyCFunction)wrap__test__asum_class__binding__atype, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for asum_class"},
    {"f90wrap_test__p_reset__binding__atype", (PyCFunction)wrap__test__p_reset__binding__atype, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for p_reset"},
    {"f90wrap_test__p_asum__binding__btype", (PyCFunction)wrap__test__p_asum__binding__btype, METH_VARARGS | METH_KEYWORDS, \
        "Binding alias for p_asum"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _librarymodule = {
    PyModuleDef_HEAD_INIT,
    "library",
    "Direct-C wrapper for _library module",
    -1,
    _library_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__library(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_librarymodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
