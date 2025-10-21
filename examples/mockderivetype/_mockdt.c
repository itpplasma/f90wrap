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
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__use_set_vars)(void);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__return_a_type_func)(int* ret_a);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__return_a_type_sub)(int* a);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__unused_type_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__unused_type_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_horrible__horrible_type_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_horrible__horrible_type_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_use_a_type__do_stuff)(double* factor, double* out);
extern void F90WRAP_F_SYMBOL(f90wrap_use_a_type__not_used)(double* x, double* y);
extern void F90WRAP_F_SYMBOL(f90wrap_leveltwomod__leveltwo_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_leveltwomod__leveltwo_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_top_level)(double* in_, double* out);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__get__a_set_real)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__set__a_set_real)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__get__a_set_bool)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__set__a_set_bool)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__get__bool)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__set__bool)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__get__integ)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__set__integ)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__get__rl)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__set__rl)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__array__vec)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__get__dtype)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__set__dtype)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__unused_type__get__rl)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_define_a_type__unused_type__set__rl)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_horrible__get__a_real)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_horrible__set__a_real)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_horrible__horrible_type__array__x)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_use_a_type__get__p)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_use_a_type__set__p)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_use_a_type__array_getitem__p_array)(int* dummy_this, int* index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_use_a_type__array_setitem__p_array)(int* dummy_this, int* index, int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_use_a_type__array_len__p_array)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_use_a_type__array__vector)(int* dummy_this, int* nd, int* dtype, int* dshape, long \
    long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_leveltwomod__leveltwo__get__rl)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_leveltwomod__leveltwo__set__rl)(int* handle, double* value);

static PyObject* wrap_define_a_type_use_set_vars(PyObject* self, PyObject* args, PyObject* kwargs)
{
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__use_set_vars)();
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* wrap_define_a_type_return_a_type_func(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int ret_a[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__return_a_type_func)(ret_a);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_a_obj = PyList_New(4);
    if (py_ret_a_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_a[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_a_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_a_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_a_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_a_obj != NULL) return py_ret_a_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_a_obj != NULL) Py_DECREF(py_ret_a_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_a_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_a_obj);
    }
    return result_tuple;
}

static PyObject* wrap_define_a_type_return_a_type_sub(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int a[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__return_a_type_sub)(a);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_a_obj = PyList_New(4);
    if (py_a_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)a[i]);
        if (item == NULL) {
            Py_DECREF(py_a_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_a_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_a_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_a_obj != NULL) return py_a_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_a_obj != NULL) Py_DECREF(py_a_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_a_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_a_obj);
    }
    return result_tuple;
}

static PyObject* wrap_define_a_type_atype_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype_initialise)(this);
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

static PyObject* wrap_define_a_type_atype_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_define_a_type_unused_type_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__unused_type_initialise)(this);
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

static PyObject* wrap_define_a_type_unused_type_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__unused_type_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_horrible_horrible_type_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_horrible__horrible_type_initialise)(this);
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

static PyObject* wrap_horrible_horrible_type_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_horrible__horrible_type_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_use_a_type_do_stuff(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_factor = NULL;
    double factor_val = 0;
    PyArrayObject* factor_scalar_arr = NULL;
    int factor_scalar_copyback = 0;
    int factor_scalar_is_array = 0;
    double out_val = 0;
    static char *kwlist[] = {"factor", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_factor)) {
        return NULL;
    }
    
    double* factor = &factor_val;
    if (PyArray_Check(py_factor)) {
        factor_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_factor, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (factor_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(factor_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument factor must have exactly one element");
            Py_DECREF(factor_scalar_arr);
            return NULL;
        }
        factor_scalar_is_array = 1;
        factor = (double*)PyArray_DATA(factor_scalar_arr);
        factor_val = factor[0];
        if (PyArray_DATA(factor_scalar_arr) != PyArray_DATA((PyArrayObject*)py_factor) || PyArray_TYPE(factor_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_factor)) {
            factor_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_factor)) {
        factor_val = (double)PyFloat_AsDouble(py_factor);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument factor must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_use_a_type__do_stuff)(factor, &out_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (factor_scalar_is_array) {
        if (factor_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_factor, factor_scalar_arr) < 0) {
                Py_DECREF(factor_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(factor_scalar_arr);
    }
    PyObject* py_out_obj = Py_BuildValue("d", out_val);
    if (py_out_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_out_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_out_obj != NULL) return py_out_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_out_obj != NULL) Py_DECREF(py_out_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_out_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_out_obj);
    }
    return result_tuple;
}

static PyObject* wrap_use_a_type_not_used(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_x = NULL;
    double x_val = 0;
    PyArrayObject* x_scalar_arr = NULL;
    int x_scalar_copyback = 0;
    int x_scalar_is_array = 0;
    double y_val = 0;
    static char *kwlist[] = {"x", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_x)) {
        return NULL;
    }
    
    double* x = &x_val;
    if (PyArray_Check(py_x)) {
        x_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_x, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (x_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(x_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument x must have exactly one element");
            Py_DECREF(x_scalar_arr);
            return NULL;
        }
        x_scalar_is_array = 1;
        x = (double*)PyArray_DATA(x_scalar_arr);
        x_val = x[0];
        if (PyArray_DATA(x_scalar_arr) != PyArray_DATA((PyArrayObject*)py_x) || PyArray_TYPE(x_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_x)) {
            x_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_x)) {
        x_val = (double)PyFloat_AsDouble(py_x);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument x must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_use_a_type__not_used)(x, &y_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (x_scalar_is_array) {
        if (x_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_x, x_scalar_arr) < 0) {
                Py_DECREF(x_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(x_scalar_arr);
    }
    PyObject* py_y_obj = Py_BuildValue("d", y_val);
    if (py_y_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_y_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_y_obj != NULL) return py_y_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_y_obj != NULL) Py_DECREF(py_y_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_y_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_y_obj);
    }
    return result_tuple;
}

static PyObject* wrap_leveltwomod_leveltwo_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_leveltwomod__leveltwo_initialise)(this);
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

static PyObject* wrap_leveltwomod_leveltwo_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_leveltwomod__leveltwo_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap__mockdt_top_level(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_in_ = NULL;
    double in__val = 0;
    PyArrayObject* in__scalar_arr = NULL;
    int in__scalar_copyback = 0;
    int in__scalar_is_array = 0;
    double out_val = 0;
    static char *kwlist[] = {"in_", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_in_)) {
        return NULL;
    }
    
    double* in_ = &in__val;
    if (PyArray_Check(py_in_)) {
        in__scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_in_, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (in__scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(in__scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument in_ must have exactly one element");
            Py_DECREF(in__scalar_arr);
            return NULL;
        }
        in__scalar_is_array = 1;
        in_ = (double*)PyArray_DATA(in__scalar_arr);
        in__val = in_[0];
        if (PyArray_DATA(in__scalar_arr) != PyArray_DATA((PyArrayObject*)py_in_) || PyArray_TYPE(in__scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_in_)) {
            in__scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_in_)) {
        in__val = (double)PyFloat_AsDouble(py_in_);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument in_ must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_top_level)(in_, &out_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (in__scalar_is_array) {
        if (in__scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_in_, in__scalar_arr) < 0) {
                Py_DECREF(in__scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(in__scalar_arr);
    }
    PyObject* py_out_obj = Py_BuildValue("d", out_val);
    if (py_out_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_out_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_out_obj != NULL) return py_out_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_out_obj != NULL) Py_DECREF(py_out_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_out_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_out_obj);
    }
    return result_tuple;
}

static PyObject* wrap_define_a_type_helper_get_a_set_real(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__get__a_set_real)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_define_a_type_helper_set_a_set_real(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"a_set_real", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__set__a_set_real)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_define_a_type_helper_get_a_set_bool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__get__a_set_bool)(&value);
    return PyBool_FromLong(value);
}

static PyObject* wrap_define_a_type_helper_set_a_set_bool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"a_set_bool", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "p", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__set__a_set_bool)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_define_a_type__atype_helper_get_bool(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__get__bool)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_define_a_type__atype_helper_set_bool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "bool", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Op", kwlist, &py_handle, &value)) {
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__set__bool)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_define_a_type__atype_helper_get_integ(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__get__integ)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_define_a_type__atype_helper_set_integ(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "integ", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__set__integ)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_define_a_type__atype_helper_get_rl(PyObject* self, PyObject* args, PyObject* kwargs)
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
    double value;
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__get__rl)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_define_a_type__atype_helper_set_rl(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "rl", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__set__rl)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_define_a_type__atype_helper_array_vec(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__array__vec)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_define_a_type__atype_helper_get_derived_dtype(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__get__dtype)(handle_handle, value_handle);
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

static PyObject* wrap_define_a_type__atype_helper_set_derived_dtype(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__atype__set__dtype)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_define_a_type__unused_type_helper_get_rl(PyObject* self, PyObject* args, PyObject* kwargs)
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
    double value;
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__unused_type__get__rl)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_define_a_type__unused_type_helper_set_rl(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "rl", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_define_a_type__unused_type__set__rl)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_horrible_helper_get_a_real(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_horrible__get__a_real)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_horrible_helper_set_a_real(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"a_real", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_horrible__set__a_real)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_horrible__horrible_type_helper_array_x(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_horrible__horrible_type__array__x)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_use_a_type_helper_get_derived_p(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if (args && PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "Getters do not take arguments");
        return NULL;
    }
    int value_handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_use_a_type__get__p)(value_handle);
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

static PyObject* wrap_use_a_type_helper_set_derived_p(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent = Py_None;
    PyObject* py_value = Py_None;
    static char *kwlist[] = {"value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_value)) {
        return NULL;
    }
    
    int parent_handle[4] = {0};
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
    F90WRAP_F_SYMBOL(f90wrap_use_a_type__set__p)(value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_use_a_type_helper_array_getitem_p_array(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    static char *kwlist[] = {"handle", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", kwlist, &py_parent, &index)) {
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
    int handle[4] = {0};
    F90WRAP_F_SYMBOL(f90wrap_use_a_type__array_getitem__p_array)(parent_handle, &index, handle);
    if (PyErr_Occurred()) {
        Py_DECREF(parent_sequence);
        return NULL;
    }
    Py_DECREF(parent_sequence);
    PyObject* result = PyList_New(4);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)handle[i]);
        if (item == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, item);
    }
    return result;
}

static PyObject* wrap_use_a_type_helper_array_setitem_p_array(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    int index = 0;
    PyObject* py_value;
    static char *kwlist[] = {"handle", "index", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiO", kwlist, &py_parent, &index, &py_value)) {
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
    PyObject* value_handle_obj = NULL;
    PyObject* value_sequence = NULL;
    Py_ssize_t value_handle_len = 0;
    if (PyObject_HasAttrString(py_value, "_handle")) {
        value_handle_obj = PyObject_GetAttrString(py_value, "_handle");
        if (value_handle_obj == NULL) { return NULL; }
        value_sequence = PySequence_Fast(value_handle_obj, "Failed to access handle sequence");
        if (value_sequence == NULL) { Py_DECREF(value_handle_obj); return NULL; }
    } else if (PySequence_Check(py_value)) {
        value_sequence = PySequence_Fast(py_value, "Argument value must be a handle sequence");
        if (value_sequence == NULL) { return NULL; }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument value must be a Fortran derived-type instance");
        return NULL;
    }
    value_handle_len = PySequence_Fast_GET_SIZE(value_sequence);
    if (value_handle_len != 4) {
        Py_DECREF(parent_sequence);
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        PyErr_SetString(PyExc_ValueError, "Unexpected handle length");
        return NULL;
    }
    Py_DECREF(parent_sequence);
    int* value = (int*)malloc(sizeof(int) * 4);
    if (value == NULL) {
        PyErr_NoMemory();
        Py_DECREF(value_sequence);
        if (value_handle_obj) Py_DECREF(value_handle_obj);
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(value_sequence, i);
        if (item == NULL) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
        value[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(value);
            Py_DECREF(value_sequence);
            if (value_handle_obj) Py_DECREF(value_handle_obj);
            return NULL;
        }
    }
    F90WRAP_F_SYMBOL(f90wrap_use_a_type__array_setitem__p_array)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_use_a_type_helper_array_len_p_array(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_parent;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_parent)) {
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
    int length = 0;
    F90WRAP_F_SYMBOL(f90wrap_use_a_type__array_len__p_array)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_use_a_type_helper_array_vector(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_use_a_type__array__vector)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_leveltwomod__leveltwo_helper_get_rl(PyObject* self, PyObject* args, PyObject* kwargs)
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
    double value;
    F90WRAP_F_SYMBOL(f90wrap_leveltwomod__leveltwo__get__rl)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_leveltwomod__leveltwo_helper_set_rl(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "rl", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_leveltwomod__leveltwo__set__rl)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* Method table for _mockdt module */
static PyMethodDef _mockdt_methods[] = {
    {"f90wrap_define_a_type__use_set_vars", (PyCFunction)wrap_define_a_type_use_set_vars, METH_VARARGS | METH_KEYWORDS, \
        "This type will be wrapped as it is used."},
    {"f90wrap_define_a_type__return_a_type_func", (PyCFunction)wrap_define_a_type_return_a_type_func, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_a_type_func"},
    {"f90wrap_define_a_type__return_a_type_sub", (PyCFunction)wrap_define_a_type_return_a_type_sub, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_a_type_sub"},
    {"f90wrap_define_a_type__atype_initialise", (PyCFunction)wrap_define_a_type_atype_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for atype"},
    {"f90wrap_define_a_type__atype_finalise", (PyCFunction)wrap_define_a_type_atype_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for atype"},
    {"f90wrap_define_a_type__unused_type_initialise", (PyCFunction)wrap_define_a_type_unused_type_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for unused_type"},
    {"f90wrap_define_a_type__unused_type_finalise", (PyCFunction)wrap_define_a_type_unused_type_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for unused_type"},
    {"f90wrap_horrible__horrible_type_initialise", (PyCFunction)wrap_horrible_horrible_type_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for horrible_type"},
    {"f90wrap_horrible__horrible_type_finalise", (PyCFunction)wrap_horrible_horrible_type_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for horrible_type"},
    {"f90wrap_use_a_type__do_stuff", (PyCFunction)wrap_use_a_type_do_stuff, METH_VARARGS | METH_KEYWORDS, "This is the \
        module which defines the type 'atype'"},
    {"f90wrap_use_a_type__not_used", (PyCFunction)wrap_use_a_type_not_used, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        not_used"},
    {"f90wrap_leveltwomod__leveltwo_initialise", (PyCFunction)wrap_leveltwomod_leveltwo_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for leveltwo"},
    {"f90wrap_leveltwomod__leveltwo_finalise", (PyCFunction)wrap_leveltwomod_leveltwo_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for leveltwo"},
    {"f90wrap_top_level", (PyCFunction)wrap__mockdt_top_level, METH_VARARGS | METH_KEYWORDS, "Example of a top-level \
        subroutine."},
    {"f90wrap_define_a_type__get__a_set_real", (PyCFunction)wrap_define_a_type_helper_get_a_set_real, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for a_set_real"},
    {"f90wrap_define_a_type__set__a_set_real", (PyCFunction)wrap_define_a_type_helper_set_a_set_real, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for a_set_real"},
    {"f90wrap_define_a_type__get__a_set_bool", (PyCFunction)wrap_define_a_type_helper_get_a_set_bool, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for a_set_bool"},
    {"f90wrap_define_a_type__set__a_set_bool", (PyCFunction)wrap_define_a_type_helper_set_a_set_bool, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for a_set_bool"},
    {"f90wrap_define_a_type__atype__get__bool", (PyCFunction)wrap_define_a_type__atype_helper_get_bool, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for bool"},
    {"f90wrap_define_a_type__atype__set__bool", (PyCFunction)wrap_define_a_type__atype_helper_set_bool, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for bool"},
    {"f90wrap_define_a_type__atype__get__integ", (PyCFunction)wrap_define_a_type__atype_helper_get_integ, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for integ"},
    {"f90wrap_define_a_type__atype__set__integ", (PyCFunction)wrap_define_a_type__atype_helper_set_integ, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for integ"},
    {"f90wrap_define_a_type__atype__get__rl", (PyCFunction)wrap_define_a_type__atype_helper_get_rl, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for rl"},
    {"f90wrap_define_a_type__atype__set__rl", (PyCFunction)wrap_define_a_type__atype_helper_set_rl, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for rl"},
    {"f90wrap_define_a_type__atype__array__vec", (PyCFunction)wrap_define_a_type__atype_helper_array_vec, METH_VARARGS | \
        METH_KEYWORDS, "Array helper for vec"},
    {"f90wrap_define_a_type__atype__get__dtype", (PyCFunction)wrap_define_a_type__atype_helper_get_derived_dtype, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for dtype"},
    {"f90wrap_define_a_type__atype__set__dtype", (PyCFunction)wrap_define_a_type__atype_helper_set_derived_dtype, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for dtype"},
    {"f90wrap_define_a_type__unused_type__get__rl", (PyCFunction)wrap_define_a_type__unused_type_helper_get_rl, METH_VARARGS \
        | METH_KEYWORDS, "Module helper for rl"},
    {"f90wrap_define_a_type__unused_type__set__rl", (PyCFunction)wrap_define_a_type__unused_type_helper_set_rl, METH_VARARGS \
        | METH_KEYWORDS, "Module helper for rl"},
    {"f90wrap_horrible__get__a_real", (PyCFunction)wrap_horrible_helper_get_a_real, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for a_real"},
    {"f90wrap_horrible__set__a_real", (PyCFunction)wrap_horrible_helper_set_a_real, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for a_real"},
    {"f90wrap_horrible__horrible_type__array__x", (PyCFunction)wrap_horrible__horrible_type_helper_array_x, METH_VARARGS | \
        METH_KEYWORDS, "Array helper for x"},
    {"f90wrap_use_a_type__get__p", (PyCFunction)wrap_use_a_type_helper_get_derived_p, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for p"},
    {"f90wrap_use_a_type__set__p", (PyCFunction)wrap_use_a_type_helper_set_derived_p, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for p"},
    {"f90wrap_use_a_type__array_getitem__p_array", (PyCFunction)wrap_use_a_type_helper_array_getitem_p_array, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for p_array"},
    {"f90wrap_use_a_type__array_setitem__p_array", (PyCFunction)wrap_use_a_type_helper_array_setitem_p_array, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for p_array"},
    {"f90wrap_use_a_type__array_len__p_array", (PyCFunction)wrap_use_a_type_helper_array_len_p_array, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for p_array"},
    {"f90wrap_use_a_type__array__vector", (PyCFunction)wrap_use_a_type_helper_array_vector, METH_VARARGS | METH_KEYWORDS, \
        "Array helper for vector"},
    {"f90wrap_leveltwomod__leveltwo__get__rl", (PyCFunction)wrap_leveltwomod__leveltwo_helper_get_rl, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for rl"},
    {"f90wrap_leveltwomod__leveltwo__set__rl", (PyCFunction)wrap_leveltwomod__leveltwo_helper_set_rl, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for rl"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _mockdtmodule = {
    PyModuleDef_HEAD_INIT,
    "mockdt",
    "Direct-C wrapper for _mockdt module",
    -1,
    _mockdt_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__mockdt(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_mockdtmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
