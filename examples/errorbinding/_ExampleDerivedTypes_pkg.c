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
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__constructor_typewithprocedure)(int* this, double* a, int* n);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__info_typewithprocedure)(int* this, int* lun);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure__get__a)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure__set__a)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure__get__n)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure__set__n)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_parameters__get__idp)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_parameters__get__isp)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__init__binding__typewithprocedure)(int* this, double* a, int* n);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__info__binding__typewithprocedure)(int* this, int* lun);

static PyObject* wrap_datatypes_constructor_typewithprocedure(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_this = NULL;
    PyObject* py_a = NULL;
    double a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"this", "a", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_this, &py_a, &py_n)) {
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
    
    double* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (double*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (double)PyFloat_AsDouble(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    int* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__constructor_typewithprocedure)(this, a, n);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
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
    if (n_scalar_is_array) {
        if (n_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n, n_scalar_arr) < 0) {
                Py_DECREF(n_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_scalar_arr);
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_info_typewithprocedure(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_this = NULL;
    PyObject* py_lun = NULL;
    int lun_val = 0;
    PyArrayObject* lun_scalar_arr = NULL;
    int lun_scalar_copyback = 0;
    int lun_scalar_is_array = 0;
    static char *kwlist[] = {"this", "lun", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_lun)) {
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
    
    int* lun = &lun_val;
    if (PyArray_Check(py_lun)) {
        lun_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_lun, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (lun_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(lun_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument lun must have exactly one element");
            Py_DECREF(lun_scalar_arr);
            return NULL;
        }
        lun_scalar_is_array = 1;
        lun = (int*)PyArray_DATA(lun_scalar_arr);
        lun_val = lun[0];
        if (PyArray_DATA(lun_scalar_arr) != PyArray_DATA((PyArrayObject*)py_lun) || PyArray_TYPE(lun_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_lun)) {
            lun_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_lun)) {
        lun_val = (int)PyLong_AsLong(py_lun);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument lun must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__info_typewithprocedure)(this, lun);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    if (lun_scalar_is_array) {
        if (lun_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_lun, lun_scalar_arr) < 0) {
                Py_DECREF(lun_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(lun_scalar_arr);
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_typewithprocedure_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure_initialise)(this);
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

static PyObject* wrap_datatypes_typewithprocedure_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__typewithprocedure_helper_get_a(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure__get__a)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_datatypes__typewithprocedure_helper_set_a(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure__set__a)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__typewithprocedure_helper_get_n(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure__get__n)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_datatypes__typewithprocedure_helper_set_n(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "n", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__typewithprocedure__set__n)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_parameters_helper_get_idp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_parameters__get__idp)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_parameters_helper_get_isp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_parameters__get__isp)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap__datatypes__init__binding__typewithprocedure(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_a = NULL;
    double a_val = 0;
    PyArrayObject* a_scalar_arr = NULL;
    int a_scalar_copyback = 0;
    int a_scalar_is_array = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"this", "a", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_this, &py_a, &py_n)) {
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
    
    double* a = &a_val;
    if (PyArray_Check(py_a)) {
        a_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_a, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (a_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(a_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument a must have exactly one element");
            Py_DECREF(a_scalar_arr);
            return NULL;
        }
        a_scalar_is_array = 1;
        a = (double*)PyArray_DATA(a_scalar_arr);
        a_val = a[0];
        if (PyArray_DATA(a_scalar_arr) != PyArray_DATA((PyArrayObject*)py_a) || PyArray_TYPE(a_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_a)) {
            a_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_a)) {
        a_val = (double)PyFloat_AsDouble(py_a);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument a must be a scalar number or NumPy array");
        return NULL;
    }
    int* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__init__binding__typewithprocedure)(this, a, n);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
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
    if (n_scalar_is_array) {
        if (n_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n, n_scalar_arr) < 0) {
                Py_DECREF(n_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n_scalar_arr);
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap__datatypes__info__binding__typewithprocedure(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_lun = NULL;
    int lun_val = 0;
    PyArrayObject* lun_scalar_arr = NULL;
    int lun_scalar_copyback = 0;
    int lun_scalar_is_array = 0;
    static char *kwlist[] = {"this", "lun", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_lun)) {
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
    
    int* lun = &lun_val;
    if (PyArray_Check(py_lun)) {
        lun_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_lun, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (lun_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(lun_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument lun must have exactly one element");
            Py_DECREF(lun_scalar_arr);
            return NULL;
        }
        lun_scalar_is_array = 1;
        lun = (int*)PyArray_DATA(lun_scalar_arr);
        lun_val = lun[0];
        if (PyArray_DATA(lun_scalar_arr) != PyArray_DATA((PyArrayObject*)py_lun) || PyArray_TYPE(lun_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_lun)) {
            lun_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_lun)) {
        lun_val = (int)PyLong_AsLong(py_lun);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument lun must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__info__binding__typewithprocedure)(this, lun);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    if (lun_scalar_is_array) {
        if (lun_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_lun, lun_scalar_arr) < 0) {
                Py_DECREF(lun_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(lun_scalar_arr);
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

/* Method table for _ExampleDerivedTypes_pkg module */
static PyMethodDef _ExampleDerivedTypes_pkg_methods[] = {
    {"f90wrap_datatypes__constructor_typewithprocedure", (PyCFunction)wrap_datatypes_constructor_typewithprocedure, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for constructor_typewithprocedure"},
    {"f90wrap_datatypes__info_typewithprocedure", (PyCFunction)wrap_datatypes_info_typewithprocedure, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for info_typewithprocedure"},
    {"f90wrap_datatypes__typewithprocedure_initialise", (PyCFunction)wrap_datatypes_typewithprocedure_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for typewithprocedure"},
    {"f90wrap_datatypes__typewithprocedure_finalise", (PyCFunction)wrap_datatypes_typewithprocedure_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for typewithprocedure"},
    {"f90wrap_datatypes__typewithprocedure__get__a", (PyCFunction)wrap_datatypes__typewithprocedure_helper_get_a, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a"},
    {"f90wrap_datatypes__typewithprocedure__set__a", (PyCFunction)wrap_datatypes__typewithprocedure_helper_set_a, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for a"},
    {"f90wrap_datatypes__typewithprocedure__get__n", (PyCFunction)wrap_datatypes__typewithprocedure_helper_get_n, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for n"},
    {"f90wrap_datatypes__typewithprocedure__set__n", (PyCFunction)wrap_datatypes__typewithprocedure_helper_set_n, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for n"},
    {"f90wrap_parameters__get__idp", (PyCFunction)wrap_parameters_helper_get_idp, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for idp"},
    {"f90wrap_parameters__get__isp", (PyCFunction)wrap_parameters_helper_get_isp, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for isp"},
    {"f90wrap_datatypes__init__binding__typewithprocedure", (PyCFunction)wrap__datatypes__init__binding__typewithprocedure, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for init"},
    {"f90wrap_datatypes__info__binding__typewithprocedure", (PyCFunction)wrap__datatypes__info__binding__typewithprocedure, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for info"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _ExampleDerivedTypes_pkgmodule = {
    PyModuleDef_HEAD_INIT,
    "ExampleDerivedTypes_pkg",
    "Direct-C wrapper for _ExampleDerivedTypes_pkg module",
    -1,
    _ExampleDerivedTypes_pkg_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__ExampleDerivedTypes_pkg(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_ExampleDerivedTypes_pkgmodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
