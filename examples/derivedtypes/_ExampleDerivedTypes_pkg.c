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
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__init_alloc_arrays)(int* dertype, int* m, int* n);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__destroy_alloc_arrays)(int* dertype);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__init_array_nested)(int* dertype, int* size_bn);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__destroy_array_nested)(int* dertype);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__nested_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__nested_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_library__return_value_func)(int* val_in, int* ret_val_out);
extern void F90WRAP_F_SYMBOL(f90wrap_library__return_value_sub)(int* val_in, int* val_out);
extern void F90WRAP_F_SYMBOL(f90wrap_library__return_a_dt_func)(int* ret_dt);
extern void F90WRAP_F_SYMBOL(f90wrap_library__do_array_stuff)(int* f90wrap_n0, int* f90wrap_n1, int* f90wrap_n2, int* \
    f90wrap_n3, int* n, double* x, double* y, double* br, double* co);
extern void F90WRAP_F_SYMBOL(f90wrap_library__only_manipulate)(int* f90wrap_n0, int* n, double* array);
extern void F90WRAP_F_SYMBOL(f90wrap_library__set_derived_type)(int* dt, int* dt_beta, double* dt_delta);
extern void F90WRAP_F_SYMBOL(f90wrap_library__modify_derived_types)(int* dt1, int* dt2, int* dt3);
extern void F90WRAP_F_SYMBOL(f90wrap_library__modify_dertype_fixed_shape_arrays)(int* dertype);
extern void F90WRAP_F_SYMBOL(f90wrap_library__return_dertype_pointer_arrays)(int* m, int* n, int* dertype);
extern void F90WRAP_F_SYMBOL(f90wrap_library__modify_dertype_pointer_arrays)(int* dertype);
extern void F90WRAP_F_SYMBOL(f90wrap_library__return_dertype_alloc_arrays)(int* m, int* n, int* dertype);
extern void F90WRAP_F_SYMBOL(f90wrap_library__modify_dertype_alloc_arrays)(int* dertype);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays__array__chi)(int* dummy_this, int* nd, int* \
    dtype, int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays__array__psi)(int* dummy_this, int* nd, int* \
    dtype, int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays__array__chi_shape)(int* dummy_this, int* nd, \
    int* dtype, int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays__array__psi_shape)(int* dummy_this, int* nd, \
    int* dtype, int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__get__alpha)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__set__alpha)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__get__beta)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__set__beta)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__get__delta)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__set__delta)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays__array__eta)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays__array__theta)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays__array__iota)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__nested__get__mu)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__nested__set__mu)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__nested__get__nu)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__nested__set__nu)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays__array__chi)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays__array__psi)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays__array__chi_shape)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays__array__psi_shape)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2__array__chi)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2__array__psi)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2__array__chi_shape)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2__array__psi_shape)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_getitem__xi)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_setitem__xi)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_len__xi)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_getitem__omicron)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_setitem__omicron)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_len__omicron)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_getitem__pi)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_setitem__pi)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_len__pi)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_parameters__get__idp)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_parameters__get__isp)(int* value);

static PyObject* wrap_datatypes_allocatable_init_alloc_arrays(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_dertype = NULL;
    PyObject* py_m = NULL;
    int m_val = 0;
    PyArrayObject* m_scalar_arr = NULL;
    int m_scalar_copyback = 0;
    int m_scalar_is_array = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"dertype", "m", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_dertype, &py_m, &py_n)) {
        return NULL;
    }
    
    PyObject* dertype_handle_obj = NULL;
    PyObject* dertype_sequence = NULL;
    Py_ssize_t dertype_handle_len = 0;
    if (PyObject_HasAttrString(py_dertype, "_handle")) {
        dertype_handle_obj = PyObject_GetAttrString(py_dertype, "_handle");
        if (dertype_handle_obj == NULL) {
            return NULL;
        }
        dertype_sequence = PySequence_Fast(dertype_handle_obj, "Failed to access handle sequence");
        if (dertype_sequence == NULL) {
            Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dertype)) {
        dertype_sequence = PySequence_Fast(py_dertype, "Argument dertype must be a handle sequence");
        if (dertype_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dertype must be a Fortran derived-type instance");
        return NULL;
    }
    dertype_handle_len = PySequence_Fast_GET_SIZE(dertype_sequence);
    if (dertype_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dertype has an invalid handle length");
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    int* dertype = (int*)malloc(sizeof(int) * dertype_handle_len);
    if (dertype == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dertype_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dertype_sequence, i);
        if (item == NULL) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
        dertype[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    }
    (void)dertype_handle_len;  /* suppress unused warnings when unchanged */
    
    int* m = &m_val;
    if (PyArray_Check(py_m)) {
        m_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_m, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (m_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(m_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument m must have exactly one element");
            Py_DECREF(m_scalar_arr);
            return NULL;
        }
        m_scalar_is_array = 1;
        m = (int*)PyArray_DATA(m_scalar_arr);
        m_val = m[0];
        if (PyArray_DATA(m_scalar_arr) != PyArray_DATA((PyArrayObject*)py_m) || PyArray_TYPE(m_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_m)) {
            m_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_m)) {
        m_val = (int)PyLong_AsLong(py_m);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument m must be a scalar number or NumPy array");
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__init_alloc_arrays)(dertype, m, n);
    if (PyErr_Occurred()) {
        if (dertype_sequence) Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        free(dertype);
        return NULL;
    }
    
    if (m_scalar_is_array) {
        if (m_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_m, m_scalar_arr) < 0) {
                Py_DECREF(m_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(m_scalar_arr);
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
    if (dertype_sequence) {
        Py_DECREF(dertype_sequence);
    }
    if (dertype_handle_obj) {
        Py_DECREF(dertype_handle_obj);
    }
    free(dertype);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_allocatable_destroy_alloc_arrays(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_dertype = NULL;
    static char *kwlist[] = {"dertype", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_dertype)) {
        return NULL;
    }
    
    PyObject* dertype_handle_obj = NULL;
    PyObject* dertype_sequence = NULL;
    Py_ssize_t dertype_handle_len = 0;
    if (PyObject_HasAttrString(py_dertype, "_handle")) {
        dertype_handle_obj = PyObject_GetAttrString(py_dertype, "_handle");
        if (dertype_handle_obj == NULL) {
            return NULL;
        }
        dertype_sequence = PySequence_Fast(dertype_handle_obj, "Failed to access handle sequence");
        if (dertype_sequence == NULL) {
            Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dertype)) {
        dertype_sequence = PySequence_Fast(py_dertype, "Argument dertype must be a handle sequence");
        if (dertype_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dertype must be a Fortran derived-type instance");
        return NULL;
    }
    dertype_handle_len = PySequence_Fast_GET_SIZE(dertype_sequence);
    if (dertype_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dertype has an invalid handle length");
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    int* dertype = (int*)malloc(sizeof(int) * dertype_handle_len);
    if (dertype == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dertype_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dertype_sequence, i);
        if (item == NULL) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
        dertype[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    }
    (void)dertype_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__destroy_alloc_arrays)(dertype);
    if (PyErr_Occurred()) {
        if (dertype_sequence) Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        free(dertype);
        return NULL;
    }
    
    if (dertype_sequence) {
        Py_DECREF(dertype_sequence);
    }
    if (dertype_handle_obj) {
        Py_DECREF(dertype_handle_obj);
    }
    free(dertype);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_allocatable_alloc_arrays_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays_initialise)(this);
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

static PyObject* wrap_datatypes_allocatable_alloc_arrays_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_init_array_nested(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_dertype = NULL;
    PyObject* py_size_bn = NULL;
    int size_bn_val = 0;
    PyArrayObject* size_bn_scalar_arr = NULL;
    int size_bn_scalar_copyback = 0;
    int size_bn_scalar_is_array = 0;
    static char *kwlist[] = {"dertype", "size_bn", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_dertype, &py_size_bn)) {
        return NULL;
    }
    
    PyObject* dertype_handle_obj = NULL;
    PyObject* dertype_sequence = NULL;
    Py_ssize_t dertype_handle_len = 0;
    if (PyObject_HasAttrString(py_dertype, "_handle")) {
        dertype_handle_obj = PyObject_GetAttrString(py_dertype, "_handle");
        if (dertype_handle_obj == NULL) {
            return NULL;
        }
        dertype_sequence = PySequence_Fast(dertype_handle_obj, "Failed to access handle sequence");
        if (dertype_sequence == NULL) {
            Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dertype)) {
        dertype_sequence = PySequence_Fast(py_dertype, "Argument dertype must be a handle sequence");
        if (dertype_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dertype must be a Fortran derived-type instance");
        return NULL;
    }
    dertype_handle_len = PySequence_Fast_GET_SIZE(dertype_sequence);
    if (dertype_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dertype has an invalid handle length");
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    int* dertype = (int*)malloc(sizeof(int) * dertype_handle_len);
    if (dertype == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dertype_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dertype_sequence, i);
        if (item == NULL) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
        dertype[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    }
    (void)dertype_handle_len;  /* suppress unused warnings when unchanged */
    
    int* size_bn = &size_bn_val;
    if (PyArray_Check(py_size_bn)) {
        size_bn_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_size_bn, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (size_bn_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(size_bn_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument size_bn must have exactly one element");
            Py_DECREF(size_bn_scalar_arr);
            return NULL;
        }
        size_bn_scalar_is_array = 1;
        size_bn = (int*)PyArray_DATA(size_bn_scalar_arr);
        size_bn_val = size_bn[0];
        if (PyArray_DATA(size_bn_scalar_arr) != PyArray_DATA((PyArrayObject*)py_size_bn) || PyArray_TYPE(size_bn_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_size_bn)) {
            size_bn_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_size_bn)) {
        size_bn_val = (int)PyLong_AsLong(py_size_bn);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument size_bn must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__init_array_nested)(dertype, size_bn);
    if (PyErr_Occurred()) {
        if (dertype_sequence) Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        free(dertype);
        return NULL;
    }
    
    if (size_bn_scalar_is_array) {
        if (size_bn_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_size_bn, size_bn_scalar_arr) < 0) {
                Py_DECREF(size_bn_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(size_bn_scalar_arr);
    }
    if (dertype_sequence) {
        Py_DECREF(dertype_sequence);
    }
    if (dertype_handle_obj) {
        Py_DECREF(dertype_handle_obj);
    }
    free(dertype);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_destroy_array_nested(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_dertype = NULL;
    static char *kwlist[] = {"dertype", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_dertype)) {
        return NULL;
    }
    
    PyObject* dertype_handle_obj = NULL;
    PyObject* dertype_sequence = NULL;
    Py_ssize_t dertype_handle_len = 0;
    if (PyObject_HasAttrString(py_dertype, "_handle")) {
        dertype_handle_obj = PyObject_GetAttrString(py_dertype, "_handle");
        if (dertype_handle_obj == NULL) {
            return NULL;
        }
        dertype_sequence = PySequence_Fast(dertype_handle_obj, "Failed to access handle sequence");
        if (dertype_sequence == NULL) {
            Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dertype)) {
        dertype_sequence = PySequence_Fast(py_dertype, "Argument dertype must be a handle sequence");
        if (dertype_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dertype must be a Fortran derived-type instance");
        return NULL;
    }
    dertype_handle_len = PySequence_Fast_GET_SIZE(dertype_sequence);
    if (dertype_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dertype has an invalid handle length");
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    int* dertype = (int*)malloc(sizeof(int) * dertype_handle_len);
    if (dertype == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dertype_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dertype_sequence, i);
        if (item == NULL) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
        dertype[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    }
    (void)dertype_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__destroy_array_nested)(dertype);
    if (PyErr_Occurred()) {
        if (dertype_sequence) Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        free(dertype);
        return NULL;
    }
    
    if (dertype_sequence) {
        Py_DECREF(dertype_sequence);
    }
    if (dertype_handle_obj) {
        Py_DECREF(dertype_handle_obj);
    }
    free(dertype);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_different_types_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types_initialise)(this);
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

static PyObject* wrap_datatypes_different_types_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_fixed_shape_arrays_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays_initialise)(this);
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

static PyObject* wrap_datatypes_fixed_shape_arrays_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_nested_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__nested_initialise)(this);
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

static PyObject* wrap_datatypes_nested_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__nested_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_pointer_arrays_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays_initialise)(this);
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

static PyObject* wrap_datatypes_pointer_arrays_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_alloc_arrays_2_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2_initialise)(this);
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

static PyObject* wrap_datatypes_alloc_arrays_2_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_array_nested_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested_initialise)(this);
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

static PyObject* wrap_datatypes_array_nested_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_library_return_value_func(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_val_in = NULL;
    int val_in_val = 0;
    PyArrayObject* val_in_scalar_arr = NULL;
    int val_in_scalar_copyback = 0;
    int val_in_scalar_is_array = 0;
    int ret_val_out_val = 0;
    static char *kwlist[] = {"val_in", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_val_in)) {
        return NULL;
    }
    
    int* val_in = &val_in_val;
    if (PyArray_Check(py_val_in)) {
        val_in_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_val_in, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (val_in_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(val_in_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument val_in must have exactly one element");
            Py_DECREF(val_in_scalar_arr);
            return NULL;
        }
        val_in_scalar_is_array = 1;
        val_in = (int*)PyArray_DATA(val_in_scalar_arr);
        val_in_val = val_in[0];
        if (PyArray_DATA(val_in_scalar_arr) != PyArray_DATA((PyArrayObject*)py_val_in) || PyArray_TYPE(val_in_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_val_in)) {
            val_in_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_val_in)) {
        val_in_val = (int)PyLong_AsLong(py_val_in);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val_in must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__return_value_func)(val_in, &ret_val_out_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (val_in_scalar_is_array) {
        if (val_in_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_val_in, val_in_scalar_arr) < 0) {
                Py_DECREF(val_in_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(val_in_scalar_arr);
    }
    PyObject* py_ret_val_out_obj = Py_BuildValue("i", ret_val_out_val);
    if (py_ret_val_out_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_val_out_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_val_out_obj != NULL) return py_ret_val_out_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_val_out_obj != NULL) Py_DECREF(py_ret_val_out_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_val_out_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_val_out_obj);
    }
    return result_tuple;
}

static PyObject* wrap_library_return_value_sub(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_val_in = NULL;
    int val_in_val = 0;
    PyArrayObject* val_in_scalar_arr = NULL;
    int val_in_scalar_copyback = 0;
    int val_in_scalar_is_array = 0;
    int val_out_val = 0;
    static char *kwlist[] = {"val_in", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_val_in)) {
        return NULL;
    }
    
    int* val_in = &val_in_val;
    if (PyArray_Check(py_val_in)) {
        val_in_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_val_in, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (val_in_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(val_in_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument val_in must have exactly one element");
            Py_DECREF(val_in_scalar_arr);
            return NULL;
        }
        val_in_scalar_is_array = 1;
        val_in = (int*)PyArray_DATA(val_in_scalar_arr);
        val_in_val = val_in[0];
        if (PyArray_DATA(val_in_scalar_arr) != PyArray_DATA((PyArrayObject*)py_val_in) || PyArray_TYPE(val_in_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_val_in)) {
            val_in_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_val_in)) {
        val_in_val = (int)PyLong_AsLong(py_val_in);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val_in must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__return_value_sub)(val_in, &val_out_val);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (val_in_scalar_is_array) {
        if (val_in_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_val_in, val_in_scalar_arr) < 0) {
                Py_DECREF(val_in_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(val_in_scalar_arr);
    }
    PyObject* py_val_out_obj = Py_BuildValue("i", val_out_val);
    if (py_val_out_obj == NULL) {
        return NULL;
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_val_out_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_val_out_obj != NULL) return py_val_out_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_val_out_obj != NULL) Py_DECREF(py_val_out_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_val_out_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_val_out_obj);
    }
    return result_tuple;
}

static PyObject* wrap_library_return_a_dt_func(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int ret_dt[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__return_a_dt_func)(ret_dt);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_ret_dt_obj = PyList_New(4);
    if (py_ret_dt_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_dt[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_dt_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_dt_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_dt_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_dt_obj != NULL) return py_ret_dt_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_dt_obj != NULL) Py_DECREF(py_ret_dt_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_dt_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_dt_obj);
    }
    return result_tuple;
}

static PyObject* wrap_library_do_array_stuff(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    int f90wrap_n1_val = 0;
    int f90wrap_n2_val = 0;
    int f90wrap_n3_val = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_x = NULL;
    PyObject* py_y = NULL;
    PyObject* py_br = NULL;
    PyObject* py_co = NULL;
    static char *kwlist[] = {"n", "x", "y", "br", "co", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO", kwlist, &py_n, &py_x, &py_y, &py_br, &py_co)) {
        return NULL;
    }
    
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
    PyArrayObject* x_arr = NULL;
    double* x = NULL;
    /* Extract x array data */
    if (!PyArray_Check(py_x)) {
        PyErr_SetString(PyExc_TypeError, "Argument x must be a NumPy array");
        return NULL;
    }
    x_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_x, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (x_arr == NULL) {
        return NULL;
    }
    x = (double*)PyArray_DATA(x_arr);
    int n0_x = (int)PyArray_DIM(x_arr, 0);
    f90wrap_n0_val = n0_x;
    
    PyArrayObject* y_arr = NULL;
    double* y = NULL;
    /* Extract y array data */
    if (!PyArray_Check(py_y)) {
        PyErr_SetString(PyExc_TypeError, "Argument y must be a NumPy array");
        return NULL;
    }
    y_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_y, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (y_arr == NULL) {
        return NULL;
    }
    y = (double*)PyArray_DATA(y_arr);
    int n0_y = (int)PyArray_DIM(y_arr, 0);
    f90wrap_n1_val = n0_y;
    
    PyArrayObject* br_arr = NULL;
    PyObject* py_br_arr = NULL;
    int br_needs_copyback = 0;
    double* br = NULL;
    /* Extract br array data */
    if (!PyArray_Check(py_br)) {
        PyErr_SetString(PyExc_TypeError, "Argument br must be a NumPy array");
        return NULL;
    }
    br_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_br, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (br_arr == NULL) {
        return NULL;
    }
    br = (double*)PyArray_DATA(br_arr);
    int n0_br = (int)PyArray_DIM(br_arr, 0);
    f90wrap_n2_val = n0_br;
    Py_INCREF(py_br);
    py_br_arr = py_br;
    if (PyArray_DATA(br_arr) != PyArray_DATA((PyArrayObject*)py_br) || PyArray_TYPE(br_arr) != \
        PyArray_TYPE((PyArrayObject*)py_br)) {
        br_needs_copyback = 1;
    }
    
    PyArrayObject* co_arr = NULL;
    PyObject* py_co_arr = NULL;
    int co_needs_copyback = 0;
    double* co = NULL;
    /* Extract co array data */
    if (!PyArray_Check(py_co)) {
        PyErr_SetString(PyExc_TypeError, "Argument co must be a NumPy array");
        return NULL;
    }
    co_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_co, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (co_arr == NULL) {
        return NULL;
    }
    co = (double*)PyArray_DATA(co_arr);
    int n0_co = (int)PyArray_DIM(co_arr, 0);
    int n1_co = (int)PyArray_DIM(co_arr, 1);
    f90wrap_n3_val = n1_co;
    Py_INCREF(py_co);
    py_co_arr = py_co;
    if (PyArray_DATA(co_arr) != PyArray_DATA((PyArrayObject*)py_co) || PyArray_TYPE(co_arr) != \
        PyArray_TYPE((PyArrayObject*)py_co)) {
        co_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__do_array_stuff)(&f90wrap_n0_val, &f90wrap_n1_val, &f90wrap_n2_val, &f90wrap_n3_val, n, \
        x, y, br, co);
    if (PyErr_Occurred()) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(py_br_arr);
        Py_XDECREF(py_co_arr);
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
    Py_DECREF(x_arr);
    Py_DECREF(y_arr);
    if (br_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_br, br_arr) < 0) {
            Py_DECREF(br_arr);
            Py_DECREF(py_br_arr);
            return NULL;
        }
    }
    Py_DECREF(br_arr);
    if (co_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_co, co_arr) < 0) {
            Py_DECREF(co_arr);
            Py_DECREF(py_co_arr);
            return NULL;
        }
    }
    Py_DECREF(co_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_br_arr != NULL) result_count++;
    if (py_co_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_br_arr != NULL) return py_br_arr;
        if (py_co_arr != NULL) return py_co_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_br_arr != NULL) Py_DECREF(py_br_arr);
        if (py_co_arr != NULL) Py_DECREF(py_co_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_br_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_br_arr);
    }
    if (py_co_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_co_arr);
    }
    return result_tuple;
}

static PyObject* wrap_library_only_manipulate(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int f90wrap_n0_val = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    PyObject* py_array = NULL;
    static char *kwlist[] = {"n", "array", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_n, &py_array)) {
        return NULL;
    }
    
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
    PyArrayObject* array_arr = NULL;
    PyObject* py_array_arr = NULL;
    int array_needs_copyback = 0;
    double* array = NULL;
    /* Extract array array data */
    if (!PyArray_Check(py_array)) {
        PyErr_SetString(PyExc_TypeError, "Argument array must be a NumPy array");
        return NULL;
    }
    array_arr = (PyArrayObject*)PyArray_FROM_OTF(
        py_array, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
    if (array_arr == NULL) {
        return NULL;
    }
    array = (double*)PyArray_DATA(array_arr);
    int n0_array = (int)PyArray_DIM(array_arr, 0);
    int n1_array = (int)PyArray_DIM(array_arr, 1);
    f90wrap_n0_val = n1_array;
    Py_INCREF(py_array);
    py_array_arr = py_array;
    if (PyArray_DATA(array_arr) != PyArray_DATA((PyArrayObject*)py_array) || PyArray_TYPE(array_arr) != \
        PyArray_TYPE((PyArrayObject*)py_array)) {
        array_needs_copyback = 1;
    }
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__only_manipulate)(&f90wrap_n0_val, n, array);
    if (PyErr_Occurred()) {
        Py_XDECREF(py_array_arr);
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
    if (array_needs_copyback) {
        if (PyArray_CopyInto((PyArrayObject*)py_array, array_arr) < 0) {
            Py_DECREF(array_arr);
            Py_DECREF(py_array_arr);
            return NULL;
        }
    }
    Py_DECREF(array_arr);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_array_arr != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_array_arr != NULL) return py_array_arr;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_array_arr != NULL) Py_DECREF(py_array_arr);
        return NULL;
    }
    int tuple_index = 0;
    if (py_array_arr != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_array_arr);
    }
    return result_tuple;
}

static PyObject* wrap_library_set_derived_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_dt_beta = NULL;
    int dt_beta_val = 0;
    PyArrayObject* dt_beta_scalar_arr = NULL;
    int dt_beta_scalar_copyback = 0;
    int dt_beta_scalar_is_array = 0;
    PyObject* py_dt_delta = NULL;
    double dt_delta_val = 0;
    PyArrayObject* dt_delta_scalar_arr = NULL;
    int dt_delta_scalar_copyback = 0;
    int dt_delta_scalar_is_array = 0;
    static char *kwlist[] = {"dt_beta", "dt_delta", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_dt_beta, &py_dt_delta)) {
        return NULL;
    }
    
    int dt[4] = {0};
    int* dt_beta = &dt_beta_val;
    if (PyArray_Check(py_dt_beta)) {
        dt_beta_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_dt_beta, NPY_INT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (dt_beta_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(dt_beta_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument dt_beta must have exactly one element");
            Py_DECREF(dt_beta_scalar_arr);
            return NULL;
        }
        dt_beta_scalar_is_array = 1;
        dt_beta = (int*)PyArray_DATA(dt_beta_scalar_arr);
        dt_beta_val = dt_beta[0];
        if (PyArray_DATA(dt_beta_scalar_arr) != PyArray_DATA((PyArrayObject*)py_dt_beta) || PyArray_TYPE(dt_beta_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_dt_beta)) {
            dt_beta_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_dt_beta)) {
        dt_beta_val = (int)PyLong_AsLong(py_dt_beta);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dt_beta must be a scalar number or NumPy array");
        return NULL;
    }
    double* dt_delta = &dt_delta_val;
    if (PyArray_Check(py_dt_delta)) {
        dt_delta_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_dt_delta, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (dt_delta_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(dt_delta_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument dt_delta must have exactly one element");
            Py_DECREF(dt_delta_scalar_arr);
            return NULL;
        }
        dt_delta_scalar_is_array = 1;
        dt_delta = (double*)PyArray_DATA(dt_delta_scalar_arr);
        dt_delta_val = dt_delta[0];
        if (PyArray_DATA(dt_delta_scalar_arr) != PyArray_DATA((PyArrayObject*)py_dt_delta) || PyArray_TYPE(dt_delta_scalar_arr) \
            != PyArray_TYPE((PyArrayObject*)py_dt_delta)) {
            dt_delta_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_dt_delta)) {
        dt_delta_val = (double)PyFloat_AsDouble(py_dt_delta);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dt_delta must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__set_derived_type)(dt, dt_beta, dt_delta);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (dt_beta_scalar_is_array) {
        if (dt_beta_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_dt_beta, dt_beta_scalar_arr) < 0) {
                Py_DECREF(dt_beta_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(dt_beta_scalar_arr);
    }
    if (dt_delta_scalar_is_array) {
        if (dt_delta_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_dt_delta, dt_delta_scalar_arr) < 0) {
                Py_DECREF(dt_delta_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(dt_delta_scalar_arr);
    }
    PyObject* py_dt_obj = PyList_New(4);
    if (py_dt_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)dt[i]);
        if (item == NULL) {
            Py_DECREF(py_dt_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_dt_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_dt_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_dt_obj != NULL) return py_dt_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_dt_obj != NULL) Py_DECREF(py_dt_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_dt_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_dt_obj);
    }
    return result_tuple;
}

static PyObject* wrap_library_modify_derived_types(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_dt1 = NULL;
    PyObject* py_dt2 = NULL;
    PyObject* py_dt3 = NULL;
    static char *kwlist[] = {"dt1", "dt2", "dt3", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_dt1, &py_dt2, &py_dt3)) {
        return NULL;
    }
    
    PyObject* dt1_handle_obj = NULL;
    PyObject* dt1_sequence = NULL;
    Py_ssize_t dt1_handle_len = 0;
    if (PyObject_HasAttrString(py_dt1, "_handle")) {
        dt1_handle_obj = PyObject_GetAttrString(py_dt1, "_handle");
        if (dt1_handle_obj == NULL) {
            return NULL;
        }
        dt1_sequence = PySequence_Fast(dt1_handle_obj, "Failed to access handle sequence");
        if (dt1_sequence == NULL) {
            Py_DECREF(dt1_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dt1)) {
        dt1_sequence = PySequence_Fast(py_dt1, "Argument dt1 must be a handle sequence");
        if (dt1_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dt1 must be a Fortran derived-type instance");
        return NULL;
    }
    dt1_handle_len = PySequence_Fast_GET_SIZE(dt1_sequence);
    if (dt1_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dt1 has an invalid handle length");
        Py_DECREF(dt1_sequence);
        if (dt1_handle_obj) Py_DECREF(dt1_handle_obj);
        return NULL;
    }
    int* dt1 = (int*)malloc(sizeof(int) * dt1_handle_len);
    if (dt1 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dt1_sequence);
        if (dt1_handle_obj) Py_DECREF(dt1_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dt1_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dt1_sequence, i);
        if (item == NULL) {
            free(dt1);
            Py_DECREF(dt1_sequence);
            if (dt1_handle_obj) Py_DECREF(dt1_handle_obj);
            return NULL;
        }
        dt1[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dt1);
            Py_DECREF(dt1_sequence);
            if (dt1_handle_obj) Py_DECREF(dt1_handle_obj);
            return NULL;
        }
    }
    (void)dt1_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* dt2_handle_obj = NULL;
    PyObject* dt2_sequence = NULL;
    Py_ssize_t dt2_handle_len = 0;
    if (PyObject_HasAttrString(py_dt2, "_handle")) {
        dt2_handle_obj = PyObject_GetAttrString(py_dt2, "_handle");
        if (dt2_handle_obj == NULL) {
            return NULL;
        }
        dt2_sequence = PySequence_Fast(dt2_handle_obj, "Failed to access handle sequence");
        if (dt2_sequence == NULL) {
            Py_DECREF(dt2_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dt2)) {
        dt2_sequence = PySequence_Fast(py_dt2, "Argument dt2 must be a handle sequence");
        if (dt2_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dt2 must be a Fortran derived-type instance");
        return NULL;
    }
    dt2_handle_len = PySequence_Fast_GET_SIZE(dt2_sequence);
    if (dt2_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dt2 has an invalid handle length");
        Py_DECREF(dt2_sequence);
        if (dt2_handle_obj) Py_DECREF(dt2_handle_obj);
        return NULL;
    }
    int* dt2 = (int*)malloc(sizeof(int) * dt2_handle_len);
    if (dt2 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dt2_sequence);
        if (dt2_handle_obj) Py_DECREF(dt2_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dt2_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dt2_sequence, i);
        if (item == NULL) {
            free(dt2);
            Py_DECREF(dt2_sequence);
            if (dt2_handle_obj) Py_DECREF(dt2_handle_obj);
            return NULL;
        }
        dt2[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dt2);
            Py_DECREF(dt2_sequence);
            if (dt2_handle_obj) Py_DECREF(dt2_handle_obj);
            return NULL;
        }
    }
    (void)dt2_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* dt3_handle_obj = NULL;
    PyObject* dt3_sequence = NULL;
    Py_ssize_t dt3_handle_len = 0;
    if (PyObject_HasAttrString(py_dt3, "_handle")) {
        dt3_handle_obj = PyObject_GetAttrString(py_dt3, "_handle");
        if (dt3_handle_obj == NULL) {
            return NULL;
        }
        dt3_sequence = PySequence_Fast(dt3_handle_obj, "Failed to access handle sequence");
        if (dt3_sequence == NULL) {
            Py_DECREF(dt3_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dt3)) {
        dt3_sequence = PySequence_Fast(py_dt3, "Argument dt3 must be a handle sequence");
        if (dt3_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dt3 must be a Fortran derived-type instance");
        return NULL;
    }
    dt3_handle_len = PySequence_Fast_GET_SIZE(dt3_sequence);
    if (dt3_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dt3 has an invalid handle length");
        Py_DECREF(dt3_sequence);
        if (dt3_handle_obj) Py_DECREF(dt3_handle_obj);
        return NULL;
    }
    int* dt3 = (int*)malloc(sizeof(int) * dt3_handle_len);
    if (dt3 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dt3_sequence);
        if (dt3_handle_obj) Py_DECREF(dt3_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dt3_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dt3_sequence, i);
        if (item == NULL) {
            free(dt3);
            Py_DECREF(dt3_sequence);
            if (dt3_handle_obj) Py_DECREF(dt3_handle_obj);
            return NULL;
        }
        dt3[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dt3);
            Py_DECREF(dt3_sequence);
            if (dt3_handle_obj) Py_DECREF(dt3_handle_obj);
            return NULL;
        }
    }
    (void)dt3_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__modify_derived_types)(dt1, dt2, dt3);
    if (PyErr_Occurred()) {
        if (dt1_sequence) Py_DECREF(dt1_sequence);
        if (dt1_handle_obj) Py_DECREF(dt1_handle_obj);
        free(dt1);
        if (dt2_sequence) Py_DECREF(dt2_sequence);
        if (dt2_handle_obj) Py_DECREF(dt2_handle_obj);
        free(dt2);
        if (dt3_sequence) Py_DECREF(dt3_sequence);
        if (dt3_handle_obj) Py_DECREF(dt3_handle_obj);
        free(dt3);
        return NULL;
    }
    
    if (dt1_sequence) {
        Py_DECREF(dt1_sequence);
    }
    if (dt1_handle_obj) {
        Py_DECREF(dt1_handle_obj);
    }
    free(dt1);
    if (dt2_sequence) {
        Py_DECREF(dt2_sequence);
    }
    if (dt2_handle_obj) {
        Py_DECREF(dt2_handle_obj);
    }
    free(dt2);
    if (dt3_sequence) {
        Py_DECREF(dt3_sequence);
    }
    if (dt3_handle_obj) {
        Py_DECREF(dt3_handle_obj);
    }
    free(dt3);
    Py_RETURN_NONE;
}

static PyObject* wrap_library_modify_dertype_fixed_shape_arrays(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int dertype[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__modify_dertype_fixed_shape_arrays)(dertype);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    PyObject* py_dertype_obj = PyList_New(4);
    if (py_dertype_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)dertype[i]);
        if (item == NULL) {
            Py_DECREF(py_dertype_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_dertype_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_dertype_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_dertype_obj != NULL) return py_dertype_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_dertype_obj != NULL) Py_DECREF(py_dertype_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_dertype_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_dertype_obj);
    }
    return result_tuple;
}

static PyObject* wrap_library_return_dertype_pointer_arrays(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_m = NULL;
    int m_val = 0;
    PyArrayObject* m_scalar_arr = NULL;
    int m_scalar_copyback = 0;
    int m_scalar_is_array = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"m", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_m, &py_n)) {
        return NULL;
    }
    
    int* m = &m_val;
    if (PyArray_Check(py_m)) {
        m_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_m, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (m_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(m_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument m must have exactly one element");
            Py_DECREF(m_scalar_arr);
            return NULL;
        }
        m_scalar_is_array = 1;
        m = (int*)PyArray_DATA(m_scalar_arr);
        m_val = m[0];
        if (PyArray_DATA(m_scalar_arr) != PyArray_DATA((PyArrayObject*)py_m) || PyArray_TYPE(m_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_m)) {
            m_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_m)) {
        m_val = (int)PyLong_AsLong(py_m);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument m must be a scalar number or NumPy array");
        return NULL;
    }
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
    int dertype[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__return_dertype_pointer_arrays)(m, n, dertype);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (m_scalar_is_array) {
        if (m_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_m, m_scalar_arr) < 0) {
                Py_DECREF(m_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(m_scalar_arr);
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
    PyObject* py_dertype_obj = PyList_New(4);
    if (py_dertype_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)dertype[i]);
        if (item == NULL) {
            Py_DECREF(py_dertype_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_dertype_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_dertype_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_dertype_obj != NULL) return py_dertype_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_dertype_obj != NULL) Py_DECREF(py_dertype_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_dertype_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_dertype_obj);
    }
    return result_tuple;
}

static PyObject* wrap_library_modify_dertype_pointer_arrays(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_dertype = NULL;
    static char *kwlist[] = {"dertype", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_dertype)) {
        return NULL;
    }
    
    PyObject* dertype_handle_obj = NULL;
    PyObject* dertype_sequence = NULL;
    Py_ssize_t dertype_handle_len = 0;
    if (PyObject_HasAttrString(py_dertype, "_handle")) {
        dertype_handle_obj = PyObject_GetAttrString(py_dertype, "_handle");
        if (dertype_handle_obj == NULL) {
            return NULL;
        }
        dertype_sequence = PySequence_Fast(dertype_handle_obj, "Failed to access handle sequence");
        if (dertype_sequence == NULL) {
            Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dertype)) {
        dertype_sequence = PySequence_Fast(py_dertype, "Argument dertype must be a handle sequence");
        if (dertype_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dertype must be a Fortran derived-type instance");
        return NULL;
    }
    dertype_handle_len = PySequence_Fast_GET_SIZE(dertype_sequence);
    if (dertype_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dertype has an invalid handle length");
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    int* dertype = (int*)malloc(sizeof(int) * dertype_handle_len);
    if (dertype == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dertype_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dertype_sequence, i);
        if (item == NULL) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
        dertype[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    }
    (void)dertype_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__modify_dertype_pointer_arrays)(dertype);
    if (PyErr_Occurred()) {
        if (dertype_sequence) Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        free(dertype);
        return NULL;
    }
    
    if (dertype_sequence) {
        Py_DECREF(dertype_sequence);
    }
    if (dertype_handle_obj) {
        Py_DECREF(dertype_handle_obj);
    }
    free(dertype);
    Py_RETURN_NONE;
}

static PyObject* wrap_library_return_dertype_alloc_arrays(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_m = NULL;
    int m_val = 0;
    PyArrayObject* m_scalar_arr = NULL;
    int m_scalar_copyback = 0;
    int m_scalar_is_array = 0;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"m", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_m, &py_n)) {
        return NULL;
    }
    
    int* m = &m_val;
    if (PyArray_Check(py_m)) {
        m_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_m, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (m_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(m_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument m must have exactly one element");
            Py_DECREF(m_scalar_arr);
            return NULL;
        }
        m_scalar_is_array = 1;
        m = (int*)PyArray_DATA(m_scalar_arr);
        m_val = m[0];
        if (PyArray_DATA(m_scalar_arr) != PyArray_DATA((PyArrayObject*)py_m) || PyArray_TYPE(m_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_m)) {
            m_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_m)) {
        m_val = (int)PyLong_AsLong(py_m);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument m must be a scalar number or NumPy array");
        return NULL;
    }
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
    int dertype[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__return_dertype_alloc_arrays)(m, n, dertype);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (m_scalar_is_array) {
        if (m_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_m, m_scalar_arr) < 0) {
                Py_DECREF(m_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(m_scalar_arr);
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
    PyObject* py_dertype_obj = PyList_New(4);
    if (py_dertype_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)dertype[i]);
        if (item == NULL) {
            Py_DECREF(py_dertype_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_dertype_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_dertype_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_dertype_obj != NULL) return py_dertype_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_dertype_obj != NULL) Py_DECREF(py_dertype_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_dertype_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_dertype_obj);
    }
    return result_tuple;
}

static PyObject* wrap_library_modify_dertype_alloc_arrays(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_dertype = NULL;
    static char *kwlist[] = {"dertype", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_dertype)) {
        return NULL;
    }
    
    PyObject* dertype_handle_obj = NULL;
    PyObject* dertype_sequence = NULL;
    Py_ssize_t dertype_handle_len = 0;
    if (PyObject_HasAttrString(py_dertype, "_handle")) {
        dertype_handle_obj = PyObject_GetAttrString(py_dertype, "_handle");
        if (dertype_handle_obj == NULL) {
            return NULL;
        }
        dertype_sequence = PySequence_Fast(dertype_handle_obj, "Failed to access handle sequence");
        if (dertype_sequence == NULL) {
            Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_dertype)) {
        dertype_sequence = PySequence_Fast(py_dertype, "Argument dertype must be a handle sequence");
        if (dertype_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument dertype must be a Fortran derived-type instance");
        return NULL;
    }
    dertype_handle_len = PySequence_Fast_GET_SIZE(dertype_sequence);
    if (dertype_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument dertype has an invalid handle length");
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    int* dertype = (int*)malloc(sizeof(int) * dertype_handle_len);
    if (dertype == NULL) {
        PyErr_NoMemory();
        Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < dertype_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(dertype_sequence, i);
        if (item == NULL) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
        dertype[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(dertype);
            Py_DECREF(dertype_sequence);
            if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
            return NULL;
        }
    }
    (void)dertype_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_library__modify_dertype_alloc_arrays)(dertype);
    if (PyErr_Occurred()) {
        if (dertype_sequence) Py_DECREF(dertype_sequence);
        if (dertype_handle_obj) Py_DECREF(dertype_handle_obj);
        free(dertype);
        return NULL;
    }
    
    if (dertype_sequence) {
        Py_DECREF(dertype_sequence);
    }
    if (dertype_handle_obj) {
        Py_DECREF(dertype_handle_obj);
    }
    free(dertype);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes_allocatable__alloc_arrays_helper_array_chi(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays__array__chi)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes_allocatable__alloc_arrays_helper_array_psi(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays__array__psi)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes_allocatable__alloc_arrays_helper_array_chi_shape(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays__array__chi_shape)(dummy_this, &nd, &dtype, dshape, \
        &handle);
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

static PyObject* wrap_datatypes_allocatable__alloc_arrays_helper_array_psi_shape(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes_allocatable__alloc_arrays__array__psi_shape)(dummy_this, &nd, &dtype, dshape, \
        &handle);
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

static PyObject* wrap_datatypes__different_types_helper_get_alpha(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__get__alpha)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_datatypes__different_types_helper_set_alpha(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "alpha", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__set__alpha)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__different_types_helper_get_beta(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__get__beta)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_datatypes__different_types_helper_set_beta(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "beta", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__set__beta)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__different_types_helper_get_delta(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__get__delta)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_datatypes__different_types_helper_set_delta(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "delta", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__different_types__set__delta)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__fixed_shape_arrays_helper_array_eta(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays__array__eta)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__fixed_shape_arrays_helper_array_theta(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays__array__theta)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__fixed_shape_arrays_helper_array_iota(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__fixed_shape_arrays__array__iota)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__nested_helper_get_derived_mu(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__nested__get__mu)(handle_handle, value_handle);
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

static PyObject* wrap_datatypes__nested_helper_set_derived_mu(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__nested__set__mu)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__nested_helper_get_derived_nu(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__nested__get__nu)(handle_handle, value_handle);
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

static PyObject* wrap_datatypes__nested_helper_set_derived_nu(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__nested__set__nu)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__pointer_arrays_helper_array_chi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays__array__chi)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__pointer_arrays_helper_array_psi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays__array__psi)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__pointer_arrays_helper_array_chi_shape(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays__array__chi_shape)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__pointer_arrays_helper_array_psi_shape(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__pointer_arrays__array__psi_shape)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__alloc_arrays_2_helper_array_chi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2__array__chi)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__alloc_arrays_2_helper_array_psi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2__array__psi)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__alloc_arrays_2_helper_array_chi_shape(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2__array__chi_shape)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__alloc_arrays_2_helper_array_psi_shape(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__alloc_arrays_2__array__psi_shape)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_datatypes__array_nested_helper_array_getitem_xi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_getitem__xi)(parent_handle, &index, handle);
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

static PyObject* wrap_datatypes__array_nested_helper_array_setitem_xi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_setitem__xi)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__array_nested_helper_array_len_xi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_len__xi)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_datatypes__array_nested_helper_array_getitem_omicron(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_getitem__omicron)(parent_handle, &index, handle);
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

static PyObject* wrap_datatypes__array_nested_helper_array_setitem_omicron(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_setitem__omicron)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__array_nested_helper_array_len_omicron(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_len__omicron)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_datatypes__array_nested_helper_array_getitem_pi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_getitem__pi)(parent_handle, &index, handle);
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

static PyObject* wrap_datatypes__array_nested_helper_array_setitem_pi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_setitem__pi)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_datatypes__array_nested_helper_array_len_pi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_datatypes__array_nested__array_len__pi)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
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

/* Method table for _ExampleDerivedTypes_pkg module */
static PyMethodDef _ExampleDerivedTypes_pkg_methods[] = {
    {"f90wrap_datatypes_allocatable__init_alloc_arrays", (PyCFunction)wrap_datatypes_allocatable_init_alloc_arrays, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for init_alloc_arrays"},
    {"f90wrap_datatypes_allocatable__destroy_alloc_arrays", (PyCFunction)wrap_datatypes_allocatable_destroy_alloc_arrays, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for destroy_alloc_arrays"},
    {"f90wrap_datatypes_allocatable__alloc_arrays_initialise", \
        (PyCFunction)wrap_datatypes_allocatable_alloc_arrays_initialise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated constructor for alloc_arrays"},
    {"f90wrap_datatypes_allocatable__alloc_arrays_finalise", (PyCFunction)wrap_datatypes_allocatable_alloc_arrays_finalise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated destructor for alloc_arrays"},
    {"f90wrap_datatypes__init_array_nested", (PyCFunction)wrap_datatypes_init_array_nested, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for init_array_nested"},
    {"f90wrap_datatypes__destroy_array_nested", (PyCFunction)wrap_datatypes_destroy_array_nested, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for destroy_array_nested"},
    {"f90wrap_datatypes__different_types_initialise", (PyCFunction)wrap_datatypes_different_types_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for different_types"},
    {"f90wrap_datatypes__different_types_finalise", (PyCFunction)wrap_datatypes_different_types_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for different_types"},
    {"f90wrap_datatypes__fixed_shape_arrays_initialise", (PyCFunction)wrap_datatypes_fixed_shape_arrays_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for fixed_shape_arrays"},
    {"f90wrap_datatypes__fixed_shape_arrays_finalise", (PyCFunction)wrap_datatypes_fixed_shape_arrays_finalise, METH_VARARGS \
        | METH_KEYWORDS, "Automatically generated destructor for fixed_shape_arrays"},
    {"f90wrap_datatypes__nested_initialise", (PyCFunction)wrap_datatypes_nested_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for nested"},
    {"f90wrap_datatypes__nested_finalise", (PyCFunction)wrap_datatypes_nested_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for nested"},
    {"f90wrap_datatypes__pointer_arrays_initialise", (PyCFunction)wrap_datatypes_pointer_arrays_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for pointer_arrays"},
    {"f90wrap_datatypes__pointer_arrays_finalise", (PyCFunction)wrap_datatypes_pointer_arrays_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for pointer_arrays"},
    {"f90wrap_datatypes__alloc_arrays_2_initialise", (PyCFunction)wrap_datatypes_alloc_arrays_2_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for alloc_arrays_2"},
    {"f90wrap_datatypes__alloc_arrays_2_finalise", (PyCFunction)wrap_datatypes_alloc_arrays_2_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for alloc_arrays_2"},
    {"f90wrap_datatypes__array_nested_initialise", (PyCFunction)wrap_datatypes_array_nested_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for array_nested"},
    {"f90wrap_datatypes__array_nested_finalise", (PyCFunction)wrap_datatypes_array_nested_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for array_nested"},
    {"f90wrap_library__return_value_func", (PyCFunction)wrap_library_return_value_func, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for return_value_func"},
    {"f90wrap_library__return_value_sub", (PyCFunction)wrap_library_return_value_sub, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for return_value_sub"},
    {"f90wrap_library__return_a_dt_func", (PyCFunction)wrap_library_return_a_dt_func, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for return_a_dt_func"},
    {"f90wrap_library__do_array_stuff", (PyCFunction)wrap_library_do_array_stuff, METH_VARARGS | METH_KEYWORDS, "Wrapper for \
        do_array_stuff"},
    {"f90wrap_library__only_manipulate", (PyCFunction)wrap_library_only_manipulate, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for only_manipulate"},
    {"f90wrap_library__set_derived_type", (PyCFunction)wrap_library_set_derived_type, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for set_derived_type"},
    {"f90wrap_library__modify_derived_types", (PyCFunction)wrap_library_modify_derived_types, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for modify_derived_types"},
    {"f90wrap_library__modify_dertype_fixed_shape_arrays", (PyCFunction)wrap_library_modify_dertype_fixed_shape_arrays, \
        METH_VARARGS | METH_KEYWORDS, "Wrapper for modify_dertype_fixed_shape_arrays"},
    {"f90wrap_library__return_dertype_pointer_arrays", (PyCFunction)wrap_library_return_dertype_pointer_arrays, METH_VARARGS \
        | METH_KEYWORDS, "Wrapper for return_dertype_pointer_arrays"},
    {"f90wrap_library__modify_dertype_pointer_arrays", (PyCFunction)wrap_library_modify_dertype_pointer_arrays, METH_VARARGS \
        | METH_KEYWORDS, "Wrapper for modify_dertype_pointer_arrays"},
    {"f90wrap_library__return_dertype_alloc_arrays", (PyCFunction)wrap_library_return_dertype_alloc_arrays, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for return_dertype_alloc_arrays"},
    {"f90wrap_library__modify_dertype_alloc_arrays", (PyCFunction)wrap_library_modify_dertype_alloc_arrays, METH_VARARGS | \
        METH_KEYWORDS, "Wrapper for modify_dertype_alloc_arrays"},
    {"f90wrap_datatypes_allocatable__alloc_arrays__array__chi", \
        (PyCFunction)wrap_datatypes_allocatable__alloc_arrays_helper_array_chi, METH_VARARGS | METH_KEYWORDS, "Array helper \
        for chi"},
    {"f90wrap_datatypes_allocatable__alloc_arrays__array__psi", \
        (PyCFunction)wrap_datatypes_allocatable__alloc_arrays_helper_array_psi, METH_VARARGS | METH_KEYWORDS, "Array helper \
        for psi"},
    {"f90wrap_datatypes_allocatable__alloc_arrays__array__chi_shape", \
        (PyCFunction)wrap_datatypes_allocatable__alloc_arrays_helper_array_chi_shape, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for chi_shape"},
    {"f90wrap_datatypes_allocatable__alloc_arrays__array__psi_shape", \
        (PyCFunction)wrap_datatypes_allocatable__alloc_arrays_helper_array_psi_shape, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for psi_shape"},
    {"f90wrap_datatypes__different_types__get__alpha", (PyCFunction)wrap_datatypes__different_types_helper_get_alpha, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for alpha"},
    {"f90wrap_datatypes__different_types__set__alpha", (PyCFunction)wrap_datatypes__different_types_helper_set_alpha, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for alpha"},
    {"f90wrap_datatypes__different_types__get__beta", (PyCFunction)wrap_datatypes__different_types_helper_get_beta, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for beta"},
    {"f90wrap_datatypes__different_types__set__beta", (PyCFunction)wrap_datatypes__different_types_helper_set_beta, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for beta"},
    {"f90wrap_datatypes__different_types__get__delta", (PyCFunction)wrap_datatypes__different_types_helper_get_delta, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for delta"},
    {"f90wrap_datatypes__different_types__set__delta", (PyCFunction)wrap_datatypes__different_types_helper_set_delta, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for delta"},
    {"f90wrap_datatypes__fixed_shape_arrays__array__eta", (PyCFunction)wrap_datatypes__fixed_shape_arrays_helper_array_eta, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for eta"},
    {"f90wrap_datatypes__fixed_shape_arrays__array__theta", \
        (PyCFunction)wrap_datatypes__fixed_shape_arrays_helper_array_theta, METH_VARARGS | METH_KEYWORDS, "Array helper for \
        theta"},
    {"f90wrap_datatypes__fixed_shape_arrays__array__iota", \
        (PyCFunction)wrap_datatypes__fixed_shape_arrays_helper_array_iota, METH_VARARGS | METH_KEYWORDS, "Array helper for \
        iota"},
    {"f90wrap_datatypes__nested__get__mu", (PyCFunction)wrap_datatypes__nested_helper_get_derived_mu, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for mu"},
    {"f90wrap_datatypes__nested__set__mu", (PyCFunction)wrap_datatypes__nested_helper_set_derived_mu, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for mu"},
    {"f90wrap_datatypes__nested__get__nu", (PyCFunction)wrap_datatypes__nested_helper_get_derived_nu, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for nu"},
    {"f90wrap_datatypes__nested__set__nu", (PyCFunction)wrap_datatypes__nested_helper_set_derived_nu, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for nu"},
    {"f90wrap_datatypes__pointer_arrays__array__chi", (PyCFunction)wrap_datatypes__pointer_arrays_helper_array_chi, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for chi"},
    {"f90wrap_datatypes__pointer_arrays__array__psi", (PyCFunction)wrap_datatypes__pointer_arrays_helper_array_psi, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for psi"},
    {"f90wrap_datatypes__pointer_arrays__array__chi_shape", \
        (PyCFunction)wrap_datatypes__pointer_arrays_helper_array_chi_shape, METH_VARARGS | METH_KEYWORDS, "Array helper for \
        chi_shape"},
    {"f90wrap_datatypes__pointer_arrays__array__psi_shape", \
        (PyCFunction)wrap_datatypes__pointer_arrays_helper_array_psi_shape, METH_VARARGS | METH_KEYWORDS, "Array helper for \
        psi_shape"},
    {"f90wrap_datatypes__alloc_arrays_2__array__chi", (PyCFunction)wrap_datatypes__alloc_arrays_2_helper_array_chi, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for chi"},
    {"f90wrap_datatypes__alloc_arrays_2__array__psi", (PyCFunction)wrap_datatypes__alloc_arrays_2_helper_array_psi, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for psi"},
    {"f90wrap_datatypes__alloc_arrays_2__array__chi_shape", \
        (PyCFunction)wrap_datatypes__alloc_arrays_2_helper_array_chi_shape, METH_VARARGS | METH_KEYWORDS, "Array helper for \
        chi_shape"},
    {"f90wrap_datatypes__alloc_arrays_2__array__psi_shape", \
        (PyCFunction)wrap_datatypes__alloc_arrays_2_helper_array_psi_shape, METH_VARARGS | METH_KEYWORDS, "Array helper for \
        psi_shape"},
    {"f90wrap_datatypes__array_nested__array_getitem__xi", \
        (PyCFunction)wrap_datatypes__array_nested_helper_array_getitem_xi, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        xi"},
    {"f90wrap_datatypes__array_nested__array_setitem__xi", \
        (PyCFunction)wrap_datatypes__array_nested_helper_array_setitem_xi, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        xi"},
    {"f90wrap_datatypes__array_nested__array_len__xi", (PyCFunction)wrap_datatypes__array_nested_helper_array_len_xi, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for xi"},
    {"f90wrap_datatypes__array_nested__array_getitem__omicron", \
        (PyCFunction)wrap_datatypes__array_nested_helper_array_getitem_omicron, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for omicron"},
    {"f90wrap_datatypes__array_nested__array_setitem__omicron", \
        (PyCFunction)wrap_datatypes__array_nested_helper_array_setitem_omicron, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for omicron"},
    {"f90wrap_datatypes__array_nested__array_len__omicron", \
        (PyCFunction)wrap_datatypes__array_nested_helper_array_len_omicron, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        omicron"},
    {"f90wrap_datatypes__array_nested__array_getitem__pi", \
        (PyCFunction)wrap_datatypes__array_nested_helper_array_getitem_pi, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        pi"},
    {"f90wrap_datatypes__array_nested__array_setitem__pi", \
        (PyCFunction)wrap_datatypes__array_nested_helper_array_setitem_pi, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        pi"},
    {"f90wrap_datatypes__array_nested__array_len__pi", (PyCFunction)wrap_datatypes__array_nested_helper_array_len_pi, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for pi"},
    {"f90wrap_parameters__get__idp", (PyCFunction)wrap_parameters_helper_get_idp, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for idp"},
    {"f90wrap_parameters__get__isp", (PyCFunction)wrap_parameters_helper_get_isp, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for isp"},
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
