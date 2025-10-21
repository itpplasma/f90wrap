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
extern void F90WRAP_F_SYMBOL(f90wrap_mcyldnad__cyldnad)(int* vol, int* radius, int* height);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__abs_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__acos_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__asin_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__cos_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__exp_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__int_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__log_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__log10_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__nint_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__sin_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__sqrt_d)(int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__get__ndv_ad)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num__get__x_ad_)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num__set__x_ad_)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num__array__xp_ad_)(int* dummy_this, int* nd, int* dtype, \
    int* dshape, long long* handle);

static PyObject* wrap_mcyldnad_cyldnad(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_radius = NULL;
    PyObject* py_height = NULL;
    static char *kwlist[] = {"radius", "height", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_radius, &py_height)) {
        return NULL;
    }
    
    int vol[4] = {0};
    PyObject* radius_handle_obj = NULL;
    PyObject* radius_sequence = NULL;
    Py_ssize_t radius_handle_len = 0;
    if (PyObject_HasAttrString(py_radius, "_handle")) {
        radius_handle_obj = PyObject_GetAttrString(py_radius, "_handle");
        if (radius_handle_obj == NULL) {
            return NULL;
        }
        radius_sequence = PySequence_Fast(radius_handle_obj, "Failed to access handle sequence");
        if (radius_sequence == NULL) {
            Py_DECREF(radius_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_radius)) {
        radius_sequence = PySequence_Fast(py_radius, "Argument radius must be a handle sequence");
        if (radius_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a Fortran derived-type instance");
        return NULL;
    }
    radius_handle_len = PySequence_Fast_GET_SIZE(radius_sequence);
    if (radius_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument radius has an invalid handle length");
        Py_DECREF(radius_sequence);
        if (radius_handle_obj) Py_DECREF(radius_handle_obj);
        return NULL;
    }
    int* radius = (int*)malloc(sizeof(int) * radius_handle_len);
    if (radius == NULL) {
        PyErr_NoMemory();
        Py_DECREF(radius_sequence);
        if (radius_handle_obj) Py_DECREF(radius_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < radius_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(radius_sequence, i);
        if (item == NULL) {
            free(radius);
            Py_DECREF(radius_sequence);
            if (radius_handle_obj) Py_DECREF(radius_handle_obj);
            return NULL;
        }
        radius[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(radius);
            Py_DECREF(radius_sequence);
            if (radius_handle_obj) Py_DECREF(radius_handle_obj);
            return NULL;
        }
    }
    (void)radius_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* height_handle_obj = NULL;
    PyObject* height_sequence = NULL;
    Py_ssize_t height_handle_len = 0;
    if (PyObject_HasAttrString(py_height, "_handle")) {
        height_handle_obj = PyObject_GetAttrString(py_height, "_handle");
        if (height_handle_obj == NULL) {
            return NULL;
        }
        height_sequence = PySequence_Fast(height_handle_obj, "Failed to access handle sequence");
        if (height_sequence == NULL) {
            Py_DECREF(height_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_height)) {
        height_sequence = PySequence_Fast(py_height, "Argument height must be a handle sequence");
        if (height_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument height must be a Fortran derived-type instance");
        return NULL;
    }
    height_handle_len = PySequence_Fast_GET_SIZE(height_sequence);
    if (height_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument height has an invalid handle length");
        Py_DECREF(height_sequence);
        if (height_handle_obj) Py_DECREF(height_handle_obj);
        return NULL;
    }
    int* height = (int*)malloc(sizeof(int) * height_handle_len);
    if (height == NULL) {
        PyErr_NoMemory();
        Py_DECREF(height_sequence);
        if (height_handle_obj) Py_DECREF(height_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < height_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(height_sequence, i);
        if (item == NULL) {
            free(height);
            Py_DECREF(height_sequence);
            if (height_handle_obj) Py_DECREF(height_handle_obj);
            return NULL;
        }
        height[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(height);
            Py_DECREF(height_sequence);
            if (height_handle_obj) Py_DECREF(height_handle_obj);
            return NULL;
        }
    }
    (void)height_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_mcyldnad__cyldnad)(vol, radius, height);
    if (PyErr_Occurred()) {
        if (radius_sequence) Py_DECREF(radius_sequence);
        if (radius_handle_obj) Py_DECREF(radius_handle_obj);
        free(radius);
        if (height_sequence) Py_DECREF(height_sequence);
        if (height_handle_obj) Py_DECREF(height_handle_obj);
        free(height);
        return NULL;
    }
    
    PyObject* py_vol_obj = PyList_New(4);
    if (py_vol_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)vol[i]);
        if (item == NULL) {
            Py_DECREF(py_vol_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_vol_obj, i, item);
    }
    if (radius_sequence) {
        Py_DECREF(radius_sequence);
    }
    if (radius_handle_obj) {
        Py_DECREF(radius_handle_obj);
    }
    free(radius);
    if (height_sequence) {
        Py_DECREF(height_sequence);
    }
    if (height_handle_obj) {
        Py_DECREF(height_handle_obj);
    }
    free(height);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_vol_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_vol_obj != NULL) return py_vol_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_vol_obj != NULL) Py_DECREF(py_vol_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_vol_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_vol_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_abs_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__abs_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_acos_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__acos_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_asin_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__asin_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_cos_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__cos_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_exp_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__exp_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_int_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    int ret_res_val = 0;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__int_d)(u, &ret_res_val);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = Py_BuildValue("i", ret_res_val);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_log_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__log_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_log10_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__log10_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_nint_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    int ret_res_val = 0;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__nint_d)(u, &ret_res_val);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = Py_BuildValue("i", ret_res_val);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_sin_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__sin_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_sqrt_d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    static char *kwlist[] = {"u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_u)) {
        return NULL;
    }
    
    PyObject* u_handle_obj = NULL;
    PyObject* u_sequence = NULL;
    Py_ssize_t u_handle_len = 0;
    if (PyObject_HasAttrString(py_u, "_handle")) {
        u_handle_obj = PyObject_GetAttrString(py_u, "_handle");
        if (u_handle_obj == NULL) {
            return NULL;
        }
        u_sequence = PySequence_Fast(u_handle_obj, "Failed to access handle sequence");
        if (u_sequence == NULL) {
            Py_DECREF(u_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_u)) {
        u_sequence = PySequence_Fast(py_u, "Argument u must be a handle sequence");
        if (u_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument u must be a Fortran derived-type instance");
        return NULL;
    }
    u_handle_len = PySequence_Fast_GET_SIZE(u_sequence);
    if (u_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument u has an invalid handle length");
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    int* u = (int*)malloc(sizeof(int) * u_handle_len);
    if (u == NULL) {
        PyErr_NoMemory();
        Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < u_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(u_sequence, i);
        if (item == NULL) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
        u[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(u);
            Py_DECREF(u_sequence);
            if (u_handle_obj) Py_DECREF(u_handle_obj);
            return NULL;
        }
    }
    (void)u_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__sqrt_d)(u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    PyObject* py_ret_res_obj = PyList_New(4);
    if (py_ret_res_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_res[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_res_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_res_obj, i, item);
    }
    if (u_sequence) {
        Py_DECREF(u_sequence);
    }
    if (u_handle_obj) {
        Py_DECREF(u_handle_obj);
    }
    free(u);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_res_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_res_obj != NULL) return py_ret_res_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_res_obj != NULL) Py_DECREF(py_ret_res_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_res_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_res_obj);
    }
    return result_tuple;
}

static PyObject* wrap_dual_num_auto_diff_dual_num_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num_initialise)(this);
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

static PyObject* wrap_dual_num_auto_diff_dual_num_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_dual_num_auto_diff_helper_get_ndv_ad(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__get__ndv_ad)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_dual_num_auto_diff__dual_num_helper_get_x_ad_(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num__get__x_ad_)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_dual_num_auto_diff__dual_num_helper_set_x_ad_(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "x_ad_", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num__set__x_ad_)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_dual_num_auto_diff__dual_num_helper_array_xp_ad_(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__dual_num__array__xp_ad_)(dummy_this, &nd, &dtype, dshape, &handle);
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

/* Method table for _Example module */
static PyMethodDef _Example_methods[] = {
    {"f90wrap_mcyldnad__cyldnad", (PyCFunction)wrap_mcyldnad_cyldnad, METH_VARARGS | METH_KEYWORDS, "Wrapper for cyldnad"},
    {"f90wrap_dual_num_auto_diff__abs_d", (PyCFunction)wrap_dual_num_auto_diff_abs_d, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for abs_d"},
    {"f90wrap_dual_num_auto_diff__acos_d", (PyCFunction)wrap_dual_num_auto_diff_acos_d, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for acos_d"},
    {"f90wrap_dual_num_auto_diff__asin_d", (PyCFunction)wrap_dual_num_auto_diff_asin_d, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for asin_d"},
    {"f90wrap_dual_num_auto_diff__cos_d", (PyCFunction)wrap_dual_num_auto_diff_cos_d, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for cos_d"},
    {"f90wrap_dual_num_auto_diff__exp_d", (PyCFunction)wrap_dual_num_auto_diff_exp_d, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for exp_d"},
    {"f90wrap_dual_num_auto_diff__int_d", (PyCFunction)wrap_dual_num_auto_diff_int_d, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for int_d"},
    {"f90wrap_dual_num_auto_diff__log_d", (PyCFunction)wrap_dual_num_auto_diff_log_d, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for log_d"},
    {"f90wrap_dual_num_auto_diff__log10_d", (PyCFunction)wrap_dual_num_auto_diff_log10_d, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for log10_d"},
    {"f90wrap_dual_num_auto_diff__nint_d", (PyCFunction)wrap_dual_num_auto_diff_nint_d, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for nint_d"},
    {"f90wrap_dual_num_auto_diff__sin_d", (PyCFunction)wrap_dual_num_auto_diff_sin_d, METH_VARARGS | METH_KEYWORDS, "Wrapper \
        for sin_d"},
    {"f90wrap_dual_num_auto_diff__sqrt_d", (PyCFunction)wrap_dual_num_auto_diff_sqrt_d, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for sqrt_d"},
    {"f90wrap_dual_num_auto_diff__dual_num_initialise", (PyCFunction)wrap_dual_num_auto_diff_dual_num_initialise, \
        METH_VARARGS | METH_KEYWORDS, "Automatically generated constructor for dual_num"},
    {"f90wrap_dual_num_auto_diff__dual_num_finalise", (PyCFunction)wrap_dual_num_auto_diff_dual_num_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for dual_num"},
    {"f90wrap_dual_num_auto_diff__get__ndv_ad", (PyCFunction)wrap_dual_num_auto_diff_helper_get_ndv_ad, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for ndv_ad"},
    {"f90wrap_dual_num_auto_diff__dual_num__get__x_ad_", (PyCFunction)wrap_dual_num_auto_diff__dual_num_helper_get_x_ad_, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for x_ad_"},
    {"f90wrap_dual_num_auto_diff__dual_num__set__x_ad_", (PyCFunction)wrap_dual_num_auto_diff__dual_num_helper_set_x_ad_, \
        METH_VARARGS | METH_KEYWORDS, "Module helper for x_ad_"},
    {"f90wrap_dual_num_auto_diff__dual_num__array__xp_ad_", \
        (PyCFunction)wrap_dual_num_auto_diff__dual_num_helper_array_xp_ad_, METH_VARARGS | METH_KEYWORDS, "Array helper for \
        xp_ad_"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _Examplemodule = {
    PyModuleDef_HEAD_INIT,
    "Example",
    "Direct-C wrapper for _Example module",
    -1,
    _Example_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__Example(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_Examplemodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
