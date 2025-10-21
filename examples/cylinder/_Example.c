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
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_dd)(int* val1, int* val2, int* ret_res, int* val3, int* \
    val4, int* val5);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_di)(int* u, int* n, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_dr)(int* u, double* n, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_ds)(int* u, float* n, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_rd)(double* r, int* u, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__min_dd)(int* val1, int* val2, int* ret_res, int* val3, int* \
    val4);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__min_dr)(int* u, double* n, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__min_ds)(int* u, float* n, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__sign_dd)(int* val1, int* val2, int* ret_res);
extern void F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__sign_rd)(double* val1, int* val2, int* ret_res);
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

static PyObject* wrap_dual_num_auto_diff_max_dd(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_val1 = NULL;
    PyObject* py_val2 = NULL;
    PyObject* py_val3 = NULL;
    PyObject* py_val4 = NULL;
    PyObject* py_val5 = NULL;
    static char *kwlist[] = {"val1", "val2", "val3", "val4", "val5", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOO", kwlist, &py_val1, &py_val2, &py_val3, &py_val4, &py_val5)) {
        return NULL;
    }
    
    PyObject* val1_handle_obj = NULL;
    PyObject* val1_sequence = NULL;
    Py_ssize_t val1_handle_len = 0;
    if (PyObject_HasAttrString(py_val1, "_handle")) {
        val1_handle_obj = PyObject_GetAttrString(py_val1, "_handle");
        if (val1_handle_obj == NULL) {
            return NULL;
        }
        val1_sequence = PySequence_Fast(val1_handle_obj, "Failed to access handle sequence");
        if (val1_sequence == NULL) {
            Py_DECREF(val1_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_val1)) {
        val1_sequence = PySequence_Fast(py_val1, "Argument val1 must be a handle sequence");
        if (val1_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val1 must be a Fortran derived-type instance");
        return NULL;
    }
    val1_handle_len = PySequence_Fast_GET_SIZE(val1_sequence);
    if (val1_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument val1 has an invalid handle length");
        Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        return NULL;
    }
    int* val1 = (int*)malloc(sizeof(int) * val1_handle_len);
    if (val1 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < val1_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(val1_sequence, i);
        if (item == NULL) {
            free(val1);
            Py_DECREF(val1_sequence);
            if (val1_handle_obj) Py_DECREF(val1_handle_obj);
            return NULL;
        }
        val1[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(val1);
            Py_DECREF(val1_sequence);
            if (val1_handle_obj) Py_DECREF(val1_handle_obj);
            return NULL;
        }
    }
    (void)val1_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* val2_handle_obj = NULL;
    PyObject* val2_sequence = NULL;
    Py_ssize_t val2_handle_len = 0;
    if (PyObject_HasAttrString(py_val2, "_handle")) {
        val2_handle_obj = PyObject_GetAttrString(py_val2, "_handle");
        if (val2_handle_obj == NULL) {
            return NULL;
        }
        val2_sequence = PySequence_Fast(val2_handle_obj, "Failed to access handle sequence");
        if (val2_sequence == NULL) {
            Py_DECREF(val2_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_val2)) {
        val2_sequence = PySequence_Fast(py_val2, "Argument val2 must be a handle sequence");
        if (val2_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val2 must be a Fortran derived-type instance");
        return NULL;
    }
    val2_handle_len = PySequence_Fast_GET_SIZE(val2_sequence);
    if (val2_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument val2 has an invalid handle length");
        Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        return NULL;
    }
    int* val2 = (int*)malloc(sizeof(int) * val2_handle_len);
    if (val2 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < val2_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(val2_sequence, i);
        if (item == NULL) {
            free(val2);
            Py_DECREF(val2_sequence);
            if (val2_handle_obj) Py_DECREF(val2_handle_obj);
            return NULL;
        }
        val2[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(val2);
            Py_DECREF(val2_sequence);
            if (val2_handle_obj) Py_DECREF(val2_handle_obj);
            return NULL;
        }
    }
    (void)val2_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    PyObject* val3_handle_obj = NULL;
    PyObject* val3_sequence = NULL;
    Py_ssize_t val3_handle_len = 0;
    int* val3 = NULL;
    if (py_val3 != Py_None) {
        if (PyObject_HasAttrString(py_val3, "_handle")) {
            val3_handle_obj = PyObject_GetAttrString(py_val3, "_handle");
            if (val3_handle_obj == NULL) {
                return NULL;
            }
            val3_sequence = PySequence_Fast(val3_handle_obj, "Failed to access handle sequence");
            if (val3_sequence == NULL) {
                Py_DECREF(val3_handle_obj);
                return NULL;
            }
        } else if (PySequence_Check(py_val3)) {
            val3_sequence = PySequence_Fast(py_val3, "Argument val3 must be a handle sequence");
            if (val3_sequence == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument val3 must be a Fortran derived-type instance");
            return NULL;
        }
        val3_handle_len = PySequence_Fast_GET_SIZE(val3_sequence);
        if (val3_handle_len != 4) {
            PyErr_SetString(PyExc_ValueError, "Argument val3 has an invalid handle length");
            Py_DECREF(val3_sequence);
            if (val3_handle_obj) Py_DECREF(val3_handle_obj);
            return NULL;
        }
        val3 = (int*)malloc(sizeof(int) * val3_handle_len);
        if (val3 == NULL) {
            PyErr_NoMemory();
            Py_DECREF(val3_sequence);
            if (val3_handle_obj) Py_DECREF(val3_handle_obj);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < val3_handle_len; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(val3_sequence, i);
            if (item == NULL) {
                free(val3);
                Py_DECREF(val3_sequence);
                if (val3_handle_obj) Py_DECREF(val3_handle_obj);
                return NULL;
            }
            val3[i] = (int)PyLong_AsLong(item);
            if (PyErr_Occurred()) {
                free(val3);
                Py_DECREF(val3_sequence);
                if (val3_handle_obj) Py_DECREF(val3_handle_obj);
                return NULL;
            }
        }
        (void)val3_handle_len;  /* suppress unused warnings when unchanged */
    }
    PyObject* val4_handle_obj = NULL;
    PyObject* val4_sequence = NULL;
    Py_ssize_t val4_handle_len = 0;
    int* val4 = NULL;
    if (py_val4 != Py_None) {
        if (PyObject_HasAttrString(py_val4, "_handle")) {
            val4_handle_obj = PyObject_GetAttrString(py_val4, "_handle");
            if (val4_handle_obj == NULL) {
                return NULL;
            }
            val4_sequence = PySequence_Fast(val4_handle_obj, "Failed to access handle sequence");
            if (val4_sequence == NULL) {
                Py_DECREF(val4_handle_obj);
                return NULL;
            }
        } else if (PySequence_Check(py_val4)) {
            val4_sequence = PySequence_Fast(py_val4, "Argument val4 must be a handle sequence");
            if (val4_sequence == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument val4 must be a Fortran derived-type instance");
            return NULL;
        }
        val4_handle_len = PySequence_Fast_GET_SIZE(val4_sequence);
        if (val4_handle_len != 4) {
            PyErr_SetString(PyExc_ValueError, "Argument val4 has an invalid handle length");
            Py_DECREF(val4_sequence);
            if (val4_handle_obj) Py_DECREF(val4_handle_obj);
            return NULL;
        }
        val4 = (int*)malloc(sizeof(int) * val4_handle_len);
        if (val4 == NULL) {
            PyErr_NoMemory();
            Py_DECREF(val4_sequence);
            if (val4_handle_obj) Py_DECREF(val4_handle_obj);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < val4_handle_len; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(val4_sequence, i);
            if (item == NULL) {
                free(val4);
                Py_DECREF(val4_sequence);
                if (val4_handle_obj) Py_DECREF(val4_handle_obj);
                return NULL;
            }
            val4[i] = (int)PyLong_AsLong(item);
            if (PyErr_Occurred()) {
                free(val4);
                Py_DECREF(val4_sequence);
                if (val4_handle_obj) Py_DECREF(val4_handle_obj);
                return NULL;
            }
        }
        (void)val4_handle_len;  /* suppress unused warnings when unchanged */
    }
    PyObject* val5_handle_obj = NULL;
    PyObject* val5_sequence = NULL;
    Py_ssize_t val5_handle_len = 0;
    int* val5 = NULL;
    if (py_val5 != Py_None) {
        if (PyObject_HasAttrString(py_val5, "_handle")) {
            val5_handle_obj = PyObject_GetAttrString(py_val5, "_handle");
            if (val5_handle_obj == NULL) {
                return NULL;
            }
            val5_sequence = PySequence_Fast(val5_handle_obj, "Failed to access handle sequence");
            if (val5_sequence == NULL) {
                Py_DECREF(val5_handle_obj);
                return NULL;
            }
        } else if (PySequence_Check(py_val5)) {
            val5_sequence = PySequence_Fast(py_val5, "Argument val5 must be a handle sequence");
            if (val5_sequence == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument val5 must be a Fortran derived-type instance");
            return NULL;
        }
        val5_handle_len = PySequence_Fast_GET_SIZE(val5_sequence);
        if (val5_handle_len != 4) {
            PyErr_SetString(PyExc_ValueError, "Argument val5 has an invalid handle length");
            Py_DECREF(val5_sequence);
            if (val5_handle_obj) Py_DECREF(val5_handle_obj);
            return NULL;
        }
        val5 = (int*)malloc(sizeof(int) * val5_handle_len);
        if (val5 == NULL) {
            PyErr_NoMemory();
            Py_DECREF(val5_sequence);
            if (val5_handle_obj) Py_DECREF(val5_handle_obj);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < val5_handle_len; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(val5_sequence, i);
            if (item == NULL) {
                free(val5);
                Py_DECREF(val5_sequence);
                if (val5_handle_obj) Py_DECREF(val5_handle_obj);
                return NULL;
            }
            val5[i] = (int)PyLong_AsLong(item);
            if (PyErr_Occurred()) {
                free(val5);
                Py_DECREF(val5_sequence);
                if (val5_handle_obj) Py_DECREF(val5_handle_obj);
                return NULL;
            }
        }
        (void)val5_handle_len;  /* suppress unused warnings when unchanged */
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_dd)(val1, val2, ret_res, val3, val4, val5);
    if (PyErr_Occurred()) {
        if (val1_sequence) Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        free(val1);
        if (val2_sequence) Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        free(val2);
        if (val3_sequence) Py_DECREF(val3_sequence);
        if (val3_handle_obj) Py_DECREF(val3_handle_obj);
        free(val3);
        if (val4_sequence) Py_DECREF(val4_sequence);
        if (val4_handle_obj) Py_DECREF(val4_handle_obj);
        free(val4);
        if (val5_sequence) Py_DECREF(val5_sequence);
        if (val5_handle_obj) Py_DECREF(val5_handle_obj);
        free(val5);
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
    if (val1_sequence) {
        Py_DECREF(val1_sequence);
    }
    if (val1_handle_obj) {
        Py_DECREF(val1_handle_obj);
    }
    free(val1);
    if (val2_sequence) {
        Py_DECREF(val2_sequence);
    }
    if (val2_handle_obj) {
        Py_DECREF(val2_handle_obj);
    }
    free(val2);
    if (val3_sequence) {
        Py_DECREF(val3_sequence);
    }
    if (val3_handle_obj) {
        Py_DECREF(val3_handle_obj);
    }
    free(val3);
    if (val4_sequence) {
        Py_DECREF(val4_sequence);
    }
    if (val4_handle_obj) {
        Py_DECREF(val4_handle_obj);
    }
    free(val4);
    if (val5_sequence) {
        Py_DECREF(val5_sequence);
    }
    if (val5_handle_obj) {
        Py_DECREF(val5_handle_obj);
    }
    free(val5);
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

static PyObject* wrap_dual_num_auto_diff_max_di(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"u", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_u, &py_n)) {
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
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_di)(u, n, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
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

static PyObject* wrap_dual_num_auto_diff_max_dr(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    PyObject* py_n = NULL;
    double n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"u", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_u, &py_n)) {
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
    
    double* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (double*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (double)PyFloat_AsDouble(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_dr)(u, n, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
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

static PyObject* wrap_dual_num_auto_diff_max_ds(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    PyObject* py_n = NULL;
    float n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"u", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_u, &py_n)) {
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
    
    float* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (float*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (float)PyFloat_AsDouble(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_ds)(u, n, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
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

static PyObject* wrap_dual_num_auto_diff_max_rd(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_r = NULL;
    double r_val = 0;
    PyArrayObject* r_scalar_arr = NULL;
    int r_scalar_copyback = 0;
    int r_scalar_is_array = 0;
    PyObject* py_u = NULL;
    static char *kwlist[] = {"r", "u", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_r, &py_u)) {
        return NULL;
    }
    
    double* r = &r_val;
    if (PyArray_Check(py_r)) {
        r_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_r, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (r_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(r_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument r must have exactly one element");
            Py_DECREF(r_scalar_arr);
            return NULL;
        }
        r_scalar_is_array = 1;
        r = (double*)PyArray_DATA(r_scalar_arr);
        r_val = r[0];
        if (PyArray_DATA(r_scalar_arr) != PyArray_DATA((PyArrayObject*)py_r) || PyArray_TYPE(r_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_r)) {
            r_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_r)) {
        r_val = (double)PyFloat_AsDouble(py_r);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument r must be a scalar number or NumPy array");
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
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__max_rd)(r, u, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
        return NULL;
    }
    
    if (r_scalar_is_array) {
        if (r_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_r, r_scalar_arr) < 0) {
                Py_DECREF(r_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(r_scalar_arr);
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

static PyObject* wrap_dual_num_auto_diff_min_dd(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_val1 = NULL;
    PyObject* py_val2 = NULL;
    PyObject* py_val3 = NULL;
    PyObject* py_val4 = NULL;
    static char *kwlist[] = {"val1", "val2", "val3", "val4", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &py_val1, &py_val2, &py_val3, &py_val4)) {
        return NULL;
    }
    
    PyObject* val1_handle_obj = NULL;
    PyObject* val1_sequence = NULL;
    Py_ssize_t val1_handle_len = 0;
    if (PyObject_HasAttrString(py_val1, "_handle")) {
        val1_handle_obj = PyObject_GetAttrString(py_val1, "_handle");
        if (val1_handle_obj == NULL) {
            return NULL;
        }
        val1_sequence = PySequence_Fast(val1_handle_obj, "Failed to access handle sequence");
        if (val1_sequence == NULL) {
            Py_DECREF(val1_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_val1)) {
        val1_sequence = PySequence_Fast(py_val1, "Argument val1 must be a handle sequence");
        if (val1_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val1 must be a Fortran derived-type instance");
        return NULL;
    }
    val1_handle_len = PySequence_Fast_GET_SIZE(val1_sequence);
    if (val1_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument val1 has an invalid handle length");
        Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        return NULL;
    }
    int* val1 = (int*)malloc(sizeof(int) * val1_handle_len);
    if (val1 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < val1_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(val1_sequence, i);
        if (item == NULL) {
            free(val1);
            Py_DECREF(val1_sequence);
            if (val1_handle_obj) Py_DECREF(val1_handle_obj);
            return NULL;
        }
        val1[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(val1);
            Py_DECREF(val1_sequence);
            if (val1_handle_obj) Py_DECREF(val1_handle_obj);
            return NULL;
        }
    }
    (void)val1_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* val2_handle_obj = NULL;
    PyObject* val2_sequence = NULL;
    Py_ssize_t val2_handle_len = 0;
    if (PyObject_HasAttrString(py_val2, "_handle")) {
        val2_handle_obj = PyObject_GetAttrString(py_val2, "_handle");
        if (val2_handle_obj == NULL) {
            return NULL;
        }
        val2_sequence = PySequence_Fast(val2_handle_obj, "Failed to access handle sequence");
        if (val2_sequence == NULL) {
            Py_DECREF(val2_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_val2)) {
        val2_sequence = PySequence_Fast(py_val2, "Argument val2 must be a handle sequence");
        if (val2_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val2 must be a Fortran derived-type instance");
        return NULL;
    }
    val2_handle_len = PySequence_Fast_GET_SIZE(val2_sequence);
    if (val2_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument val2 has an invalid handle length");
        Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        return NULL;
    }
    int* val2 = (int*)malloc(sizeof(int) * val2_handle_len);
    if (val2 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < val2_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(val2_sequence, i);
        if (item == NULL) {
            free(val2);
            Py_DECREF(val2_sequence);
            if (val2_handle_obj) Py_DECREF(val2_handle_obj);
            return NULL;
        }
        val2[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(val2);
            Py_DECREF(val2_sequence);
            if (val2_handle_obj) Py_DECREF(val2_handle_obj);
            return NULL;
        }
    }
    (void)val2_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    PyObject* val3_handle_obj = NULL;
    PyObject* val3_sequence = NULL;
    Py_ssize_t val3_handle_len = 0;
    int* val3 = NULL;
    if (py_val3 != Py_None) {
        if (PyObject_HasAttrString(py_val3, "_handle")) {
            val3_handle_obj = PyObject_GetAttrString(py_val3, "_handle");
            if (val3_handle_obj == NULL) {
                return NULL;
            }
            val3_sequence = PySequence_Fast(val3_handle_obj, "Failed to access handle sequence");
            if (val3_sequence == NULL) {
                Py_DECREF(val3_handle_obj);
                return NULL;
            }
        } else if (PySequence_Check(py_val3)) {
            val3_sequence = PySequence_Fast(py_val3, "Argument val3 must be a handle sequence");
            if (val3_sequence == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument val3 must be a Fortran derived-type instance");
            return NULL;
        }
        val3_handle_len = PySequence_Fast_GET_SIZE(val3_sequence);
        if (val3_handle_len != 4) {
            PyErr_SetString(PyExc_ValueError, "Argument val3 has an invalid handle length");
            Py_DECREF(val3_sequence);
            if (val3_handle_obj) Py_DECREF(val3_handle_obj);
            return NULL;
        }
        val3 = (int*)malloc(sizeof(int) * val3_handle_len);
        if (val3 == NULL) {
            PyErr_NoMemory();
            Py_DECREF(val3_sequence);
            if (val3_handle_obj) Py_DECREF(val3_handle_obj);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < val3_handle_len; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(val3_sequence, i);
            if (item == NULL) {
                free(val3);
                Py_DECREF(val3_sequence);
                if (val3_handle_obj) Py_DECREF(val3_handle_obj);
                return NULL;
            }
            val3[i] = (int)PyLong_AsLong(item);
            if (PyErr_Occurred()) {
                free(val3);
                Py_DECREF(val3_sequence);
                if (val3_handle_obj) Py_DECREF(val3_handle_obj);
                return NULL;
            }
        }
        (void)val3_handle_len;  /* suppress unused warnings when unchanged */
    }
    PyObject* val4_handle_obj = NULL;
    PyObject* val4_sequence = NULL;
    Py_ssize_t val4_handle_len = 0;
    int* val4 = NULL;
    if (py_val4 != Py_None) {
        if (PyObject_HasAttrString(py_val4, "_handle")) {
            val4_handle_obj = PyObject_GetAttrString(py_val4, "_handle");
            if (val4_handle_obj == NULL) {
                return NULL;
            }
            val4_sequence = PySequence_Fast(val4_handle_obj, "Failed to access handle sequence");
            if (val4_sequence == NULL) {
                Py_DECREF(val4_handle_obj);
                return NULL;
            }
        } else if (PySequence_Check(py_val4)) {
            val4_sequence = PySequence_Fast(py_val4, "Argument val4 must be a handle sequence");
            if (val4_sequence == NULL) {
                return NULL;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, "Argument val4 must be a Fortran derived-type instance");
            return NULL;
        }
        val4_handle_len = PySequence_Fast_GET_SIZE(val4_sequence);
        if (val4_handle_len != 4) {
            PyErr_SetString(PyExc_ValueError, "Argument val4 has an invalid handle length");
            Py_DECREF(val4_sequence);
            if (val4_handle_obj) Py_DECREF(val4_handle_obj);
            return NULL;
        }
        val4 = (int*)malloc(sizeof(int) * val4_handle_len);
        if (val4 == NULL) {
            PyErr_NoMemory();
            Py_DECREF(val4_sequence);
            if (val4_handle_obj) Py_DECREF(val4_handle_obj);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < val4_handle_len; ++i) {
            PyObject* item = PySequence_Fast_GET_ITEM(val4_sequence, i);
            if (item == NULL) {
                free(val4);
                Py_DECREF(val4_sequence);
                if (val4_handle_obj) Py_DECREF(val4_handle_obj);
                return NULL;
            }
            val4[i] = (int)PyLong_AsLong(item);
            if (PyErr_Occurred()) {
                free(val4);
                Py_DECREF(val4_sequence);
                if (val4_handle_obj) Py_DECREF(val4_handle_obj);
                return NULL;
            }
        }
        (void)val4_handle_len;  /* suppress unused warnings when unchanged */
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__min_dd)(val1, val2, ret_res, val3, val4);
    if (PyErr_Occurred()) {
        if (val1_sequence) Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        free(val1);
        if (val2_sequence) Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        free(val2);
        if (val3_sequence) Py_DECREF(val3_sequence);
        if (val3_handle_obj) Py_DECREF(val3_handle_obj);
        free(val3);
        if (val4_sequence) Py_DECREF(val4_sequence);
        if (val4_handle_obj) Py_DECREF(val4_handle_obj);
        free(val4);
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
    if (val1_sequence) {
        Py_DECREF(val1_sequence);
    }
    if (val1_handle_obj) {
        Py_DECREF(val1_handle_obj);
    }
    free(val1);
    if (val2_sequence) {
        Py_DECREF(val2_sequence);
    }
    if (val2_handle_obj) {
        Py_DECREF(val2_handle_obj);
    }
    free(val2);
    if (val3_sequence) {
        Py_DECREF(val3_sequence);
    }
    if (val3_handle_obj) {
        Py_DECREF(val3_handle_obj);
    }
    free(val3);
    if (val4_sequence) {
        Py_DECREF(val4_sequence);
    }
    if (val4_handle_obj) {
        Py_DECREF(val4_handle_obj);
    }
    free(val4);
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

static PyObject* wrap_dual_num_auto_diff_min_dr(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    PyObject* py_n = NULL;
    double n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"u", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_u, &py_n)) {
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
    
    double* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (double*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (double)PyFloat_AsDouble(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__min_dr)(u, n, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
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

static PyObject* wrap_dual_num_auto_diff_min_ds(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_u = NULL;
    PyObject* py_n = NULL;
    float n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"u", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_u, &py_n)) {
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
    
    float* n = &n_val;
    if (PyArray_Check(py_n)) {
        n_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n must have exactly one element");
            Py_DECREF(n_scalar_arr);
            return NULL;
        }
        n_scalar_is_array = 1;
        n = (float*)PyArray_DATA(n_scalar_arr);
        n_val = n[0];
        if (PyArray_DATA(n_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n) || PyArray_TYPE(n_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n)) {
            n_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n)) {
        n_val = (float)PyFloat_AsDouble(py_n);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n must be a scalar number or NumPy array");
        return NULL;
    }
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__min_ds)(u, n, ret_res);
    if (PyErr_Occurred()) {
        if (u_sequence) Py_DECREF(u_sequence);
        if (u_handle_obj) Py_DECREF(u_handle_obj);
        free(u);
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

static PyObject* wrap_dual_num_auto_diff_sign_dd(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_val1 = NULL;
    PyObject* py_val2 = NULL;
    static char *kwlist[] = {"val1", "val2", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_val1, &py_val2)) {
        return NULL;
    }
    
    PyObject* val1_handle_obj = NULL;
    PyObject* val1_sequence = NULL;
    Py_ssize_t val1_handle_len = 0;
    if (PyObject_HasAttrString(py_val1, "_handle")) {
        val1_handle_obj = PyObject_GetAttrString(py_val1, "_handle");
        if (val1_handle_obj == NULL) {
            return NULL;
        }
        val1_sequence = PySequence_Fast(val1_handle_obj, "Failed to access handle sequence");
        if (val1_sequence == NULL) {
            Py_DECREF(val1_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_val1)) {
        val1_sequence = PySequence_Fast(py_val1, "Argument val1 must be a handle sequence");
        if (val1_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val1 must be a Fortran derived-type instance");
        return NULL;
    }
    val1_handle_len = PySequence_Fast_GET_SIZE(val1_sequence);
    if (val1_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument val1 has an invalid handle length");
        Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        return NULL;
    }
    int* val1 = (int*)malloc(sizeof(int) * val1_handle_len);
    if (val1 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < val1_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(val1_sequence, i);
        if (item == NULL) {
            free(val1);
            Py_DECREF(val1_sequence);
            if (val1_handle_obj) Py_DECREF(val1_handle_obj);
            return NULL;
        }
        val1[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(val1);
            Py_DECREF(val1_sequence);
            if (val1_handle_obj) Py_DECREF(val1_handle_obj);
            return NULL;
        }
    }
    (void)val1_handle_len;  /* suppress unused warnings when unchanged */
    
    PyObject* val2_handle_obj = NULL;
    PyObject* val2_sequence = NULL;
    Py_ssize_t val2_handle_len = 0;
    if (PyObject_HasAttrString(py_val2, "_handle")) {
        val2_handle_obj = PyObject_GetAttrString(py_val2, "_handle");
        if (val2_handle_obj == NULL) {
            return NULL;
        }
        val2_sequence = PySequence_Fast(val2_handle_obj, "Failed to access handle sequence");
        if (val2_sequence == NULL) {
            Py_DECREF(val2_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_val2)) {
        val2_sequence = PySequence_Fast(py_val2, "Argument val2 must be a handle sequence");
        if (val2_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val2 must be a Fortran derived-type instance");
        return NULL;
    }
    val2_handle_len = PySequence_Fast_GET_SIZE(val2_sequence);
    if (val2_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument val2 has an invalid handle length");
        Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        return NULL;
    }
    int* val2 = (int*)malloc(sizeof(int) * val2_handle_len);
    if (val2 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < val2_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(val2_sequence, i);
        if (item == NULL) {
            free(val2);
            Py_DECREF(val2_sequence);
            if (val2_handle_obj) Py_DECREF(val2_handle_obj);
            return NULL;
        }
        val2[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(val2);
            Py_DECREF(val2_sequence);
            if (val2_handle_obj) Py_DECREF(val2_handle_obj);
            return NULL;
        }
    }
    (void)val2_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__sign_dd)(val1, val2, ret_res);
    if (PyErr_Occurred()) {
        if (val1_sequence) Py_DECREF(val1_sequence);
        if (val1_handle_obj) Py_DECREF(val1_handle_obj);
        free(val1);
        if (val2_sequence) Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        free(val2);
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
    if (val1_sequence) {
        Py_DECREF(val1_sequence);
    }
    if (val1_handle_obj) {
        Py_DECREF(val1_handle_obj);
    }
    free(val1);
    if (val2_sequence) {
        Py_DECREF(val2_sequence);
    }
    if (val2_handle_obj) {
        Py_DECREF(val2_handle_obj);
    }
    free(val2);
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

static PyObject* wrap_dual_num_auto_diff_sign_rd(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_val1 = NULL;
    double val1_val = 0;
    PyArrayObject* val1_scalar_arr = NULL;
    int val1_scalar_copyback = 0;
    int val1_scalar_is_array = 0;
    PyObject* py_val2 = NULL;
    static char *kwlist[] = {"val1", "val2", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_val1, &py_val2)) {
        return NULL;
    }
    
    double* val1 = &val1_val;
    if (PyArray_Check(py_val1)) {
        val1_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_val1, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (val1_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(val1_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument val1 must have exactly one element");
            Py_DECREF(val1_scalar_arr);
            return NULL;
        }
        val1_scalar_is_array = 1;
        val1 = (double*)PyArray_DATA(val1_scalar_arr);
        val1_val = val1[0];
        if (PyArray_DATA(val1_scalar_arr) != PyArray_DATA((PyArrayObject*)py_val1) || PyArray_TYPE(val1_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_val1)) {
            val1_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_val1)) {
        val1_val = (double)PyFloat_AsDouble(py_val1);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val1 must be a scalar number or NumPy array");
        return NULL;
    }
    PyObject* val2_handle_obj = NULL;
    PyObject* val2_sequence = NULL;
    Py_ssize_t val2_handle_len = 0;
    if (PyObject_HasAttrString(py_val2, "_handle")) {
        val2_handle_obj = PyObject_GetAttrString(py_val2, "_handle");
        if (val2_handle_obj == NULL) {
            return NULL;
        }
        val2_sequence = PySequence_Fast(val2_handle_obj, "Failed to access handle sequence");
        if (val2_sequence == NULL) {
            Py_DECREF(val2_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_val2)) {
        val2_sequence = PySequence_Fast(py_val2, "Argument val2 must be a handle sequence");
        if (val2_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument val2 must be a Fortran derived-type instance");
        return NULL;
    }
    val2_handle_len = PySequence_Fast_GET_SIZE(val2_sequence);
    if (val2_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument val2 has an invalid handle length");
        Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        return NULL;
    }
    int* val2 = (int*)malloc(sizeof(int) * val2_handle_len);
    if (val2 == NULL) {
        PyErr_NoMemory();
        Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < val2_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(val2_sequence, i);
        if (item == NULL) {
            free(val2);
            Py_DECREF(val2_sequence);
            if (val2_handle_obj) Py_DECREF(val2_handle_obj);
            return NULL;
        }
        val2[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(val2);
            Py_DECREF(val2_sequence);
            if (val2_handle_obj) Py_DECREF(val2_handle_obj);
            return NULL;
        }
    }
    (void)val2_handle_len;  /* suppress unused warnings when unchanged */
    
    int ret_res[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_dual_num_auto_diff__sign_rd)(val1, val2, ret_res);
    if (PyErr_Occurred()) {
        if (val2_sequence) Py_DECREF(val2_sequence);
        if (val2_handle_obj) Py_DECREF(val2_handle_obj);
        free(val2);
        return NULL;
    }
    
    if (val1_scalar_is_array) {
        if (val1_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_val1, val1_scalar_arr) < 0) {
                Py_DECREF(val1_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(val1_scalar_arr);
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
    if (val2_sequence) {
        Py_DECREF(val2_sequence);
    }
    if (val2_handle_obj) {
        Py_DECREF(val2_handle_obj);
    }
    free(val2);
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
    {"f90wrap_dual_num_auto_diff__max_dd", (PyCFunction)wrap_dual_num_auto_diff_max_dd, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for max_dd"},
    {"f90wrap_dual_num_auto_diff__max_di", (PyCFunction)wrap_dual_num_auto_diff_max_di, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for max_di"},
    {"f90wrap_dual_num_auto_diff__max_dr", (PyCFunction)wrap_dual_num_auto_diff_max_dr, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for max_dr"},
    {"f90wrap_dual_num_auto_diff__max_ds", (PyCFunction)wrap_dual_num_auto_diff_max_ds, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for max_ds"},
    {"f90wrap_dual_num_auto_diff__max_rd", (PyCFunction)wrap_dual_num_auto_diff_max_rd, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for max_rd"},
    {"f90wrap_dual_num_auto_diff__min_dd", (PyCFunction)wrap_dual_num_auto_diff_min_dd, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for min_dd"},
    {"f90wrap_dual_num_auto_diff__min_dr", (PyCFunction)wrap_dual_num_auto_diff_min_dr, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for min_dr"},
    {"f90wrap_dual_num_auto_diff__min_ds", (PyCFunction)wrap_dual_num_auto_diff_min_ds, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for min_ds"},
    {"f90wrap_dual_num_auto_diff__sign_dd", (PyCFunction)wrap_dual_num_auto_diff_sign_dd, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for sign_dd"},
    {"f90wrap_dual_num_auto_diff__sign_rd", (PyCFunction)wrap_dual_num_auto_diff_sign_rd, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for sign_rd"},
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
