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
extern void F90WRAP_F_SYMBOL(f90wrap_m_base_poly__polygone_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_base_poly__polygone_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__get_circle_radius)(int* my_circle, double* ret_radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__get_ball_radius)(int* my_ball, double* ret_radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__construct_square)(int* ret_construct_square, float* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__square_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__construct_circle)(int* ret_construct_circle, float* rc, float* rb);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__construct_ball)(int* ret_construct_ball, float* rc, float* rb);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__ball_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_3d_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_3d_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__get__pi)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle__get__length)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle__set__length)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle__get__width)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle__set__width)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond__get__length)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond__set__length)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond__get__width)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond__set__width)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_getitem__alloc_type)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_setitem__alloc_type)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_len__alloc_type)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_getitem__ptr_type)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_setitem__ptr_type)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_len__ptr_type)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_getitem__alloc_class)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_setitem__alloc_class)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_len__alloc_class)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_getitem__ptr_class)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_setitem__ptr_class)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_len__ptr_class)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__get__scalar_class)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__set__scalar_class)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__get__scalar_type)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__set__scalar_type)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__get__n)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__set__n)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__circle__get__radius)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__circle__set__radius)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_getitem__alloc_type)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_setitem__alloc_type)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_len__alloc_type)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_getitem__ptr_type)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_setitem__ptr_type)(int* dummy_this, int* index, int* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_len__ptr_type)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_getitem__alloc_class)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_setitem__alloc_class)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_len__alloc_class)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_getitem__ptr_class)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_setitem__ptr_class)(int* dummy_this, int* index, \
    int* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_len__ptr_class)(int* dummy_this, int* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__get__scalar_class)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__set__scalar_class)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__get__scalar_type)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__set__scalar_type)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__get__n)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__set__n)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__array__array__buf)(int* dummy_this, int* nd, int* dtype, int* dshape, \
    long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__array__array__values)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_3d__array__values_3d)(int* dummy_this, int* nd, int* dtype, int* \
    dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_m_base_poly__is_polygone__binding__polygone)(int* this, int* ret_is_polygone);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__perimeter__binding__rectangle)(int* this, double* ret_perimeter);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_square__binding__rectangle)(int* this, int* ret_is_square);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__area__binding__rectangle)(int* this, double* ret_area);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_polygone__binding__rectangle)(int* this, int* ret_is_polygone);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__square)(int* this, float* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_square__binding__square)(int* this, int* ret_is_square);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__area__binding__square)(int* this, double* ret_area);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_equal__binding__square)(int* this, int* other, int* ret_is_equal);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__copy__binding__square)(int* this, int* from_);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__create_diamond__binding__square)(int* this, int* \
    ret_square_create_diamond);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__perimeter__binding__square)(int* this, double* ret_perimeter);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_polygone__binding__square)(int* this, int* ret_is_polygone);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__diamond)(int* this, double* width, double* length);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__info__binding__diamond)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__copy__binding__diamond)(int* this, int* other);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_polygone__binding__diamond)(int* this, int* ret_is_polygone);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__list_square)(int* this, int* n);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__area__binding__circle)(int* this, double* ret_area);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__print__binding__circle)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__obj_name__binding__circle)(int* obj);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__copy__binding__circle)(int* this, int* from_);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__circle)(int* this, float* radius);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__private_method__binding__circle)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__perimeter_4__binding__circle)(int* this, float* radius, float* \
    ret_perimeter);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__perimeter_8__binding__circle)(int* this, double* radius, double* \
    ret_perimeter);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__circle_free__binding__circle)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__volume__binding__ball)(int* this, double* ret_volume);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__area__binding__ball)(int* this, double* ret_area);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__private_method__binding__ball)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__list_circle)(int* this, int* n);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__array)(int* this, int* n);
extern void F90WRAP_F_SYMBOL(f90wrap_m_geometry__init_3d__binding__array_3d)(int* this, int* n1, int* n2, int* n3);

static PyObject* wrap_m_base_poly_polygone_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_base_poly__polygone_initialise)(this);
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

static PyObject* wrap_m_base_poly_polygone_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_base_poly__polygone_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_get_circle_radius(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_my_circle = NULL;
    double ret_radius_val = 0;
    static char *kwlist[] = {"my_circle", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_my_circle)) {
        return NULL;
    }
    
    PyObject* my_circle_handle_obj = NULL;
    PyObject* my_circle_sequence = NULL;
    Py_ssize_t my_circle_handle_len = 0;
    if (PyObject_HasAttrString(py_my_circle, "_handle")) {
        my_circle_handle_obj = PyObject_GetAttrString(py_my_circle, "_handle");
        if (my_circle_handle_obj == NULL) {
            return NULL;
        }
        my_circle_sequence = PySequence_Fast(my_circle_handle_obj, "Failed to access handle sequence");
        if (my_circle_sequence == NULL) {
            Py_DECREF(my_circle_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_my_circle)) {
        my_circle_sequence = PySequence_Fast(py_my_circle, "Argument my_circle must be a handle sequence");
        if (my_circle_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument my_circle must be a Fortran derived-type instance");
        return NULL;
    }
    my_circle_handle_len = PySequence_Fast_GET_SIZE(my_circle_sequence);
    if (my_circle_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument my_circle has an invalid handle length");
        Py_DECREF(my_circle_sequence);
        if (my_circle_handle_obj) Py_DECREF(my_circle_handle_obj);
        return NULL;
    }
    int* my_circle = (int*)malloc(sizeof(int) * my_circle_handle_len);
    if (my_circle == NULL) {
        PyErr_NoMemory();
        Py_DECREF(my_circle_sequence);
        if (my_circle_handle_obj) Py_DECREF(my_circle_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < my_circle_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(my_circle_sequence, i);
        if (item == NULL) {
            free(my_circle);
            Py_DECREF(my_circle_sequence);
            if (my_circle_handle_obj) Py_DECREF(my_circle_handle_obj);
            return NULL;
        }
        my_circle[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(my_circle);
            Py_DECREF(my_circle_sequence);
            if (my_circle_handle_obj) Py_DECREF(my_circle_handle_obj);
            return NULL;
        }
    }
    (void)my_circle_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__get_circle_radius)(my_circle, &ret_radius_val);
    if (PyErr_Occurred()) {
        if (my_circle_sequence) Py_DECREF(my_circle_sequence);
        if (my_circle_handle_obj) Py_DECREF(my_circle_handle_obj);
        free(my_circle);
        return NULL;
    }
    
    PyObject* py_ret_radius_obj = Py_BuildValue("d", ret_radius_val);
    if (py_ret_radius_obj == NULL) {
        return NULL;
    }
    if (my_circle_sequence) {
        Py_DECREF(my_circle_sequence);
    }
    if (my_circle_handle_obj) {
        Py_DECREF(my_circle_handle_obj);
    }
    free(my_circle);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_radius_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_radius_obj != NULL) return py_ret_radius_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_radius_obj != NULL) Py_DECREF(py_ret_radius_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_radius_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_radius_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_geometry_get_ball_radius(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_my_ball = NULL;
    double ret_radius_val = 0;
    static char *kwlist[] = {"my_ball", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_my_ball)) {
        return NULL;
    }
    
    PyObject* my_ball_handle_obj = NULL;
    PyObject* my_ball_sequence = NULL;
    Py_ssize_t my_ball_handle_len = 0;
    if (PyObject_HasAttrString(py_my_ball, "_handle")) {
        my_ball_handle_obj = PyObject_GetAttrString(py_my_ball, "_handle");
        if (my_ball_handle_obj == NULL) {
            return NULL;
        }
        my_ball_sequence = PySequence_Fast(my_ball_handle_obj, "Failed to access handle sequence");
        if (my_ball_sequence == NULL) {
            Py_DECREF(my_ball_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_my_ball)) {
        my_ball_sequence = PySequence_Fast(py_my_ball, "Argument my_ball must be a handle sequence");
        if (my_ball_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument my_ball must be a Fortran derived-type instance");
        return NULL;
    }
    my_ball_handle_len = PySequence_Fast_GET_SIZE(my_ball_sequence);
    if (my_ball_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument my_ball has an invalid handle length");
        Py_DECREF(my_ball_sequence);
        if (my_ball_handle_obj) Py_DECREF(my_ball_handle_obj);
        return NULL;
    }
    int* my_ball = (int*)malloc(sizeof(int) * my_ball_handle_len);
    if (my_ball == NULL) {
        PyErr_NoMemory();
        Py_DECREF(my_ball_sequence);
        if (my_ball_handle_obj) Py_DECREF(my_ball_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < my_ball_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(my_ball_sequence, i);
        if (item == NULL) {
            free(my_ball);
            Py_DECREF(my_ball_sequence);
            if (my_ball_handle_obj) Py_DECREF(my_ball_handle_obj);
            return NULL;
        }
        my_ball[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(my_ball);
            Py_DECREF(my_ball_sequence);
            if (my_ball_handle_obj) Py_DECREF(my_ball_handle_obj);
            return NULL;
        }
    }
    (void)my_ball_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__get_ball_radius)(my_ball, &ret_radius_val);
    if (PyErr_Occurred()) {
        if (my_ball_sequence) Py_DECREF(my_ball_sequence);
        if (my_ball_handle_obj) Py_DECREF(my_ball_handle_obj);
        free(my_ball);
        return NULL;
    }
    
    PyObject* py_ret_radius_obj = Py_BuildValue("d", ret_radius_val);
    if (py_ret_radius_obj == NULL) {
        return NULL;
    }
    if (my_ball_sequence) {
        Py_DECREF(my_ball_sequence);
    }
    if (my_ball_handle_obj) {
        Py_DECREF(my_ball_handle_obj);
    }
    free(my_ball);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_radius_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_radius_obj != NULL) return py_ret_radius_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_radius_obj != NULL) Py_DECREF(py_ret_radius_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_radius_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_radius_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_geometry_rectangle_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle_initialise)(this);
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

static PyObject* wrap_m_geometry_rectangle_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_construct_square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_length = NULL;
    float length_val = 0;
    PyArrayObject* length_scalar_arr = NULL;
    int length_scalar_copyback = 0;
    int length_scalar_is_array = 0;
    static char *kwlist[] = {"length", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_length)) {
        return NULL;
    }
    
    int ret_construct_square[4] = {0};
    float* length = &length_val;
    if (PyArray_Check(py_length)) {
        length_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_length, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (length_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(length_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument length must have exactly one element");
            Py_DECREF(length_scalar_arr);
            return NULL;
        }
        length_scalar_is_array = 1;
        length = (float*)PyArray_DATA(length_scalar_arr);
        length_val = length[0];
        if (PyArray_DATA(length_scalar_arr) != PyArray_DATA((PyArrayObject*)py_length) || PyArray_TYPE(length_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_length)) {
            length_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_length)) {
        length_val = (float)PyFloat_AsDouble(py_length);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument length must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__construct_square)(ret_construct_square, length);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (length_scalar_is_array) {
        if (length_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_length, length_scalar_arr) < 0) {
                Py_DECREF(length_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(length_scalar_arr);
    }
    PyObject* py_ret_construct_square_obj = PyList_New(4);
    if (py_ret_construct_square_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_construct_square[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_construct_square_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_construct_square_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_construct_square_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_construct_square_obj != NULL) return py_ret_construct_square_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_construct_square_obj != NULL) Py_DECREF(py_ret_construct_square_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_construct_square_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_construct_square_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_geometry_square_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__square_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_diamond_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond_initialise)(this);
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

static PyObject* wrap_m_geometry_diamond_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_list_square_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square_initialise)(this);
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

static PyObject* wrap_m_geometry_list_square_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_construct_circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_rc = NULL;
    float rc_val = 0;
    PyArrayObject* rc_scalar_arr = NULL;
    int rc_scalar_copyback = 0;
    int rc_scalar_is_array = 0;
    PyObject* py_rb = NULL;
    float rb_val = 0;
    PyArrayObject* rb_scalar_arr = NULL;
    int rb_scalar_copyback = 0;
    int rb_scalar_is_array = 0;
    static char *kwlist[] = {"rc", "rb", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_rc, &py_rb)) {
        return NULL;
    }
    
    int ret_construct_circle[4] = {0};
    float* rc = &rc_val;
    if (PyArray_Check(py_rc)) {
        rc_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_rc, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (rc_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(rc_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument rc must have exactly one element");
            Py_DECREF(rc_scalar_arr);
            return NULL;
        }
        rc_scalar_is_array = 1;
        rc = (float*)PyArray_DATA(rc_scalar_arr);
        rc_val = rc[0];
        if (PyArray_DATA(rc_scalar_arr) != PyArray_DATA((PyArrayObject*)py_rc) || PyArray_TYPE(rc_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_rc)) {
            rc_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_rc)) {
        rc_val = (float)PyFloat_AsDouble(py_rc);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument rc must be a scalar number or NumPy array");
        return NULL;
    }
    float* rb = &rb_val;
    if (PyArray_Check(py_rb)) {
        rb_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_rb, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (rb_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(rb_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument rb must have exactly one element");
            Py_DECREF(rb_scalar_arr);
            return NULL;
        }
        rb_scalar_is_array = 1;
        rb = (float*)PyArray_DATA(rb_scalar_arr);
        rb_val = rb[0];
        if (PyArray_DATA(rb_scalar_arr) != PyArray_DATA((PyArrayObject*)py_rb) || PyArray_TYPE(rb_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_rb)) {
            rb_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_rb)) {
        rb_val = (float)PyFloat_AsDouble(py_rb);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument rb must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__construct_circle)(ret_construct_circle, rc, rb);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (rc_scalar_is_array) {
        if (rc_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_rc, rc_scalar_arr) < 0) {
                Py_DECREF(rc_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(rc_scalar_arr);
    }
    if (rb_scalar_is_array) {
        if (rb_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_rb, rb_scalar_arr) < 0) {
                Py_DECREF(rb_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(rb_scalar_arr);
    }
    PyObject* py_ret_construct_circle_obj = PyList_New(4);
    if (py_ret_construct_circle_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_construct_circle[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_construct_circle_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_construct_circle_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_construct_circle_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_construct_circle_obj != NULL) return py_ret_construct_circle_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_construct_circle_obj != NULL) Py_DECREF(py_ret_construct_circle_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_construct_circle_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_construct_circle_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_geometry_construct_ball(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_rc = NULL;
    float rc_val = 0;
    PyArrayObject* rc_scalar_arr = NULL;
    int rc_scalar_copyback = 0;
    int rc_scalar_is_array = 0;
    PyObject* py_rb = NULL;
    float rb_val = 0;
    PyArrayObject* rb_scalar_arr = NULL;
    int rb_scalar_copyback = 0;
    int rb_scalar_is_array = 0;
    static char *kwlist[] = {"rc", "rb", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_rc, &py_rb)) {
        return NULL;
    }
    
    int ret_construct_ball[4] = {0};
    float* rc = &rc_val;
    if (PyArray_Check(py_rc)) {
        rc_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_rc, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (rc_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(rc_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument rc must have exactly one element");
            Py_DECREF(rc_scalar_arr);
            return NULL;
        }
        rc_scalar_is_array = 1;
        rc = (float*)PyArray_DATA(rc_scalar_arr);
        rc_val = rc[0];
        if (PyArray_DATA(rc_scalar_arr) != PyArray_DATA((PyArrayObject*)py_rc) || PyArray_TYPE(rc_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_rc)) {
            rc_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_rc)) {
        rc_val = (float)PyFloat_AsDouble(py_rc);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument rc must be a scalar number or NumPy array");
        return NULL;
    }
    float* rb = &rb_val;
    if (PyArray_Check(py_rb)) {
        rb_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_rb, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (rb_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(rb_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument rb must have exactly one element");
            Py_DECREF(rb_scalar_arr);
            return NULL;
        }
        rb_scalar_is_array = 1;
        rb = (float*)PyArray_DATA(rb_scalar_arr);
        rb_val = rb[0];
        if (PyArray_DATA(rb_scalar_arr) != PyArray_DATA((PyArrayObject*)py_rb) || PyArray_TYPE(rb_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_rb)) {
            rb_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_rb)) {
        rb_val = (float)PyFloat_AsDouble(py_rb);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument rb must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__construct_ball)(ret_construct_ball, rc, rb);
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    if (rc_scalar_is_array) {
        if (rc_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_rc, rc_scalar_arr) < 0) {
                Py_DECREF(rc_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(rc_scalar_arr);
    }
    if (rb_scalar_is_array) {
        if (rb_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_rb, rb_scalar_arr) < 0) {
                Py_DECREF(rb_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(rb_scalar_arr);
    }
    PyObject* py_ret_construct_ball_obj = PyList_New(4);
    if (py_ret_construct_ball_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_construct_ball[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_construct_ball_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_construct_ball_obj, i, item);
    }
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_construct_ball_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_construct_ball_obj != NULL) return py_ret_construct_ball_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_construct_ball_obj != NULL) Py_DECREF(py_ret_construct_ball_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_construct_ball_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_construct_ball_obj);
    }
    return result_tuple;
}

static PyObject* wrap_m_geometry_ball_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__ball_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_list_circle_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle_initialise)(this);
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

static PyObject* wrap_m_geometry_list_circle_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_array_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_initialise)(this);
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

static PyObject* wrap_m_geometry_array_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_array_3d_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_3d_initialise)(this);
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

static PyObject* wrap_m_geometry_array_3d_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_3d_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry_helper_get_pi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__get__pi)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_geometry__rectangle_helper_get_length(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle__get__length)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_geometry__rectangle_helper_set_length(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "length", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle__set__length)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__rectangle_helper_get_width(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle__get__width)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_geometry__rectangle_helper_set_width(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "width", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__rectangle__set__width)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__diamond_helper_get_length(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond__get__length)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_geometry__diamond_helper_set_length(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "length", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond__set__length)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__diamond_helper_get_width(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond__get__width)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_geometry__diamond_helper_set_width(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "width", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__diamond__set__width)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_square_helper_array_getitem_alloc_type(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_getitem__alloc_type)(parent_handle, &index, handle);
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

static PyObject* wrap_m_geometry__list_square_helper_array_setitem_alloc_type(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_setitem__alloc_type)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_square_helper_array_len_alloc_type(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_len__alloc_type)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_m_geometry__list_square_helper_array_getitem_ptr_type(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_getitem__ptr_type)(parent_handle, &index, handle);
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

static PyObject* wrap_m_geometry__list_square_helper_array_setitem_ptr_type(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_setitem__ptr_type)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_square_helper_array_len_ptr_type(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_len__ptr_type)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_m_geometry__list_square_helper_array_getitem_alloc_class(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_getitem__alloc_class)(parent_handle, &index, handle);
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

static PyObject* wrap_m_geometry__list_square_helper_array_setitem_alloc_class(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_setitem__alloc_class)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_square_helper_array_len_alloc_class(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_len__alloc_class)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_m_geometry__list_square_helper_array_getitem_ptr_class(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_getitem__ptr_class)(parent_handle, &index, handle);
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

static PyObject* wrap_m_geometry__list_square_helper_array_setitem_ptr_class(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_setitem__ptr_class)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_square_helper_array_len_ptr_class(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__array_len__ptr_class)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_m_geometry__list_square_helper_get_derived_scalar_class(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__get__scalar_class)(handle_handle, value_handle);
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

static PyObject* wrap_m_geometry__list_square_helper_set_derived_scalar_class(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__set__scalar_class)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_square_helper_get_derived_scalar_type(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__get__scalar_type)(handle_handle, value_handle);
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

static PyObject* wrap_m_geometry__list_square_helper_set_derived_scalar_type(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__set__scalar_type)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_square_helper_get_n(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__get__n)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_geometry__list_square_helper_set_n(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_square__set__n)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__circle_helper_get_radius(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__circle__get__radius)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_m_geometry__circle_helper_set_radius(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "radius", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__circle__set__radius)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_circle_helper_array_getitem_alloc_type(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_getitem__alloc_type)(parent_handle, &index, handle);
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

static PyObject* wrap_m_geometry__list_circle_helper_array_setitem_alloc_type(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_setitem__alloc_type)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_circle_helper_array_len_alloc_type(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_len__alloc_type)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_m_geometry__list_circle_helper_array_getitem_ptr_type(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_getitem__ptr_type)(parent_handle, &index, handle);
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

static PyObject* wrap_m_geometry__list_circle_helper_array_setitem_ptr_type(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_setitem__ptr_type)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_circle_helper_array_len_ptr_type(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_len__ptr_type)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_m_geometry__list_circle_helper_array_getitem_alloc_class(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_getitem__alloc_class)(parent_handle, &index, handle);
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

static PyObject* wrap_m_geometry__list_circle_helper_array_setitem_alloc_class(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_setitem__alloc_class)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_circle_helper_array_len_alloc_class(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_len__alloc_class)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_m_geometry__list_circle_helper_array_getitem_ptr_class(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_getitem__ptr_class)(parent_handle, &index, handle);
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

static PyObject* wrap_m_geometry__list_circle_helper_array_setitem_ptr_class(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_setitem__ptr_class)(parent_handle, &index, value);
    free(value);
    Py_DECREF(value_sequence);
    if (value_handle_obj) Py_DECREF(value_handle_obj);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_circle_helper_array_len_ptr_class(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__array_len__ptr_class)(parent_handle, &length);
    Py_DECREF(parent_sequence);
    return PyLong_FromLong((long)length);
}

static PyObject* wrap_m_geometry__list_circle_helper_get_derived_scalar_class(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__get__scalar_class)(handle_handle, value_handle);
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

static PyObject* wrap_m_geometry__list_circle_helper_set_derived_scalar_class(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__set__scalar_class)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_circle_helper_get_derived_scalar_type(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__get__scalar_type)(handle_handle, value_handle);
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

static PyObject* wrap_m_geometry__list_circle_helper_set_derived_scalar_type(PyObject* self, PyObject* args, PyObject* \
    kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__set__scalar_type)(parent_handle, value_handle);
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__list_circle_helper_get_n(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__get__n)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_m_geometry__list_circle_helper_set_n(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__list_circle__set__n)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_m_geometry__array_helper_array_buf(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__array__array__buf)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_m_geometry__array_helper_array_values(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__array__array__values)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_m_geometry__array_3d_helper_array_values_3d(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__array_3d__array__values_3d)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap__m_base_poly__is_polygone__binding__polygone(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    int ret_is_polygone_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_base_poly__is_polygone__binding__polygone)(this, &ret_is_polygone_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_is_polygone_obj = Py_BuildValue("i", ret_is_polygone_val);
    if (py_ret_is_polygone_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_is_polygone_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_is_polygone_obj != NULL) return py_ret_is_polygone_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_is_polygone_obj != NULL) Py_DECREF(py_ret_is_polygone_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_is_polygone_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_is_polygone_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__perimeter__binding__rectangle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    double ret_perimeter_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__perimeter__binding__rectangle)(this, &ret_perimeter_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_perimeter_obj = Py_BuildValue("d", ret_perimeter_val);
    if (py_ret_perimeter_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_perimeter_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_perimeter_obj != NULL) return py_ret_perimeter_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_perimeter_obj != NULL) Py_DECREF(py_ret_perimeter_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_perimeter_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_perimeter_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__is_square__binding__rectangle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    int ret_is_square_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_square__binding__rectangle)(this, &ret_is_square_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_is_square_obj = Py_BuildValue("i", ret_is_square_val);
    if (py_ret_is_square_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_is_square_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_is_square_obj != NULL) return py_ret_is_square_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_is_square_obj != NULL) Py_DECREF(py_ret_is_square_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_is_square_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_is_square_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__area__binding__rectangle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    double ret_area_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__area__binding__rectangle)(this, &ret_area_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_area_obj = Py_BuildValue("d", ret_area_val);
    if (py_ret_area_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_area_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_area_obj != NULL) return py_ret_area_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_area_obj != NULL) Py_DECREF(py_ret_area_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_area_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_area_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__is_polygone__binding__rectangle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    int ret_is_polygone_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_base_poly__is_polygone__binding__polygone_rectangle)(this, &ret_is_polygone_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_is_polygone_obj = Py_BuildValue("i", ret_is_polygone_val);
    if (py_ret_is_polygone_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_is_polygone_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_is_polygone_obj != NULL) return py_ret_is_polygone_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_is_polygone_obj != NULL) Py_DECREF(py_ret_is_polygone_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_is_polygone_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_is_polygone_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__init__binding__square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_length = NULL;
    float length_val = 0;
    PyArrayObject* length_scalar_arr = NULL;
    int length_scalar_copyback = 0;
    int length_scalar_is_array = 0;
    static char *kwlist[] = {"this", "length", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_length)) {
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
    
    float* length = &length_val;
    if (PyArray_Check(py_length)) {
        length_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_length, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (length_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(length_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument length must have exactly one element");
            Py_DECREF(length_scalar_arr);
            return NULL;
        }
        length_scalar_is_array = 1;
        length = (float*)PyArray_DATA(length_scalar_arr);
        length_val = length[0];
        if (PyArray_DATA(length_scalar_arr) != PyArray_DATA((PyArrayObject*)py_length) || PyArray_TYPE(length_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_length)) {
            length_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_length)) {
        length_val = (float)PyFloat_AsDouble(py_length);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument length must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__square)(this, length);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    if (length_scalar_is_array) {
        if (length_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_length, length_scalar_arr) < 0) {
                Py_DECREF(length_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(length_scalar_arr);
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

static PyObject* wrap__m_geometry__is_square__binding__square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    int ret_is_square_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_square__binding__square)(this, &ret_is_square_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_is_square_obj = Py_BuildValue("i", ret_is_square_val);
    if (py_ret_is_square_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_is_square_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_is_square_obj != NULL) return py_ret_is_square_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_is_square_obj != NULL) Py_DECREF(py_ret_is_square_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_is_square_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_is_square_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__area__binding__square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    double ret_area_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__area__binding__square)(this, &ret_area_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_area_obj = Py_BuildValue("d", ret_area_val);
    if (py_ret_area_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_area_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_area_obj != NULL) return py_ret_area_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_area_obj != NULL) Py_DECREF(py_ret_area_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_area_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_area_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__is_equal__binding__square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_other = NULL;
    int ret_is_equal_val = 0;
    static char *kwlist[] = {"this", "other", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_other)) {
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
    
    PyObject* other_handle_obj = NULL;
    PyObject* other_sequence = NULL;
    Py_ssize_t other_handle_len = 0;
    if (PyObject_HasAttrString(py_other, "_handle")) {
        other_handle_obj = PyObject_GetAttrString(py_other, "_handle");
        if (other_handle_obj == NULL) {
            return NULL;
        }
        other_sequence = PySequence_Fast(other_handle_obj, "Failed to access handle sequence");
        if (other_sequence == NULL) {
            Py_DECREF(other_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_other)) {
        other_sequence = PySequence_Fast(py_other, "Argument other must be a handle sequence");
        if (other_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument other must be a Fortran derived-type instance");
        return NULL;
    }
    other_handle_len = PySequence_Fast_GET_SIZE(other_sequence);
    if (other_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument other has an invalid handle length");
        Py_DECREF(other_sequence);
        if (other_handle_obj) Py_DECREF(other_handle_obj);
        return NULL;
    }
    int* other = (int*)malloc(sizeof(int) * other_handle_len);
    if (other == NULL) {
        PyErr_NoMemory();
        Py_DECREF(other_sequence);
        if (other_handle_obj) Py_DECREF(other_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < other_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(other_sequence, i);
        if (item == NULL) {
            free(other);
            Py_DECREF(other_sequence);
            if (other_handle_obj) Py_DECREF(other_handle_obj);
            return NULL;
        }
        other[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(other);
            Py_DECREF(other_sequence);
            if (other_handle_obj) Py_DECREF(other_handle_obj);
            return NULL;
        }
    }
    (void)other_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__is_equal__binding__square)(this, other, &ret_is_equal_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        if (other_sequence) Py_DECREF(other_sequence);
        if (other_handle_obj) Py_DECREF(other_handle_obj);
        free(other);
        return NULL;
    }
    
    PyObject* py_ret_is_equal_obj = Py_BuildValue("i", ret_is_equal_val);
    if (py_ret_is_equal_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    if (other_sequence) {
        Py_DECREF(other_sequence);
    }
    if (other_handle_obj) {
        Py_DECREF(other_handle_obj);
    }
    free(other);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_is_equal_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_is_equal_obj != NULL) return py_ret_is_equal_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_is_equal_obj != NULL) Py_DECREF(py_ret_is_equal_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_is_equal_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_is_equal_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__copy__binding__square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_from_ = NULL;
    static char *kwlist[] = {"this", "from_", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_from_)) {
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
    
    PyObject* from__handle_obj = NULL;
    PyObject* from__sequence = NULL;
    Py_ssize_t from__handle_len = 0;
    if (PyObject_HasAttrString(py_from_, "_handle")) {
        from__handle_obj = PyObject_GetAttrString(py_from_, "_handle");
        if (from__handle_obj == NULL) {
            return NULL;
        }
        from__sequence = PySequence_Fast(from__handle_obj, "Failed to access handle sequence");
        if (from__sequence == NULL) {
            Py_DECREF(from__handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_from_)) {
        from__sequence = PySequence_Fast(py_from_, "Argument from_ must be a handle sequence");
        if (from__sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument from_ must be a Fortran derived-type instance");
        return NULL;
    }
    from__handle_len = PySequence_Fast_GET_SIZE(from__sequence);
    if (from__handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument from_ has an invalid handle length");
        Py_DECREF(from__sequence);
        if (from__handle_obj) Py_DECREF(from__handle_obj);
        return NULL;
    }
    int* from_ = (int*)malloc(sizeof(int) * from__handle_len);
    if (from_ == NULL) {
        PyErr_NoMemory();
        Py_DECREF(from__sequence);
        if (from__handle_obj) Py_DECREF(from__handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < from__handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(from__sequence, i);
        if (item == NULL) {
            free(from_);
            Py_DECREF(from__sequence);
            if (from__handle_obj) Py_DECREF(from__handle_obj);
            return NULL;
        }
        from_[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(from_);
            Py_DECREF(from__sequence);
            if (from__handle_obj) Py_DECREF(from__handle_obj);
            return NULL;
        }
    }
    (void)from__handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__copy__binding__square)(this, from_);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        if (from__sequence) Py_DECREF(from__sequence);
        if (from__handle_obj) Py_DECREF(from__handle_obj);
        free(from_);
        return NULL;
    }
    
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    if (from__sequence) {
        Py_DECREF(from__sequence);
    }
    if (from__handle_obj) {
        Py_DECREF(from__handle_obj);
    }
    free(from_);
    Py_RETURN_NONE;
}

static PyObject* wrap__m_geometry__create_diamond__binding__square(PyObject* self, PyObject* args, PyObject* kwargs)
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
    
    int ret_square_create_diamond[4] = {0};
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__create_diamond__binding__square)(this, ret_square_create_diamond);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_square_create_diamond_obj = PyList_New(4);
    if (py_ret_square_create_diamond_obj == NULL) {
        return NULL;
    }
    for (int i = 0; i < 4; ++i) {
        PyObject* item = PyLong_FromLong((long)ret_square_create_diamond[i]);
        if (item == NULL) {
            Py_DECREF(py_ret_square_create_diamond_obj);
            return NULL;
        }
        PyList_SET_ITEM(py_ret_square_create_diamond_obj, i, item);
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_square_create_diamond_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_square_create_diamond_obj != NULL) return py_ret_square_create_diamond_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_square_create_diamond_obj != NULL) Py_DECREF(py_ret_square_create_diamond_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_square_create_diamond_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_square_create_diamond_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__perimeter__binding__square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    double ret_perimeter_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__perimeter__binding__rectangle_square)(this, &ret_perimeter_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_perimeter_obj = Py_BuildValue("d", ret_perimeter_val);
    if (py_ret_perimeter_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_perimeter_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_perimeter_obj != NULL) return py_ret_perimeter_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_perimeter_obj != NULL) Py_DECREF(py_ret_perimeter_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_perimeter_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_perimeter_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__is_polygone__binding__square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    int ret_is_polygone_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_base_poly__is_polygone__binding__polygone_rectang5400)(this, &ret_is_polygone_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_is_polygone_obj = Py_BuildValue("i", ret_is_polygone_val);
    if (py_ret_is_polygone_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_is_polygone_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_is_polygone_obj != NULL) return py_ret_is_polygone_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_is_polygone_obj != NULL) Py_DECREF(py_ret_is_polygone_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_is_polygone_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_is_polygone_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__init__binding__diamond(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_width = NULL;
    double width_val = 0;
    PyArrayObject* width_scalar_arr = NULL;
    int width_scalar_copyback = 0;
    int width_scalar_is_array = 0;
    PyObject* py_length = NULL;
    double length_val = 0;
    PyArrayObject* length_scalar_arr = NULL;
    int length_scalar_copyback = 0;
    int length_scalar_is_array = 0;
    static char *kwlist[] = {"this", "width", "length", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO", kwlist, &py_this, &py_width, &py_length)) {
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
    
    double* width = &width_val;
    if (PyArray_Check(py_width)) {
        width_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_width, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (width_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(width_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument width must have exactly one element");
            Py_DECREF(width_scalar_arr);
            return NULL;
        }
        width_scalar_is_array = 1;
        width = (double*)PyArray_DATA(width_scalar_arr);
        width_val = width[0];
        if (PyArray_DATA(width_scalar_arr) != PyArray_DATA((PyArrayObject*)py_width) || PyArray_TYPE(width_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_width)) {
            width_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_width)) {
        width_val = (double)PyFloat_AsDouble(py_width);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument width must be a scalar number or NumPy array");
        return NULL;
    }
    double* length = &length_val;
    if (PyArray_Check(py_length)) {
        length_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_length, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (length_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(length_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument length must have exactly one element");
            Py_DECREF(length_scalar_arr);
            return NULL;
        }
        length_scalar_is_array = 1;
        length = (double*)PyArray_DATA(length_scalar_arr);
        length_val = length[0];
        if (PyArray_DATA(length_scalar_arr) != PyArray_DATA((PyArrayObject*)py_length) || PyArray_TYPE(length_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_length)) {
            length_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_length)) {
        length_val = (double)PyFloat_AsDouble(py_length);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument length must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__diamond)(this, width, length);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    if (width_scalar_is_array) {
        if (width_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_width, width_scalar_arr) < 0) {
                Py_DECREF(width_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(width_scalar_arr);
    }
    if (length_scalar_is_array) {
        if (length_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_length, length_scalar_arr) < 0) {
                Py_DECREF(length_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(length_scalar_arr);
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

static PyObject* wrap__m_geometry__info__binding__diamond(PyObject* self, PyObject* args, PyObject* kwargs)
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__info__binding__diamond)(this);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
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

static PyObject* wrap__m_geometry__copy__binding__diamond(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_other = NULL;
    static char *kwlist[] = {"this", "other", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_other)) {
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
    
    PyObject* other_handle_obj = NULL;
    PyObject* other_sequence = NULL;
    Py_ssize_t other_handle_len = 0;
    if (PyObject_HasAttrString(py_other, "_handle")) {
        other_handle_obj = PyObject_GetAttrString(py_other, "_handle");
        if (other_handle_obj == NULL) {
            return NULL;
        }
        other_sequence = PySequence_Fast(other_handle_obj, "Failed to access handle sequence");
        if (other_sequence == NULL) {
            Py_DECREF(other_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_other)) {
        other_sequence = PySequence_Fast(py_other, "Argument other must be a handle sequence");
        if (other_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument other must be a Fortran derived-type instance");
        return NULL;
    }
    other_handle_len = PySequence_Fast_GET_SIZE(other_sequence);
    if (other_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument other has an invalid handle length");
        Py_DECREF(other_sequence);
        if (other_handle_obj) Py_DECREF(other_handle_obj);
        return NULL;
    }
    int* other = (int*)malloc(sizeof(int) * other_handle_len);
    if (other == NULL) {
        PyErr_NoMemory();
        Py_DECREF(other_sequence);
        if (other_handle_obj) Py_DECREF(other_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < other_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(other_sequence, i);
        if (item == NULL) {
            free(other);
            Py_DECREF(other_sequence);
            if (other_handle_obj) Py_DECREF(other_handle_obj);
            return NULL;
        }
        other[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(other);
            Py_DECREF(other_sequence);
            if (other_handle_obj) Py_DECREF(other_handle_obj);
            return NULL;
        }
    }
    (void)other_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__copy__binding__diamond)(this, other);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        if (other_sequence) Py_DECREF(other_sequence);
        if (other_handle_obj) Py_DECREF(other_handle_obj);
        free(other);
        return NULL;
    }
    
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    if (other_sequence) {
        Py_DECREF(other_sequence);
    }
    if (other_handle_obj) {
        Py_DECREF(other_handle_obj);
    }
    free(other);
    Py_RETURN_NONE;
}

static PyObject* wrap__m_geometry__is_polygone__binding__diamond(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    int ret_is_polygone_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_base_poly__is_polygone__binding__polygone_diamond)(this, &ret_is_polygone_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_is_polygone_obj = Py_BuildValue("i", ret_is_polygone_val);
    if (py_ret_is_polygone_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_is_polygone_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_is_polygone_obj != NULL) return py_ret_is_polygone_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_is_polygone_obj != NULL) Py_DECREF(py_ret_is_polygone_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_is_polygone_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_is_polygone_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__init__binding__list_square(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"this", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_n)) {
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__list_square)(this, n);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
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
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap__m_geometry__area__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    double ret_area_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__area__binding__circle)(this, &ret_area_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_area_obj = Py_BuildValue("d", ret_area_val);
    if (py_ret_area_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_area_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_area_obj != NULL) return py_ret_area_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_area_obj != NULL) Py_DECREF(py_ret_area_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_area_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_area_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__print__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__print__binding__circle)(this);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
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

static PyObject* wrap__m_geometry__obj_name__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_obj = NULL;
    static char *kwlist[] = {"obj", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_obj)) {
        return NULL;
    }
    
    PyObject* obj_handle_obj = NULL;
    PyObject* obj_sequence = NULL;
    Py_ssize_t obj_handle_len = 0;
    if (PyObject_HasAttrString(py_obj, "_handle")) {
        obj_handle_obj = PyObject_GetAttrString(py_obj, "_handle");
        if (obj_handle_obj == NULL) {
            return NULL;
        }
        obj_sequence = PySequence_Fast(obj_handle_obj, "Failed to access handle sequence");
        if (obj_sequence == NULL) {
            Py_DECREF(obj_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_obj)) {
        obj_sequence = PySequence_Fast(py_obj, "Argument obj must be a handle sequence");
        if (obj_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument obj must be a Fortran derived-type instance");
        return NULL;
    }
    obj_handle_len = PySequence_Fast_GET_SIZE(obj_sequence);
    if (obj_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument obj has an invalid handle length");
        Py_DECREF(obj_sequence);
        if (obj_handle_obj) Py_DECREF(obj_handle_obj);
        return NULL;
    }
    int* obj = (int*)malloc(sizeof(int) * obj_handle_len);
    if (obj == NULL) {
        PyErr_NoMemory();
        Py_DECREF(obj_sequence);
        if (obj_handle_obj) Py_DECREF(obj_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < obj_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(obj_sequence, i);
        if (item == NULL) {
            free(obj);
            Py_DECREF(obj_sequence);
            if (obj_handle_obj) Py_DECREF(obj_handle_obj);
            return NULL;
        }
        obj[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(obj);
            Py_DECREF(obj_sequence);
            if (obj_handle_obj) Py_DECREF(obj_handle_obj);
            return NULL;
        }
    }
    (void)obj_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__obj_name__binding__circle)(obj);
    if (PyErr_Occurred()) {
        if (obj_sequence) Py_DECREF(obj_sequence);
        if (obj_handle_obj) Py_DECREF(obj_handle_obj);
        free(obj);
        return NULL;
    }
    
    if (obj_sequence) {
        Py_DECREF(obj_sequence);
    }
    if (obj_handle_obj) {
        Py_DECREF(obj_handle_obj);
    }
    free(obj);
    Py_RETURN_NONE;
}

static PyObject* wrap__m_geometry__copy__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_from_ = NULL;
    static char *kwlist[] = {"this", "from_", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_from_)) {
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
    
    PyObject* from__handle_obj = NULL;
    PyObject* from__sequence = NULL;
    Py_ssize_t from__handle_len = 0;
    if (PyObject_HasAttrString(py_from_, "_handle")) {
        from__handle_obj = PyObject_GetAttrString(py_from_, "_handle");
        if (from__handle_obj == NULL) {
            return NULL;
        }
        from__sequence = PySequence_Fast(from__handle_obj, "Failed to access handle sequence");
        if (from__sequence == NULL) {
            Py_DECREF(from__handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_from_)) {
        from__sequence = PySequence_Fast(py_from_, "Argument from_ must be a handle sequence");
        if (from__sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument from_ must be a Fortran derived-type instance");
        return NULL;
    }
    from__handle_len = PySequence_Fast_GET_SIZE(from__sequence);
    if (from__handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument from_ has an invalid handle length");
        Py_DECREF(from__sequence);
        if (from__handle_obj) Py_DECREF(from__handle_obj);
        return NULL;
    }
    int* from_ = (int*)malloc(sizeof(int) * from__handle_len);
    if (from_ == NULL) {
        PyErr_NoMemory();
        Py_DECREF(from__sequence);
        if (from__handle_obj) Py_DECREF(from__handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < from__handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(from__sequence, i);
        if (item == NULL) {
            free(from_);
            Py_DECREF(from__sequence);
            if (from__handle_obj) Py_DECREF(from__handle_obj);
            return NULL;
        }
        from_[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(from_);
            Py_DECREF(from__sequence);
            if (from__handle_obj) Py_DECREF(from__handle_obj);
            return NULL;
        }
    }
    (void)from__handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__copy__binding__circle)(this, from_);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        if (from__sequence) Py_DECREF(from__sequence);
        if (from__handle_obj) Py_DECREF(from__handle_obj);
        free(from_);
        return NULL;
    }
    
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    if (from__sequence) {
        Py_DECREF(from__sequence);
    }
    if (from__handle_obj) {
        Py_DECREF(from__handle_obj);
    }
    free(from_);
    Py_RETURN_NONE;
}

static PyObject* wrap__m_geometry__init__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    static char *kwlist[] = {"this", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_radius)) {
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
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__circle)(this, radius);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
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

static PyObject* wrap__m_geometry__private_method__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__private_method__binding__circle)(this);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
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

static PyObject* wrap__m_geometry__perimeter_4__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_radius = NULL;
    float radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    float ret_perimeter_val = 0;
    static char *kwlist[] = {"this", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_radius)) {
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
    
    float* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT32, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (float*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (float)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__perimeter_4__binding__circle)(this, radius, &ret_perimeter_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    PyObject* py_ret_perimeter_obj = Py_BuildValue("d", ret_perimeter_val);
    if (py_ret_perimeter_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_perimeter_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_perimeter_obj != NULL) return py_ret_perimeter_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_perimeter_obj != NULL) Py_DECREF(py_ret_perimeter_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_perimeter_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_perimeter_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__perimeter_8__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_radius = NULL;
    double radius_val = 0;
    PyArrayObject* radius_scalar_arr = NULL;
    int radius_scalar_copyback = 0;
    int radius_scalar_is_array = 0;
    double ret_perimeter_val = 0;
    static char *kwlist[] = {"this", "radius", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_radius)) {
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
    
    double* radius = &radius_val;
    if (PyArray_Check(py_radius)) {
        radius_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_radius, NPY_FLOAT64, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (radius_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(radius_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument radius must have exactly one element");
            Py_DECREF(radius_scalar_arr);
            return NULL;
        }
        radius_scalar_is_array = 1;
        radius = (double*)PyArray_DATA(radius_scalar_arr);
        radius_val = radius[0];
        if (PyArray_DATA(radius_scalar_arr) != PyArray_DATA((PyArrayObject*)py_radius) || PyArray_TYPE(radius_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_radius)) {
            radius_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_radius)) {
        radius_val = (double)PyFloat_AsDouble(py_radius);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument radius must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__perimeter_8__binding__circle)(this, radius, &ret_perimeter_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    if (radius_scalar_is_array) {
        if (radius_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_radius, radius_scalar_arr) < 0) {
                Py_DECREF(radius_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(radius_scalar_arr);
    }
    PyObject* py_ret_perimeter_obj = Py_BuildValue("d", ret_perimeter_val);
    if (py_ret_perimeter_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_perimeter_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_perimeter_obj != NULL) return py_ret_perimeter_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_perimeter_obj != NULL) Py_DECREF(py_ret_perimeter_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_perimeter_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_perimeter_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__circle_free__binding__circle(PyObject* self, PyObject* args, PyObject* kwargs)
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__circle_free__binding__circle)(this);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
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

static PyObject* wrap__m_geometry__volume__binding__ball(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    double ret_volume_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__volume__binding__ball)(this, &ret_volume_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_volume_obj = Py_BuildValue("d", ret_volume_val);
    if (py_ret_volume_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_volume_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_volume_obj != NULL) return py_ret_volume_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_volume_obj != NULL) Py_DECREF(py_ret_volume_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_volume_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_volume_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__area__binding__ball(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    double ret_area_val = 0;
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__area__binding__ball)(this, &ret_area_val);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    PyObject* py_ret_area_obj = Py_BuildValue("d", ret_area_val);
    if (py_ret_area_obj == NULL) {
        return NULL;
    }
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    /* Build result tuple, filtering out NULL objects */
    int result_count = 0;
    if (py_ret_area_obj != NULL) result_count++;
    if (result_count == 0) {
        Py_RETURN_NONE;
    }
    if (result_count == 1) {
        if (py_ret_area_obj != NULL) return py_ret_area_obj;
    }
    PyObject* result_tuple = PyTuple_New(result_count);
    if (result_tuple == NULL) {
        if (py_ret_area_obj != NULL) Py_DECREF(py_ret_area_obj);
        return NULL;
    }
    int tuple_index = 0;
    if (py_ret_area_obj != NULL) {
        PyTuple_SET_ITEM(result_tuple, tuple_index++, py_ret_area_obj);
    }
    return result_tuple;
}

static PyObject* wrap__m_geometry__private_method__binding__ball(PyObject* self, PyObject* args, PyObject* kwargs)
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
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__private_method__binding__ball)(this);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
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

static PyObject* wrap__m_geometry__init__binding__list_circle(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"this", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_n)) {
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__list_circle)(this, n);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
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
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap__m_geometry__init__binding__array(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_n = NULL;
    int n_val = 0;
    PyArrayObject* n_scalar_arr = NULL;
    int n_scalar_copyback = 0;
    int n_scalar_is_array = 0;
    static char *kwlist[] = {"this", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist, &py_this, &py_n)) {
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
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__init__binding__array)(this, n);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
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
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap__m_geometry__init_3d__binding__array_3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_this = NULL;
    PyObject* py_n1 = NULL;
    int n1_val = 0;
    PyArrayObject* n1_scalar_arr = NULL;
    int n1_scalar_copyback = 0;
    int n1_scalar_is_array = 0;
    PyObject* py_n2 = NULL;
    int n2_val = 0;
    PyArrayObject* n2_scalar_arr = NULL;
    int n2_scalar_copyback = 0;
    int n2_scalar_is_array = 0;
    PyObject* py_n3 = NULL;
    int n3_val = 0;
    PyArrayObject* n3_scalar_arr = NULL;
    int n3_scalar_copyback = 0;
    int n3_scalar_is_array = 0;
    static char *kwlist[] = {"this", "n1", "n2", "n3", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO", kwlist, &py_this, &py_n1, &py_n2, &py_n3)) {
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
    
    int* n1 = &n1_val;
    if (PyArray_Check(py_n1)) {
        n1_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n1, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n1_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n1_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n1 must have exactly one element");
            Py_DECREF(n1_scalar_arr);
            return NULL;
        }
        n1_scalar_is_array = 1;
        n1 = (int*)PyArray_DATA(n1_scalar_arr);
        n1_val = n1[0];
        if (PyArray_DATA(n1_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n1) || PyArray_TYPE(n1_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n1)) {
            n1_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n1)) {
        n1_val = (int)PyLong_AsLong(py_n1);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n1 must be a scalar number or NumPy array");
        return NULL;
    }
    int* n2 = &n2_val;
    if (PyArray_Check(py_n2)) {
        n2_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n2, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n2_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n2_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n2 must have exactly one element");
            Py_DECREF(n2_scalar_arr);
            return NULL;
        }
        n2_scalar_is_array = 1;
        n2 = (int*)PyArray_DATA(n2_scalar_arr);
        n2_val = n2[0];
        if (PyArray_DATA(n2_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n2) || PyArray_TYPE(n2_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n2)) {
            n2_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n2)) {
        n2_val = (int)PyLong_AsLong(py_n2);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n2 must be a scalar number or NumPy array");
        return NULL;
    }
    int* n3 = &n3_val;
    if (PyArray_Check(py_n3)) {
        n3_scalar_arr = (PyArrayObject*)PyArray_FROM_OTF(
            py_n3, NPY_INT, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_FORCECAST);
        if (n3_scalar_arr == NULL) {
            return NULL;
        }
        if (PyArray_SIZE(n3_scalar_arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "Argument n3 must have exactly one element");
            Py_DECREF(n3_scalar_arr);
            return NULL;
        }
        n3_scalar_is_array = 1;
        n3 = (int*)PyArray_DATA(n3_scalar_arr);
        n3_val = n3[0];
        if (PyArray_DATA(n3_scalar_arr) != PyArray_DATA((PyArrayObject*)py_n3) || PyArray_TYPE(n3_scalar_arr) != \
            PyArray_TYPE((PyArrayObject*)py_n3)) {
            n3_scalar_copyback = 1;
        }
    } else if (PyNumber_Check(py_n3)) {
        n3_val = (int)PyLong_AsLong(py_n3);
        if (PyErr_Occurred()) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument n3 must be a scalar number or NumPy array");
        return NULL;
    }
    /* Call f90wrap helper */
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_m_geometry__init_3d__binding__array_3d)(this, n1, n2, n3);
    if (PyErr_Occurred()) {
        if (this_sequence) Py_DECREF(this_sequence);
        if (this_handle_obj) Py_DECREF(this_handle_obj);
        free(this);
        return NULL;
    }
    
    if (n1_scalar_is_array) {
        if (n1_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n1, n1_scalar_arr) < 0) {
                Py_DECREF(n1_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n1_scalar_arr);
    }
    if (n2_scalar_is_array) {
        if (n2_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n2, n2_scalar_arr) < 0) {
                Py_DECREF(n2_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n2_scalar_arr);
    }
    if (n3_scalar_is_array) {
        if (n3_scalar_copyback) {
            if (PyArray_CopyInto((PyArrayObject*)py_n3, n3_scalar_arr) < 0) {
                Py_DECREF(n3_scalar_arr);
                return NULL;
            }
        }
        Py_DECREF(n3_scalar_arr);
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

/* Method table for _pywrapper module */
static PyMethodDef _pywrapper_methods[] = {
    {"f90wrap_m_base_poly__polygone_initialise", (PyCFunction)wrap_m_base_poly_polygone_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for polygone"},
    {"f90wrap_m_base_poly__polygone_finalise", (PyCFunction)wrap_m_base_poly_polygone_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for polygone"},
    {"f90wrap_m_geometry__get_circle_radius", (PyCFunction)wrap_m_geometry_get_circle_radius, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for get_circle_radius"},
    {"f90wrap_m_geometry__get_ball_radius", (PyCFunction)wrap_m_geometry_get_ball_radius, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for get_ball_radius"},
    {"f90wrap_m_geometry__rectangle_initialise", (PyCFunction)wrap_m_geometry_rectangle_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for rectangle"},
    {"f90wrap_m_geometry__rectangle_finalise", (PyCFunction)wrap_m_geometry_rectangle_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for rectangle"},
    {"f90wrap_m_geometry__construct_square", (PyCFunction)wrap_m_geometry_construct_square, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for construct_square"},
    {"f90wrap_m_geometry__square_finalise", (PyCFunction)wrap_m_geometry_square_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for square"},
    {"f90wrap_m_geometry__diamond_initialise", (PyCFunction)wrap_m_geometry_diamond_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for diamond"},
    {"f90wrap_m_geometry__diamond_finalise", (PyCFunction)wrap_m_geometry_diamond_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for diamond"},
    {"f90wrap_m_geometry__list_square_initialise", (PyCFunction)wrap_m_geometry_list_square_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for list_square"},
    {"f90wrap_m_geometry__list_square_finalise", (PyCFunction)wrap_m_geometry_list_square_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for list_square"},
    {"f90wrap_m_geometry__construct_circle", (PyCFunction)wrap_m_geometry_construct_circle, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for construct_circle"},
    {"f90wrap_m_geometry__construct_ball", (PyCFunction)wrap_m_geometry_construct_ball, METH_VARARGS | METH_KEYWORDS, \
        "Wrapper for construct_ball"},
    {"f90wrap_m_geometry__ball_finalise", (PyCFunction)wrap_m_geometry_ball_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for ball"},
    {"f90wrap_m_geometry__list_circle_initialise", (PyCFunction)wrap_m_geometry_list_circle_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for list_circle"},
    {"f90wrap_m_geometry__list_circle_finalise", (PyCFunction)wrap_m_geometry_list_circle_finalise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated destructor for list_circle"},
    {"f90wrap_m_geometry__array_initialise", (PyCFunction)wrap_m_geometry_array_initialise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated constructor for array"},
    {"f90wrap_m_geometry__array_finalise", (PyCFunction)wrap_m_geometry_array_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for array"},
    {"f90wrap_m_geometry__array_3d_initialise", (PyCFunction)wrap_m_geometry_array_3d_initialise, METH_VARARGS | \
        METH_KEYWORDS, "Automatically generated constructor for array_3d"},
    {"f90wrap_m_geometry__array_3d_finalise", (PyCFunction)wrap_m_geometry_array_3d_finalise, METH_VARARGS | METH_KEYWORDS, \
        "Automatically generated destructor for array_3d"},
    {"f90wrap_m_geometry__get__pi", (PyCFunction)wrap_m_geometry_helper_get_pi, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for pi"},
    {"f90wrap_m_geometry__rectangle__get__length", (PyCFunction)wrap_m_geometry__rectangle_helper_get_length, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for length"},
    {"f90wrap_m_geometry__rectangle__set__length", (PyCFunction)wrap_m_geometry__rectangle_helper_set_length, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for length"},
    {"f90wrap_m_geometry__rectangle__get__width", (PyCFunction)wrap_m_geometry__rectangle_helper_get_width, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for width"},
    {"f90wrap_m_geometry__rectangle__set__width", (PyCFunction)wrap_m_geometry__rectangle_helper_set_width, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for width"},
    {"f90wrap_m_geometry__diamond__get__length", (PyCFunction)wrap_m_geometry__diamond_helper_get_length, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for length"},
    {"f90wrap_m_geometry__diamond__set__length", (PyCFunction)wrap_m_geometry__diamond_helper_set_length, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for length"},
    {"f90wrap_m_geometry__diamond__get__width", (PyCFunction)wrap_m_geometry__diamond_helper_get_width, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for width"},
    {"f90wrap_m_geometry__diamond__set__width", (PyCFunction)wrap_m_geometry__diamond_helper_set_width, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for width"},
    {"f90wrap_m_geometry__list_square__array_getitem__alloc_type", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_getitem_alloc_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for alloc_type"},
    {"f90wrap_m_geometry__list_square__array_setitem__alloc_type", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_setitem_alloc_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for alloc_type"},
    {"f90wrap_m_geometry__list_square__array_len__alloc_type", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_len_alloc_type, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for alloc_type"},
    {"f90wrap_m_geometry__list_square__array_getitem__ptr_type", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_getitem_ptr_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ptr_type"},
    {"f90wrap_m_geometry__list_square__array_setitem__ptr_type", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_setitem_ptr_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ptr_type"},
    {"f90wrap_m_geometry__list_square__array_len__ptr_type", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_len_ptr_type, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for ptr_type"},
    {"f90wrap_m_geometry__list_square__array_getitem__alloc_class", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_getitem_alloc_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for alloc_class"},
    {"f90wrap_m_geometry__list_square__array_setitem__alloc_class", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_setitem_alloc_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for alloc_class"},
    {"f90wrap_m_geometry__list_square__array_len__alloc_class", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_len_alloc_class, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for alloc_class"},
    {"f90wrap_m_geometry__list_square__array_getitem__ptr_class", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_getitem_ptr_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ptr_class"},
    {"f90wrap_m_geometry__list_square__array_setitem__ptr_class", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_setitem_ptr_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ptr_class"},
    {"f90wrap_m_geometry__list_square__array_len__ptr_class", \
        (PyCFunction)wrap_m_geometry__list_square_helper_array_len_ptr_class, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for ptr_class"},
    {"f90wrap_m_geometry__list_square__get__scalar_class", \
        (PyCFunction)wrap_m_geometry__list_square_helper_get_derived_scalar_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for scalar_class"},
    {"f90wrap_m_geometry__list_square__set__scalar_class", \
        (PyCFunction)wrap_m_geometry__list_square_helper_set_derived_scalar_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for scalar_class"},
    {"f90wrap_m_geometry__list_square__get__scalar_type", \
        (PyCFunction)wrap_m_geometry__list_square_helper_get_derived_scalar_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for scalar_type"},
    {"f90wrap_m_geometry__list_square__set__scalar_type", \
        (PyCFunction)wrap_m_geometry__list_square_helper_set_derived_scalar_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for scalar_type"},
    {"f90wrap_m_geometry__list_square__get__n", (PyCFunction)wrap_m_geometry__list_square_helper_get_n, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for n"},
    {"f90wrap_m_geometry__list_square__set__n", (PyCFunction)wrap_m_geometry__list_square_helper_set_n, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for n"},
    {"f90wrap_m_geometry__circle__get__radius", (PyCFunction)wrap_m_geometry__circle_helper_get_radius, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for radius"},
    {"f90wrap_m_geometry__circle__set__radius", (PyCFunction)wrap_m_geometry__circle_helper_set_radius, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for radius"},
    {"f90wrap_m_geometry__list_circle__array_getitem__alloc_type", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_getitem_alloc_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for alloc_type"},
    {"f90wrap_m_geometry__list_circle__array_setitem__alloc_type", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_setitem_alloc_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for alloc_type"},
    {"f90wrap_m_geometry__list_circle__array_len__alloc_type", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_len_alloc_type, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for alloc_type"},
    {"f90wrap_m_geometry__list_circle__array_getitem__ptr_type", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_getitem_ptr_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ptr_type"},
    {"f90wrap_m_geometry__list_circle__array_setitem__ptr_type", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_setitem_ptr_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ptr_type"},
    {"f90wrap_m_geometry__list_circle__array_len__ptr_type", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_len_ptr_type, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for ptr_type"},
    {"f90wrap_m_geometry__list_circle__array_getitem__alloc_class", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_getitem_alloc_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for alloc_class"},
    {"f90wrap_m_geometry__list_circle__array_setitem__alloc_class", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_setitem_alloc_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for alloc_class"},
    {"f90wrap_m_geometry__list_circle__array_len__alloc_class", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_len_alloc_class, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for alloc_class"},
    {"f90wrap_m_geometry__list_circle__array_getitem__ptr_class", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_getitem_ptr_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ptr_class"},
    {"f90wrap_m_geometry__list_circle__array_setitem__ptr_class", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_setitem_ptr_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ptr_class"},
    {"f90wrap_m_geometry__list_circle__array_len__ptr_class", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_array_len_ptr_class, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for ptr_class"},
    {"f90wrap_m_geometry__list_circle__get__scalar_class", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_get_derived_scalar_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for scalar_class"},
    {"f90wrap_m_geometry__list_circle__set__scalar_class", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_set_derived_scalar_class, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for scalar_class"},
    {"f90wrap_m_geometry__list_circle__get__scalar_type", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_get_derived_scalar_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for scalar_type"},
    {"f90wrap_m_geometry__list_circle__set__scalar_type", \
        (PyCFunction)wrap_m_geometry__list_circle_helper_set_derived_scalar_type, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for scalar_type"},
    {"f90wrap_m_geometry__list_circle__get__n", (PyCFunction)wrap_m_geometry__list_circle_helper_get_n, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for n"},
    {"f90wrap_m_geometry__list_circle__set__n", (PyCFunction)wrap_m_geometry__list_circle_helper_set_n, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for n"},
    {"f90wrap_m_geometry__array__array__buf", (PyCFunction)wrap_m_geometry__array_helper_array_buf, METH_VARARGS | \
        METH_KEYWORDS, "Array helper for buf"},
    {"f90wrap_m_geometry__array__array__values", (PyCFunction)wrap_m_geometry__array_helper_array_values, METH_VARARGS | \
        METH_KEYWORDS, "Array helper for values"},
    {"f90wrap_m_geometry__array_3d__array__values_3d", (PyCFunction)wrap_m_geometry__array_3d_helper_array_values_3d, \
        METH_VARARGS | METH_KEYWORDS, "Array helper for values_3d"},
    {"f90wrap_m_base_poly__is_polygone__binding__polygone", (PyCFunction)wrap__m_base_poly__is_polygone__binding__polygone, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for is_polygone"},
    {"f90wrap_m_geometry__perimeter__binding__rectangle", (PyCFunction)wrap__m_geometry__perimeter__binding__rectangle, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for perimeter"},
    {"f90wrap_m_geometry__is_square__binding__rectangle", (PyCFunction)wrap__m_geometry__is_square__binding__rectangle, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for is_square"},
    {"f90wrap_m_geometry__area__binding__rectangle", (PyCFunction)wrap__m_geometry__area__binding__rectangle, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for area"},
    {"f90wrap_m_geometry__is_polygone__binding__rectangle", (PyCFunction)wrap__m_geometry__is_polygone__binding__rectangle, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for is_polygone"},
    {"f90wrap_m_geometry__init__binding__square", (PyCFunction)wrap__m_geometry__init__binding__square, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for init"},
    {"f90wrap_m_geometry__is_square__binding__square", (PyCFunction)wrap__m_geometry__is_square__binding__square, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for is_square"},
    {"f90wrap_m_geometry__area__binding__square", (PyCFunction)wrap__m_geometry__area__binding__square, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for area"},
    {"f90wrap_m_geometry__is_equal__binding__square", (PyCFunction)wrap__m_geometry__is_equal__binding__square, METH_VARARGS \
        | METH_KEYWORDS, "Binding alias for is_equal"},
    {"f90wrap_m_geometry__copy__binding__square", (PyCFunction)wrap__m_geometry__copy__binding__square, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for copy"},
    {"f90wrap_m_geometry__create_diamond__binding__square", (PyCFunction)wrap__m_geometry__create_diamond__binding__square, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for create_diamond"},
    {"f90wrap_m_geometry__perimeter__binding__square", (PyCFunction)wrap__m_geometry__perimeter__binding__square, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for perimeter"},
    {"f90wrap_m_geometry__is_polygone__binding__square", (PyCFunction)wrap__m_geometry__is_polygone__binding__square, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for is_polygone"},
    {"f90wrap_m_geometry__init__binding__diamond", (PyCFunction)wrap__m_geometry__init__binding__diamond, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for init"},
    {"f90wrap_m_geometry__info__binding__diamond", (PyCFunction)wrap__m_geometry__info__binding__diamond, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for info"},
    {"f90wrap_m_geometry__copy__binding__diamond", (PyCFunction)wrap__m_geometry__copy__binding__diamond, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for copy"},
    {"f90wrap_m_geometry__is_polygone__binding__diamond", (PyCFunction)wrap__m_geometry__is_polygone__binding__diamond, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for is_polygone"},
    {"f90wrap_m_geometry__init__binding__list_square", (PyCFunction)wrap__m_geometry__init__binding__list_square, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for init"},
    {"f90wrap_m_geometry__area__binding__circle", (PyCFunction)wrap__m_geometry__area__binding__circle, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for area"},
    {"f90wrap_m_geometry__print__binding__circle", (PyCFunction)wrap__m_geometry__print__binding__circle, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for print"},
    {"f90wrap_m_geometry__obj_name__binding__circle", (PyCFunction)wrap__m_geometry__obj_name__binding__circle, METH_VARARGS \
        | METH_KEYWORDS, "Binding alias for obj_name"},
    {"f90wrap_m_geometry__copy__binding__circle", (PyCFunction)wrap__m_geometry__copy__binding__circle, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for copy"},
    {"f90wrap_m_geometry__init__binding__circle", (PyCFunction)wrap__m_geometry__init__binding__circle, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for init"},
    {"f90wrap_m_geometry__private_method__binding__circle", (PyCFunction)wrap__m_geometry__private_method__binding__circle, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for private_method"},
    {"f90wrap_m_geometry__perimeter_4__binding__circle", (PyCFunction)wrap__m_geometry__perimeter_4__binding__circle, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for perimeter_4"},
    {"f90wrap_m_geometry__perimeter_8__binding__circle", (PyCFunction)wrap__m_geometry__perimeter_8__binding__circle, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for perimeter_8"},
    {"f90wrap_m_geometry__circle_free__binding__circle", (PyCFunction)wrap__m_geometry__circle_free__binding__circle, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for circle_free"},
    {"f90wrap_m_geometry__volume__binding__ball", (PyCFunction)wrap__m_geometry__volume__binding__ball, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for volume"},
    {"f90wrap_m_geometry__area__binding__ball", (PyCFunction)wrap__m_geometry__area__binding__ball, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for area"},
    {"f90wrap_m_geometry__private_method__binding__ball", (PyCFunction)wrap__m_geometry__private_method__binding__ball, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for private_method"},
    {"f90wrap_m_geometry__init__binding__list_circle", (PyCFunction)wrap__m_geometry__init__binding__list_circle, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for init"},
    {"f90wrap_m_geometry__init__binding__array", (PyCFunction)wrap__m_geometry__init__binding__array, METH_VARARGS | \
        METH_KEYWORDS, "Binding alias for init"},
    {"f90wrap_m_geometry__init_3d__binding__array_3d", (PyCFunction)wrap__m_geometry__init_3d__binding__array_3d, \
        METH_VARARGS | METH_KEYWORDS, "Binding alias for init_3d"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef _pywrappermodule = {
    PyModuleDef_HEAD_INIT,
    "pywrapper",
    "Direct-C wrapper for _pywrapper module",
    -1,
    _pywrapper_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__pywrapper(void)
{
    import_array();  /* Initialize NumPy */
    PyObject* module = PyModule_Create(&_pywrappermodule);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
