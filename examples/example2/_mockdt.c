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
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef_initialise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef_finalise)(int* this);
extern void F90WRAP_F_SYMBOL(f90wrap_set_defaults)(int* solver);
extern void F90WRAP_F_SYMBOL(f90wrap_assign_constants)(void);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__get__ng)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__set__ng)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__get__ngpsi)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__set__ngpsi)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__array__ecinv)(int* dummy_this, int* nd, int* dtype, int* dshape, long \
    long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__array__xg)(int* dummy_this, int* nd, int* dtype, int* dshape, long long* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__array__fcinv)(int* dummy_this, int* nd, int* dtype, int* dshape, long \
    long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__array__wg)(int* dummy_this, int* nd, int* dtype, int* dshape, long long* \
    handle);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__array__xgpsi)(int* dummy_this, int* nd, int* dtype, int* dshape, long \
    long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_gaussian__array__wgpsi)(int* dummy_this, int* nd, int* dtype, int* dshape, long \
    long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__trimswitch)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__trimswitch)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__updateguess)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__updateguess)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__deltaaief83)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__deltaai9421)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__linrzswitch)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__linrzswitch)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__timemare3b3)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__timemar4f99)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__freewak3c80)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__freewak8069)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__windtund117)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__windtun0496)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__rigidbldcbe)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__rigidbl1493)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__fet_qddot)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__fet_qddot)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__fet_resc250)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__fet_res15d8)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__store_f538c)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__store_fb24a)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__fet_res230e)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__fet_resf178)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__airframevib)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__airframevib)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__fusharm)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__fusharm)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__axialdof)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__axialdof)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__composiee25)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__composic943)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__trimtecb616)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__trimtec7319)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__trimswe87ad)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__trimswe913a)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__ntimeel2005)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__ntimeelb6b3)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nbladeharm)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nbladeharm)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nblademodes)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nblademodes)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__array__modeorder)(int* dummy_this, int* nd, \
    int* dtype, int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__ncosinf3b81)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__ncosinf375e)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nmaxinf0dda)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nmaxinff41c)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__linflm)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__linflm)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__linrzpts)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__linrzpts)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__control2fe2)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__control0e2b)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nrevolu91d0)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nrevolu57c7)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nazim)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nazim)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__ntimesteps)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__ntimesteps)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nred)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nred)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nred2)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nred2)(int* handle, int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__trimcona726)(int* handle, double* \
    value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__trimcon7c3a)(int* handle, double* \
    value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__integerror)(int* handle, double* \
    value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__integerror)(int* handle, double* \
    value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__linrzpert)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__linrzpert)(int* handle, double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__control513b)(int* handle, double* \
    value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__controlfbaf)(int* handle, double* \
    value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__control3a84)(int* handle, double* \
    value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__control732d)(int* handle, double* \
    value);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__array__jac)(int* dummy_this, int* nd, int* \
    dtype, int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__array__jac2)(int* dummy_this, int* nd, int* \
    dtype, int* dshape, long long* handle);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__rdp)(int* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__zero)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__zero)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__one)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__one)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__half)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__half)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__two)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__two)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__three)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__three)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__four)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__four)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__six)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__six)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__eight)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__eight)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__pi)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__pi)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__twopi)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__twopi)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__d2r)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__d2r)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__r2d)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__r2d)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__xk2fps)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__xk2fps)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__lb2n)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__lb2n)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__ftlb2nm)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__ftlb2nm)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__one80)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__one80)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__ft2m)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__ft2m)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__gsi)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__gsi)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__gfps)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__gfps)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__three60)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__three60)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__in2ft)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__in2ft)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__get__five)(double* value);
extern void F90WRAP_F_SYMBOL(f90wrap_precision__set__five)(double* value);

static PyObject* wrap_defineallproperties_solveroptionsdef_initialise(PyObject* self, PyObject* args, PyObject* kwargs)
{
    int this[4] = {0};
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef_initialise)(this);
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

static PyObject* wrap_defineallproperties_solveroptionsdef_finalise(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef_finalise)(this);
    if (this_sequence) {
        Py_DECREF(this_sequence);
    }
    if (this_handle_obj) {
        Py_DECREF(this_handle_obj);
    }
    free(this);
    Py_RETURN_NONE;
}

static PyObject* wrap__mockdt_set_defaults(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* py_solver = NULL;
    static char *kwlist[] = {"solver", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &py_solver)) {
        return NULL;
    }
    
    PyObject* solver_handle_obj = NULL;
    PyObject* solver_sequence = NULL;
    Py_ssize_t solver_handle_len = 0;
    if (PyObject_HasAttrString(py_solver, "_handle")) {
        solver_handle_obj = PyObject_GetAttrString(py_solver, "_handle");
        if (solver_handle_obj == NULL) {
            return NULL;
        }
        solver_sequence = PySequence_Fast(solver_handle_obj, "Failed to access handle sequence");
        if (solver_sequence == NULL) {
            Py_DECREF(solver_handle_obj);
            return NULL;
        }
    } else if (PySequence_Check(py_solver)) {
        solver_sequence = PySequence_Fast(py_solver, "Argument solver must be a handle sequence");
        if (solver_sequence == NULL) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Argument solver must be a Fortran derived-type instance");
        return NULL;
    }
    solver_handle_len = PySequence_Fast_GET_SIZE(solver_sequence);
    if (solver_handle_len != 4) {
        PyErr_SetString(PyExc_ValueError, "Argument solver has an invalid handle length");
        Py_DECREF(solver_sequence);
        if (solver_handle_obj) Py_DECREF(solver_handle_obj);
        return NULL;
    }
    int* solver = (int*)malloc(sizeof(int) * solver_handle_len);
    if (solver == NULL) {
        PyErr_NoMemory();
        Py_DECREF(solver_sequence);
        if (solver_handle_obj) Py_DECREF(solver_handle_obj);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < solver_handle_len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(solver_sequence, i);
        if (item == NULL) {
            free(solver);
            Py_DECREF(solver_sequence);
            if (solver_handle_obj) Py_DECREF(solver_handle_obj);
            return NULL;
        }
        solver[i] = (int)PyLong_AsLong(item);
        if (PyErr_Occurred()) {
            free(solver);
            Py_DECREF(solver_sequence);
            if (solver_handle_obj) Py_DECREF(solver_handle_obj);
            return NULL;
        }
    }
    (void)solver_handle_len;  /* suppress unused warnings when unchanged */
    
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_set_defaults)(solver);
    if (PyErr_Occurred()) {
        if (solver_sequence) Py_DECREF(solver_sequence);
        if (solver_handle_obj) Py_DECREF(solver_handle_obj);
        free(solver);
        return NULL;
    }
    
    if (solver_sequence) {
        Py_DECREF(solver_sequence);
    }
    if (solver_handle_obj) {
        Py_DECREF(solver_handle_obj);
    }
    free(solver);
    Py_RETURN_NONE;
}

static PyObject* wrap__mockdt_assign_constants(PyObject* self, PyObject* args, PyObject* kwargs)
{
    /* Call f90wrap helper */
    F90WRAP_F_SYMBOL(f90wrap_assign_constants)();
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* wrap_gaussian_helper_get_ng(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_gaussian__get__ng)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_gaussian_helper_set_ng(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"ng", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_gaussian__set__ng)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_gaussian_helper_get_ngpsi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_gaussian__get__ngpsi)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_gaussian_helper_set_ngpsi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    int value;
    static char *kwlist[] = {"ngpsi", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_gaussian__set__ngpsi)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_gaussian_helper_array_ecinv(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_gaussian__array__ecinv)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_gaussian_helper_array_xg(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_gaussian__array__xg)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_gaussian_helper_array_fcinv(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_gaussian__array__fcinv)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_gaussian_helper_array_wg(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_gaussian__array__wg)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_gaussian_helper_array_xgpsi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_gaussian__array__xgpsi)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_gaussian_helper_array_wgpsi(PyObject* self, PyObject* args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_gaussian__array__wgpsi)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_trimswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__trimswitch)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_trimswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "trimswitch", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__trimswitch)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_updateguess(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__updateguess)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_updateguess(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "updateguess", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__updateguess)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_deltaairloads(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__deltaaief83)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_deltaairloads(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "deltaairloads", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__deltaai9421)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_linrzswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__linrzswitch)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_linrzswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "linrzswitch", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__linrzswitch)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_timemarchswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__timemare3b3)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_timemarchswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "timemarchswitch", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__timemar4f99)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_freewakeswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__freewak3c80)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_freewakeswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "freewakeswitch", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__freewak8069)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_windtunnelswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__windtund117)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_windtunnelswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "windtunnelswitch", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__windtun0496)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_rigidbladeswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__rigidbldcbe)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_rigidbladeswitch(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "rigidbladeswitch", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__rigidbl1493)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_fet_qddot(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__fet_qddot)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_fet_qddot(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "fet_qddot", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__fet_qddot)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_fet_response(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__fet_resc250)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_fet_response(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "fet_response", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__fet_res15d8)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_store_fet_responsejac(PyObject* self, PyObject* \
    args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__store_f538c)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_store_fet_responsejac(PyObject* self, PyObject* \
    args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "store_fet_responsejac", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__store_fb24a)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_fet_responsejacavail(PyObject* self, PyObject* \
    args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__fet_res230e)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_fet_responsejacavail(PyObject* self, PyObject* \
    args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "fet_responsejacavail", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__fet_resf178)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_airframevib(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__airframevib)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_airframevib(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "airframevib", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__airframevib)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_fusharm(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__fusharm)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_fusharm(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "fusharm", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__fusharm)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_axialdof(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__axialdof)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_axialdof(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "axialdof", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__axialdof)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_composite_coupling(PyObject* self, PyObject* \
    args, PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__composiee25)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyBool_FromLong(value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_composite_coupling(PyObject* self, PyObject* \
    args, PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "composite_coupling", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__composic943)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_trimtechnique(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__trimtecb616)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_trimtechnique(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "trimtechnique", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__trimtec7319)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_trimsweepoption(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__trimswe87ad)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_trimsweepoption(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "trimsweepoption", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__trimswe913a)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_ntimeelements(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__ntimeel2005)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_ntimeelements(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "ntimeelements", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__ntimeelb6b3)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_nbladeharm(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nbladeharm)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_nbladeharm(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "nbladeharm", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nbladeharm)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_nblademodes(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nblademodes)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_nblademodes(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "nblademodes", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nblademodes)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_array_modeorder(PyObject* self, PyObject* args, \
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__array__modeorder)(dummy_this, &nd, &dtype, dshape, \
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

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_ncosinflowharm(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__ncosinf3b81)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_ncosinflowharm(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "ncosinflowharm", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__ncosinf375e)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_nmaxinflowpoly(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nmaxinf0dda)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_nmaxinflowpoly(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "nmaxinflowpoly", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nmaxinff41c)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_linflm(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__linflm)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_linflm(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "linflm", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__linflm)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_linrzpts(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__linrzpts)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_linrzpts(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "linrzpts", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__linrzpts)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_controlhistoption(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__control2fe2)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_controlhistoption(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "controlhistoption", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__control0e2b)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_nrevolutions(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nrevolu91d0)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_nrevolutions(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "nrevolutions", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nrevolu57c7)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_nazim(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nazim)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_nazim(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "nazim", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nazim)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_ntimesteps(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__ntimesteps)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_ntimesteps(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "ntimesteps", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__ntimesteps)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_nred(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nred)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_nred(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "nred", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nred)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_nred2(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__nred2)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("i", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_nred2(PyObject* self, PyObject* args, PyObject* \
    kwargs)
{
    (void)self;
    PyObject* py_handle;
    int value;
    static char *kwlist[] = {"handle", "nred2", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__nred2)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_trimconvergence(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__trimcona726)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_trimconvergence(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "trimconvergence", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__trimcon7c3a)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_integerror(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__integerror)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_integerror(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "integerror", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__integerror)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_linrzpert(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__linrzpert)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_linrzpert(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "linrzpert", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__linrzpert)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_controlamplitude(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__control513b)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_controlamplitude(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "controlamplitude", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__controlfbaf)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_get_controlfrequency(PyObject* self, PyObject* args, \
    PyObject* kwargs)
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__get__control3a84)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("d", value);
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_set_controlfrequency(PyObject* self, PyObject* args, \
    PyObject* kwargs)
{
    (void)self;
    PyObject* py_handle;
    double value;
    static char *kwlist[] = {"handle", "controlfrequency", NULL};
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__set__control732d)(this_handle, &value);
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_array_jac(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__array__jac)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_defineallproperties__solveroptionsdef_helper_array_jac2(PyObject* self, PyObject* args, PyObject* \
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
    F90WRAP_F_SYMBOL(f90wrap_defineallproperties__solveroptionsdef__array__jac2)(dummy_this, &nd, &dtype, dshape, &handle);
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

static PyObject* wrap_precision_helper_get_rdp(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    int value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__rdp)(&value);
    return Py_BuildValue("i", value);
}

static PyObject* wrap_precision_helper_get_zero(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__zero)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_zero(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"zero", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__zero)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_one(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__one)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_one(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"one", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__one)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_half(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__half)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_half(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"half", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__half)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_two(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__two)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_two(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"two", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__two)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_three(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__three)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_three(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"three", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__three)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_four(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__four)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_four(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"four", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__four)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_six(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__six)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_six(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"six", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__six)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_eight(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__eight)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_eight(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"eight", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__eight)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_pi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__pi)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_pi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"pi", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__pi)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_twopi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__twopi)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_twopi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"twopi", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__twopi)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_d2r(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__d2r)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_d2r(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"d2r", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__d2r)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_r2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__r2d)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_r2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"r2d", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__r2d)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_xk2fps(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__xk2fps)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_xk2fps(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"xk2fps", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__xk2fps)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_lb2n(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__lb2n)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_lb2n(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"lb2n", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__lb2n)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_ftlb2nm(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__ftlb2nm)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_ftlb2nm(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"ftlb2nm", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__ftlb2nm)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_one80(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__one80)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_one80(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"one80", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__one80)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_ft2m(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__ft2m)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_ft2m(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"ft2m", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__ft2m)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_gsi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__gsi)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_gsi(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"gsi", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__gsi)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_gfps(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__gfps)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_gfps(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"gfps", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__gfps)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_three60(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__three60)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_three60(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"three60", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__three60)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_in2ft(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__in2ft)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_in2ft(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"in2ft", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__in2ft)(&value);
    Py_RETURN_NONE;
}

static PyObject* wrap_precision_helper_get_five(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    if ((args && PyTuple_Size(args) != 0) || (kwargs && PyDict_Size(kwargs) != 0)) {
        PyErr_SetString(PyExc_TypeError, "This helper does not accept arguments");
        return NULL;
    }
    
    double value;
    F90WRAP_F_SYMBOL(f90wrap_precision__get__five)(&value);
    return Py_BuildValue("d", value);
}

static PyObject* wrap_precision_helper_set_five(PyObject* self, PyObject* args, PyObject* kwargs)
{
    (void)self;
    double value;
    static char *kwlist[] = {"five", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d", kwlist, &value)) {
        return NULL;
    }
    F90WRAP_F_SYMBOL(f90wrap_precision__set__five)(&value);
    Py_RETURN_NONE;
}

/* Method table for _mockdt module */
static PyMethodDef _mockdt_methods[] = {
    {"f90wrap_defineallproperties__solveroptionsdef_initialise", \
        (PyCFunction)wrap_defineallproperties_solveroptionsdef_initialise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated constructor for solveroptionsdef"},
    {"f90wrap_defineallproperties__solveroptionsdef_finalise", \
        (PyCFunction)wrap_defineallproperties_solveroptionsdef_finalise, METH_VARARGS | METH_KEYWORDS, "Automatically \
        generated destructor for solveroptionsdef"},
    {"f90wrap_set_defaults", (PyCFunction)wrap__mockdt_set_defaults, METH_VARARGS | METH_KEYWORDS, \
        "======================================================================="},
    {"f90wrap_assign_constants", (PyCFunction)wrap__mockdt_assign_constants, METH_VARARGS | METH_KEYWORDS, \
        "======================================================================="},
    {"f90wrap_gaussian__get__ng", (PyCFunction)wrap_gaussian_helper_get_ng, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        ng"},
    {"f90wrap_gaussian__set__ng", (PyCFunction)wrap_gaussian_helper_set_ng, METH_VARARGS | METH_KEYWORDS, "Module helper for \
        ng"},
    {"f90wrap_gaussian__get__ngpsi", (PyCFunction)wrap_gaussian_helper_get_ngpsi, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ngpsi"},
    {"f90wrap_gaussian__set__ngpsi", (PyCFunction)wrap_gaussian_helper_set_ngpsi, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ngpsi"},
    {"f90wrap_gaussian__array__ecinv", (PyCFunction)wrap_gaussian_helper_array_ecinv, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for ecinv"},
    {"f90wrap_gaussian__array__xg", (PyCFunction)wrap_gaussian_helper_array_xg, METH_VARARGS | METH_KEYWORDS, "Array helper \
        for xg"},
    {"f90wrap_gaussian__array__fcinv", (PyCFunction)wrap_gaussian_helper_array_fcinv, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for fcinv"},
    {"f90wrap_gaussian__array__wg", (PyCFunction)wrap_gaussian_helper_array_wg, METH_VARARGS | METH_KEYWORDS, "Array helper \
        for wg"},
    {"f90wrap_gaussian__array__xgpsi", (PyCFunction)wrap_gaussian_helper_array_xgpsi, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for xgpsi"},
    {"f90wrap_gaussian__array__wgpsi", (PyCFunction)wrap_gaussian_helper_array_wgpsi, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for wgpsi"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__trimswitch", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_trimswitch, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for trimswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__trimswitch", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_trimswitch, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for trimswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__updateguess", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_updateguess, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for updateguess"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__updateguess", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_updateguess, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for updateguess"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__deltaaief83", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_deltaairloads, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for deltaairloads"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__deltaai9421", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_deltaairloads, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for deltaairloads"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__linrzswitch", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_linrzswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for linrzswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__linrzswitch", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_linrzswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for linrzswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__timemare3b3", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_timemarchswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for timemarchswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__timemar4f99", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_timemarchswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for timemarchswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__freewak3c80", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_freewakeswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for freewakeswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__freewak8069", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_freewakeswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for freewakeswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__windtund117", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_windtunnelswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for windtunnelswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__windtun0496", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_windtunnelswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for windtunnelswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__rigidbldcbe", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_rigidbladeswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for rigidbladeswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__rigidbl1493", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_rigidbladeswitch, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for rigidbladeswitch"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__fet_qddot", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_fet_qddot, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for fet_qddot"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__fet_qddot", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_fet_qddot, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for fet_qddot"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__fet_resc250", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_fet_response, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for fet_response"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__fet_res15d8", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_fet_response, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for fet_response"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__store_f538c", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_store_fet_responsejac, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for store_fet_responsejac"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__store_fb24a", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_store_fet_responsejac, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for store_fet_responsejac"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__fet_res230e", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_fet_responsejacavail, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for fet_responsejacavail"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__fet_resf178", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_fet_responsejacavail, METH_VARARGS | \
        METH_KEYWORDS, "Module helper for fet_responsejacavail"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__airframevib", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_airframevib, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for airframevib"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__airframevib", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_airframevib, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for airframevib"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__fusharm", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_fusharm, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for fusharm"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__fusharm", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_fusharm, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for fusharm"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__axialdof", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_axialdof, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for axialdof"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__axialdof", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_axialdof, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for axialdof"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__composiee25", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_composite_coupling, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for composite_coupling"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__composic943", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_composite_coupling, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for composite_coupling"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__trimtecb616", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_trimtechnique, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for trimtechnique"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__trimtec7319", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_trimtechnique, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for trimtechnique"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__trimswe87ad", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_trimsweepoption, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for trimsweepoption"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__trimswe913a", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_trimsweepoption, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for trimsweepoption"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__ntimeel2005", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_ntimeelements, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for ntimeelements"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__ntimeelb6b3", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_ntimeelements, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for ntimeelements"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__nbladeharm", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_nbladeharm, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for nbladeharm"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__nbladeharm", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_nbladeharm, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for nbladeharm"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__nblademodes", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_nblademodes, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for nblademodes"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__nblademodes", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_nblademodes, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for nblademodes"},
    {"f90wrap_defineallproperties__solveroptionsdef__array__modeorder", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_array_modeorder, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for modeorder"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__ncosinf3b81", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_ncosinflowharm, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for ncosinflowharm"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__ncosinf375e", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_ncosinflowharm, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for ncosinflowharm"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__nmaxinf0dda", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_nmaxinflowpoly, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for nmaxinflowpoly"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__nmaxinff41c", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_nmaxinflowpoly, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for nmaxinflowpoly"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__linflm", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_linflm, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for linflm"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__linflm", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_linflm, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for linflm"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__linrzpts", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_linrzpts, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for linrzpts"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__linrzpts", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_linrzpts, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for linrzpts"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__control2fe2", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_controlhistoption, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for controlhistoption"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__control0e2b", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_controlhistoption, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for controlhistoption"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__nrevolu91d0", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_nrevolutions, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for nrevolutions"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__nrevolu57c7", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_nrevolutions, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for nrevolutions"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__nazim", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_nazim, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for nazim"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__nazim", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_nazim, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for nazim"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__ntimesteps", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_ntimesteps, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ntimesteps"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__ntimesteps", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_ntimesteps, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ntimesteps"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__nred", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_nred, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for nred"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__nred", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_nred, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for nred"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__nred2", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_nred2, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for nred2"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__nred2", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_nred2, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for nred2"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__trimcona726", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_trimconvergence, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for trimconvergence"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__trimcon7c3a", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_trimconvergence, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for trimconvergence"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__integerror", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_integerror, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for integerror"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__integerror", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_integerror, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for integerror"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__linrzpert", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_linrzpert, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for linrzpert"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__linrzpert", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_linrzpert, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for linrzpert"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__control513b", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_controlamplitude, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for controlamplitude"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__controlfbaf", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_controlamplitude, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for controlamplitude"},
    {"f90wrap_defineallproperties__solveroptionsdef__get__control3a84", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_get_controlfrequency, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for controlfrequency"},
    {"f90wrap_defineallproperties__solveroptionsdef__set__control732d", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_set_controlfrequency, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for controlfrequency"},
    {"f90wrap_defineallproperties__solveroptionsdef__array__jac", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_array_jac, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for jac"},
    {"f90wrap_defineallproperties__solveroptionsdef__array__jac2", \
        (PyCFunction)wrap_defineallproperties__solveroptionsdef_helper_array_jac2, METH_VARARGS | METH_KEYWORDS, "Array \
        helper for jac2"},
    {"f90wrap_precision__get__rdp", (PyCFunction)wrap_precision_helper_get_rdp, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for rdp"},
    {"f90wrap_precision__get__zero", (PyCFunction)wrap_precision_helper_get_zero, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for zero"},
    {"f90wrap_precision__set__zero", (PyCFunction)wrap_precision_helper_set_zero, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for zero"},
    {"f90wrap_precision__get__one", (PyCFunction)wrap_precision_helper_get_one, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for one"},
    {"f90wrap_precision__set__one", (PyCFunction)wrap_precision_helper_set_one, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for one"},
    {"f90wrap_precision__get__half", (PyCFunction)wrap_precision_helper_get_half, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for half"},
    {"f90wrap_precision__set__half", (PyCFunction)wrap_precision_helper_set_half, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for half"},
    {"f90wrap_precision__get__two", (PyCFunction)wrap_precision_helper_get_two, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for two"},
    {"f90wrap_precision__set__two", (PyCFunction)wrap_precision_helper_set_two, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for two"},
    {"f90wrap_precision__get__three", (PyCFunction)wrap_precision_helper_get_three, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for three"},
    {"f90wrap_precision__set__three", (PyCFunction)wrap_precision_helper_set_three, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for three"},
    {"f90wrap_precision__get__four", (PyCFunction)wrap_precision_helper_get_four, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for four"},
    {"f90wrap_precision__set__four", (PyCFunction)wrap_precision_helper_set_four, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for four"},
    {"f90wrap_precision__get__six", (PyCFunction)wrap_precision_helper_get_six, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for six"},
    {"f90wrap_precision__set__six", (PyCFunction)wrap_precision_helper_set_six, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for six"},
    {"f90wrap_precision__get__eight", (PyCFunction)wrap_precision_helper_get_eight, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for eight"},
    {"f90wrap_precision__set__eight", (PyCFunction)wrap_precision_helper_set_eight, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for eight"},
    {"f90wrap_precision__get__pi", (PyCFunction)wrap_precision_helper_get_pi, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for pi"},
    {"f90wrap_precision__set__pi", (PyCFunction)wrap_precision_helper_set_pi, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for pi"},
    {"f90wrap_precision__get__twopi", (PyCFunction)wrap_precision_helper_get_twopi, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for twopi"},
    {"f90wrap_precision__set__twopi", (PyCFunction)wrap_precision_helper_set_twopi, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for twopi"},
    {"f90wrap_precision__get__d2r", (PyCFunction)wrap_precision_helper_get_d2r, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for d2r"},
    {"f90wrap_precision__set__d2r", (PyCFunction)wrap_precision_helper_set_d2r, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for d2r"},
    {"f90wrap_precision__get__r2d", (PyCFunction)wrap_precision_helper_get_r2d, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for r2d"},
    {"f90wrap_precision__set__r2d", (PyCFunction)wrap_precision_helper_set_r2d, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for r2d"},
    {"f90wrap_precision__get__xk2fps", (PyCFunction)wrap_precision_helper_get_xk2fps, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for xk2fps"},
    {"f90wrap_precision__set__xk2fps", (PyCFunction)wrap_precision_helper_set_xk2fps, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for xk2fps"},
    {"f90wrap_precision__get__lb2n", (PyCFunction)wrap_precision_helper_get_lb2n, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for lb2n"},
    {"f90wrap_precision__set__lb2n", (PyCFunction)wrap_precision_helper_set_lb2n, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for lb2n"},
    {"f90wrap_precision__get__ftlb2nm", (PyCFunction)wrap_precision_helper_get_ftlb2nm, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for ftlb2nm"},
    {"f90wrap_precision__set__ftlb2nm", (PyCFunction)wrap_precision_helper_set_ftlb2nm, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for ftlb2nm"},
    {"f90wrap_precision__get__one80", (PyCFunction)wrap_precision_helper_get_one80, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for one80"},
    {"f90wrap_precision__set__one80", (PyCFunction)wrap_precision_helper_set_one80, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for one80"},
    {"f90wrap_precision__get__ft2m", (PyCFunction)wrap_precision_helper_get_ft2m, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ft2m"},
    {"f90wrap_precision__set__ft2m", (PyCFunction)wrap_precision_helper_set_ft2m, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for ft2m"},
    {"f90wrap_precision__get__gsi", (PyCFunction)wrap_precision_helper_get_gsi, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for gsi"},
    {"f90wrap_precision__set__gsi", (PyCFunction)wrap_precision_helper_set_gsi, METH_VARARGS | METH_KEYWORDS, "Module helper \
        for gsi"},
    {"f90wrap_precision__get__gfps", (PyCFunction)wrap_precision_helper_get_gfps, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for gfps"},
    {"f90wrap_precision__set__gfps", (PyCFunction)wrap_precision_helper_set_gfps, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for gfps"},
    {"f90wrap_precision__get__three60", (PyCFunction)wrap_precision_helper_get_three60, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for three60"},
    {"f90wrap_precision__set__three60", (PyCFunction)wrap_precision_helper_set_three60, METH_VARARGS | METH_KEYWORDS, \
        "Module helper for three60"},
    {"f90wrap_precision__get__in2ft", (PyCFunction)wrap_precision_helper_get_in2ft, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for in2ft"},
    {"f90wrap_precision__set__in2ft", (PyCFunction)wrap_precision_helper_set_in2ft, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for in2ft"},
    {"f90wrap_precision__get__five", (PyCFunction)wrap_precision_helper_get_five, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for five"},
    {"f90wrap_precision__set__five", (PyCFunction)wrap_precision_helper_set_five, METH_VARARGS | METH_KEYWORDS, "Module \
        helper for five"},
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
