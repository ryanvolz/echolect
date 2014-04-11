#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

#cython: embedsignature=True

cimport cython
import numpy as np
cimport numpy as np

np.import_array() # or else we get segfaults when calling numpy C-api

ctypedef fused float_t:
    float
    double

ctypedef fused complex_t:
    float complex
    double complex

ctypedef fused single_t:
    float
    float complex

ctypedef fused double_t:
    double
    double complex

ctypedef fused htype:
    float
    double
    float complex
    double complex

ctypedef fused xtype:
    float
    double
    float complex
    double complex

ctypedef fused outtype:
   float
   double
   float complex
   double complex

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline _conv(htype[::1] h, xtype[::1] x, outtype[::1] out):
    cdef Py_ssize_t L = h.shape[0]
    cdef Py_ssize_t M = x.shape[0]
    cdef Py_ssize_t m, l
    cdef xtype x_m

    # avoid invalid type combinations to allow for compilation
    # these should never be used in practice if calling from conv
    if outtype in float_t and (htype in complex_t or xtype in complex_t):
        raise TypeError('out type must be compatible with the types of h and x')
    else:
        for m in range(M):
            x_m = x[m]
            for l in range(L):
                out[m + l] = out[m + l] + h[l]*x_m

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef conv(htype[::1] h, xtype[::1] x):
    cdef Py_ssize_t L = h.shape[0]
    cdef Py_ssize_t M = x.shape[0]
    cdef Py_ssize_t outlen = M + L - 1

    # ndarray with result that we will return
    cdef np.ndarray out
    # memory views we use for output memory, pointing out to this
    # can't declare variables inside if statement, so declare all here even though 3 go unused
    cdef cython.float[::1] out_float32
    cdef cython.double[::1] out_float64
    cdef cython.floatcomplex[::1] out_complex64
    cdef cython.doublecomplex[::1] out_complex128
    if (htype is cython.float and xtype is cython.float):
        out = np.PyArray_ZEROS(1, <np.npy_intp*>&outlen, np.NPY_FLOAT32, 0)
        out_float32 = out
        _conv(h, x, out_float32)
        return out
    elif (htype in float_t and xtype in float_t): # but not both single floats
        out = np.PyArray_ZEROS(1, <np.npy_intp*>&outlen, np.NPY_FLOAT64, 0)
        out_float64 = out
        _conv(h, x, out_float64)
        return out
    elif (htype in single_t and xtype in single_t): # but at least one is complex
        out = np.PyArray_ZEROS(1, <np.npy_intp*>&outlen, np.NPY_COMPLEX64, 0)
        out_complex64 = out
        _conv(h, x, out_complex64)
        return out
    else:
        out = np.PyArray_ZEROS(1, <np.npy_intp*>&outlen, np.NPY_COMPLEX128, 0)
        out_complex128 = out
        _conv(h, x, out_complex128)
        return out

def Conv(htype[::1] h, Py_ssize_t M, xdtype):
    cdef htype[::1] h2 = h # work around closure scope bug which doesn't include fused arguments
    
    if xdtype == np.float32:
        def cython_conv(cython.float[::1] x):
            return conv(h2, x)
    elif xdtype == np.float64:
        def cython_conv(cython.double[::1] x):
            return conv(h2, x)
    elif xdtype == np.complex64:
        def cython_conv(cython.floatcomplex[::1] x):
            return conv(h2, x)
    elif xdtype == np.complex128:
        def cython_conv(cython.doublecomplex[::1] x):
            return conv(h2, x)
    else:
        raise ValueError('xdtype must be single or double precision (complex) floating point')

    return cython_conv