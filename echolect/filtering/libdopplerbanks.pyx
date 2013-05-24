# Copyright 2013 Ryan Volz

# This file is part of echolect.

# Echolect is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Echolect is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with echolect.  If not, see <http://www.gnu.org/licenses/>.

#cython: embedsignature=True

from __future__ import division
cimport cython
from cython.parallel import prange
from cython cimport view
import numpy as np
cimport numpy as np
cimport pyfftw

np.import_array() # or else we get segfaults when calling numpy C-api

ctypedef fused htype:
    cython.float
    cython.double
    cython.floatcomplex
    cython.doublecomplex

ctypedef fused xtype:
    cython.float
    cython.double
    cython.floatcomplex
    cython.doublecomplex

ctypedef fused ytype:
    cython.floatcomplex
    cython.doublecomplex

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef sweepspectra(htype[::1] hrev, ytype[:, ::1] demodpad, ytype[:, ::1] y_aligned, 
                  pyfftw.FFTW fft, Py_ssize_t step, Py_ssize_t N, Py_ssize_t M, 
                  xtype[::1] x):
    # implements Doppler filter:
    # y[n, p] = SUM_k exp(-2*pi*j*n*(k - (L-1))/N) * (h[k] * x[p - k])
    #         = SUM_k exp(-2*pi*j*n*k/N) * (hrev[k] * x[p - (L-1) + k])
    cdef Py_ssize_t L = hrev.shape[0]
    cdef Py_ssize_t outlen = demodpad.shape[0]
    cdef Py_ssize_t p, p0, k, kstart, kstop

    cdef np.ndarray y_ndarray
    cdef ytype[:, ::view.contiguous] y
    cdef np.npy_intp *yshape = [N, outlen]
    if ytype is cython.floatcomplex:
        # we set every entry, so empty is ok
        y_ndarray = np.PyArray_EMPTY(2, yshape, np.NPY_COMPLEX64, 0)
    elif ytype is cython.doublecomplex:
        # we set every entry, so empty is ok
        y_ndarray = np.PyArray_EMPTY(2, yshape, np.NPY_COMPLEX128, 0)
    y = y_ndarray

    # np.multiply(xshifted, hrev, demodpad[:, :L]) :
    for p in prange(outlen, nogil=True):
        # constraints on k from bounds of x:
        # p + (L-1) + k >= 0:
        #       k >= (L-1) - p
        # p + (L-1) + k <= M-1:
        #       k <= (M-1) + (L-1) - p < M + (L-1) - p
        p0 = p - (L - 1)
        kstart = max(0, -p0)
        kstop = min(L, M - p0)
        for k in range(kstart, kstop):
            demodpad[p, k] = hrev[k]*x[p0 + k]

    fft.execute() # input is demodpad, output is y_aligned
    y[:, :] = y_aligned.T[::step, :]

    return y_ndarray

def SweepSpectraCython(htype[::1] h, ytype[:, ::1] demodpad, ytype[:, ::1] y_aligned, 
                       pyfftw.FFTW fft, Py_ssize_t step, Py_ssize_t N, Py_ssize_t M,
                       xdtype):
    cdef ytype[:, ::1] demodpad2 = demodpad # work around closure scope bug which doesn't include fused arguments
    cdef ytype[:, ::1] y_aligned2 = y_aligned # work around closure scope bug which doesn't include fused arguments

    hrev_python = h[::-1].copy()
    cdef htype[::1] hrev = hrev_python

    if xdtype == np.float32:
        def sweepspectra_cython(cython.float[::1] x):
            return sweepspectra(hrev, demodpad2, y_aligned2, fft, step, N, M, x)
    elif xdtype is np.float64:
        def sweepspectra_cython(cython.double[::1] x):
            return sweepspectra(hrev, demodpad2, y_aligned2, fft, step, N, M, x)
    elif xdtype is np.complex64:
        def sweepspectra_cython(cython.floatcomplex[::1] x):
            return sweepspectra(hrev, demodpad2, y_aligned2, fft, step, N, M, x)
    elif xdtype is np.complex128:
        def sweepspectra_cython(cython.doublecomplex[::1] x):
            return sweepspectra(hrev, demodpad2, y_aligned2, fft, step, N, M, x)

    return sweepspectra_cython