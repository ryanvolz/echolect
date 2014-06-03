#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np
import unittest
import itertools
try:
    import numba
except ImportError:
    HAS_NUMBA = False
else:
    del numba

from echolect.filtering import filters

def get_random_uniform(shape, dtype):
    x = np.empty(shape, dtype)
    x.real = 2*np.random.rand(*shape) - 1
    if np.iscomplexobj(x):
        x.imag = 2*np.random.rand(*shape) - 1
    return x

def check_filters(L, M, hdtype, xdtype):
    h = get_random_uniform((L,), hdtype)
    x = get_random_uniform((M,), xdtype)

    # first in list is used for reference
    filts = [
        filters.Conv(h, M),
        filters.CythonConv(h, M, xdtype),
        filters.SparseConv(h, M),
        filters.StridedConv(h, M),
        filters.FFTPack(h, M, xdtype, powerof2=True),
        filters.FFTW(h, M, xdtype, powerof2=True),
        filters.NumpyFFT(h, M, xdtype, powerof2=True),
    ]
    if HAS_NUMBA:
        filts.extend([
            filters.NumbaConv(h, M, xdtype),
            filters.NumbaFFTW(h, M, xdtype, powerof2=True),
        ])

    err_msg = 'Result of filter "{0}" does not match filter "{1}"'

    reffilt = filts[0]
    y0 = reffilt(x)
    for filt in filts[1:]:
        y1 = filt(x)
        np.testing.assert_array_almost_equal(y0, y1, decimal=5,
                                             err_msg=err_msg.format(filt.func_name,
                                                                    reffilt.func_name))

def test_filters():
    Ls = (13, 50)
    Ms = (13, 50)
    hdtypes = (np.float64, np.complex64)
    xdtypes = (np.float64, np.complex64)

    for L, M, hdtype, xdtype in itertools.product(Ls, Ms, hdtypes, xdtypes):
        yield check_filters, L, M, hdtype, xdtype

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture','--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)
