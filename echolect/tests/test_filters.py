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

import numpy as np
import unittest
import itertools

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
    filts = [filters.Conv(h, M),
             filters.CythonConv(h, M, xdtype),
             filters.NumbaConv(h, M, xdtype),
             filters.SparseConv(h, M),
             filters.StridedConv(h, M),
             filters.FFTPack(h, M, xdtype, powerof2=True),
             filters.FFTW(h, M, xdtype, powerof2=True),
             filters.NumbaFFTW(h, M, xdtype, powerof2=True),
             filters.NumpyFFT(h, M, xdtype, powerof2=True)]
    
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