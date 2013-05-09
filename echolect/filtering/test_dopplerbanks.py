import numpy as np
import unittest
import itertools

import dopplerbanks

def get_random_uniform(shape, dtype):
    x = np.empty(shape, dtype)
    x.real = 2*np.random.rand(*shape) - 1
    if np.iscomplexobj(x):
        x.imag = 2*np.random.rand(*shape) - 1
    return x

def check_doppler_banks(L, M, N, hdtype, xdtype):
    h = get_random_uniform((L,), hdtype)
    x = get_random_uniform((M,), xdtype)
    
    # first in list is used for reference
    filts = [dopplerbanks.ShiftConv(h, N, M),
             dopplerbanks.ShiftConvFFT(h, N, M, xdtype, powerof2=True),
             dopplerbanks.ShiftConvNumbaFFT(h, N, M, xdtype, powerof2=True),
             dopplerbanks.ShiftConvSparse(h, N, M),
             dopplerbanks.SweepSpectraCython(h, N, M, xdtype),
             dopplerbanks.SweepSpectraNumba(h, N, M, xdtype),
             dopplerbanks.SweepSpectraStridedInput(h, N, M, xdtype),
             dopplerbanks.SweepSpectraStridedTaps(h, N, M, xdtype)]
    
    err_msg = 'Result of filter "{0}" does not match filter "{1}"'
    
    reffilt = filts[0]
    y0 = reffilt(x)
    for filt in filts[1:]:
        y1 = filt(x)
        np.testing.assert_array_almost_equal(y0, y1, err_msg=err_msg.format(filt.func_name, reffilt.func_name))

def test_doppler_banks():
    Ls = (16, 16, 16, 10, 10)
    Ms = (16, 10, 50, 64, 64)
    Ns = (16, 16, 16, 64, 128)
    hdtypes = (np.float64, np.complex64)
    xdtypes = (np.float64, np.complex64)
    
    for (L, M, N), hdtype, xdtype in itertools.product(zip(Ls, Ms, Ns), hdtypes, xdtypes):
        yield check_doppler_banks, L, M, N, hdtype, xdtype

if __name__ == '__main__':
    import nose
    #nose.runmodule(argv=[__file__,'-vvs','--nologcapture','--stop','--pdb','--pdb-failure'],
                   #exit=False)
    nose.runmodule(argv=[__file__,'-vvs','--nologcapture'],
                   exit=False)