import numpy as np

import filters
import dopplerbanks
from util import apply_to_2d, convslice, downsample, pow2, time_filters, upsample, zero_pad

# ******** Filter functions ********

def measure_filters(h, M, xdtype=np.complex_, number=100, disp=True, meas_all=True):
    flist = [filters.CythonConv(h, M, xdtype),
             filters.NumbaConv(h, M, xdtype),
             filters.NumbaFFTW(h, M, xdtype, powerof2=True)]
    if meas_all:
        flist.extend([filters.Conv(h, M),
                      filters.StridedConv(h, M),
                      filters.SparseConv(h, M),
                      filters.FFTPack(h, M, xdtype, powerof2=True),
                      filters.FFTW(h, M, xdtype, powerof2=True),
                      filters.NumpyFFT(h, M, xdtype, powerof2=True)])
    x = np.empty(M, xdtype)
    x.real = 2*np.random.rand(M) - 1
    if np.iscomplexobj(x):
        x.imag = 2*np.random.rand(M) - 1
    times = time_filters(flist, x, number)

    # sort in order of times
    tups = zip(times, flist)
    tups.sort()

    if disp:
        for time, filt in tups:
            print(filt.func_name + ': {0} s per call'.format(time/number))

    times, flist = zip(*tups)
    return times, flist

def Filter(h, M, xdtype=np.complex_, measure=True):
    if measure is True:
        times, flist = measure_filters(h, M, xdtype, number=10, disp=False, meas_all=False)
        filt = flist[np.argmin(times)]
    else:
        filt = filters.FFTW(h, M, xdtype, powerof2=True)

    return filt

def filter(h, x):
    xshape = x.shape
    filt = Filter(h, xshape[-1], measure=False)
    if len(xshape) > 1:
        return apply_to_2d(filt, x)
    return filt(x)

def doppler_coefs(h, f):
    """Doppler shift the given filter h to match normalized frequency f.

    The Doppler-shifted filter applied to a signal x will give the same result
    (except for a constant phase shift) as the original filter applied to a
    Doppler-shifted x, where x has been modulated by the complex exponential
    with normalized frequency f.

    The result of using this filter will also be equivalent to the k'th filter
    of a N-filter Doppler bank if f == k/N.

    Parameters
    ----------
    h : 1-D ndarray
        Coefficients of the filter to be Doppler-shifted.

    f : float, typically in [-0.5, 0.5]
        Normalized frequency (true frequency = f/T Hz) of Doppler shift.

    Returns
    -------
    hd : 1-D ndarray
        Coefficients of the Doppler-shifted filter.

    """
    L = len(h)
    hd = h*np.exp(-2*np.pi*1j*np.arange(L)*f)[::-1]

    return hd


# ******** Doppler bank functions ********

def measure_doppler_banks(h, N, M, xdtype=np.complex_, number=100, disp=True, meas_all=True):
    flist = [dopplerbanks.ShiftConvFFT(h, N, M, xdtype, powerof2=True),
             dopplerbanks.SweepSpectraCython(h, N, M, xdtype),
             dopplerbanks.SweepSpectraNumba(h, N, M, xdtype),
             dopplerbanks.SweepSpectraStridedInput(h, N, M, xdtype),
             ]
    if meas_all:
        flist.extend([dopplerbanks.ShiftConv(h, N, M),
                      dopplerbanks.ShiftConvNumbaFFT(h, N, M, xdtype, powerof2=True),
                      dopplerbanks.ShiftConvSparse(h, N, M),
                      dopplerbanks.ShiftConvSparseMod(h, N, M),
                      dopplerbanks.SweepSpectraStridedTaps(h, N, M),
                      dopplerbanks.SweepSpectraStridedTapsMod(h, N, M)])
    x = np.empty(M, xdtype)
    x.real = 2*np.random.rand(M) - 1
    if np.iscomplexobj(x):
        x.imag = 2*np.random.rand(M) - 1
    times = time_filters(flist, x, number)

    # sort in order of times
    tups = zip(times, flist)
    tups.sort()

    if disp:
        for time, filt in tups:
            print(filt.func_name + ': {0} s per call'.format(time/number))
    
    times, flist = zip(*tups)
    return times, flist

def DopplerBank(h, N, M, xdtype=np.complex_, measure=True):
    if measure is True:
        times, flist = measure_doppler_banks(h, N, M, xdtype, number=10, disp=False, meas_all=False)
        bank = flist[np.argmin(times)]
    else:
        bank = dopplerbanks.SweepSpectraStridedInput(h, N, M, xdtype)

    return bank

def DopplerBankMax(h, N, M, xdtype=np.complex_, measure=True):
    bank = DopplerBank(h, N, M, xdtype, measure)

    def doppler_bank_max(x):
        """Apply a Doppler filter bank to the input, selecting frequency of
        maximum response.

        Parameters
        ----------
        x : 1-D ndarray
            Values to be filtered.

        Returns
        -------
        y : 1-D ndarray
            Filtered values for frequency with maximum response.

        f : float
            Normalized frequency (true frequency = f/T Hz) of maximum response

        """
        y = bank(x)
        shift = np.unravel_index(np.argmax(y.real**2 + y.imag**2), y.shape)[0]

        f = float(shift)/N
        return y[shift], f
    
    doppler_bank_max.__dict__.update(bank.__dict__)
    doppler_bank_max.bank = bank

    return doppler_bank_max

def doppler_bank(h, N, x):
    xshape = x.shape
    filt = DopplerBank(h, N, xshape[-1], x.dtype, measure=False)
    if len(xshape) > 1:
        return apply_to_2d(filt, x)
    return filt(x)

def doppler_bank_max(h, N, x):
    xshape = x.shape
    filt = DopplerBankMax(h, N, xshape[-1], x.dtype, measure=False)
    if len(xshape) > 1:
        return apply_to_2d(filt, x)
    return filt(x)


# ******** Matched filter functions ********

def matched_coefs(s):
    return s.conj()[::-1]

def Matched(s, M, xdtype=np.complex_, measure=True):
    h = matched_coefs(s)
    return Filter(h, M, xdtype, measure)

def MatchedDoppler(s, N, M, xdtype=np.complex_, measure=True):
    h = matched_coefs(s)
    return DopplerBank(h, N, M, xdtype, measure)

def MatchedDopplerMax(s, N, M, xdtype=np.complex_, measure=True):
    h = matched_coefs(s)
    return DopplerBankMax(h, N, M, xdtype, measure)

def matched(s, x):
    xshape = x.shape
    filt = Matched(s, xshape[-1], x.dtype, measure=False)
    if len(xshape) > 1:
        return apply_to_2d(filt, x)
    return filt(x)

def matched_doppler(s, N, x):
    xshape = x.shape
    filt = MatchedDoppler(s, N, xshape[-1], x.dtype, measure=False)
    if len(xshape) > 1:
        return apply_to_2d(filt, x)
    return filt(x)

def matched_doppler_max(s, N, x):
    xshape = x.shape
    filt = MatchedDopplerMax(s, N, xshape[-1], x.dtype, measure=False)
    if len(xshape) > 1:
        return apply_to_2d(filt, x)
    return filt(x)