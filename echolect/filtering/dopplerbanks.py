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
import scipy as sp
import scipy.sparse
import pyfftw
import numba
from numba.decorators import jit, autojit
import multiprocessing

from . import filters
from .util import pow2, zero_pad
from echolect.filtering import libdopplerbanks

_THREADS = multiprocessing.cpu_count()


def dopplerbank_dec(h, N, M, **kwargs):
    def decor(filt):
        return filters.filter_dec(h, M, N=N, **kwargs)(filt)

    return decor


# ******** Doppler banks implemented via convolution after frequency-shifting ********

def ShiftConv(h, N, M):
    # implements Doppler filter:
    # y[n, p] = SUM_k (exp(-2*pi*j*n*(k - (L-1))/N) * h[k]) * x[p - k]
    #         = SUM_k (exp(-2*pi*j*n*k/N) * s*[k]) * x[p - (L-1) + k]
    L = len(h)
    outlen = M + L - 1

    dopplermat = np.exp(-2*np.pi*1j*np.arange(N)[:, np.newaxis]*np.arange(L)/N)[:, ::-1]
    dopplermat.astype(np.result_type(h.dtype, np.complex64)) # cast to complex type with precision of h
    hbank = h*dopplermat

    @dopplerbank_dec(h, N, M, hbank=hbank)
    def shiftconv(x):
        y = np.empty((N, outlen), np.result_type(hbank.dtype, x.dtype))
        for k, h in enumerate(hbank):
            y[k] = np.convolve(h, x)

        return y

    return shiftconv

def ShiftConvFFT(h, N, M, xdtype=np.complex_, powerof2=True):
    # implements Doppler filter:
    # y[n, p] = SUM_k (exp(-2*pi*j*n*(k - (L-1))/N) * h[k]) * x[p - k]
    #         = SUM_k (exp(-2*pi*j*n*k/N) * s*[k]) * x[p - (L-1) + k]
    L = len(h)
    outlen = M + L - 1
    nfft = outlen
    if powerof2:
        nfft = pow2(nfft)

    dopplermat = np.exp(-2*np.pi*1j*np.arange(N)[:, np.newaxis]*np.arange(L)/N)[:, ::-1]
    dopplermat.astype(np.result_type(h.dtype, np.complex64)) # cast to complex type with precision of h
    hbank = h*dopplermat
    # speed not critical here, just use numpy fft
    hbankpad = zero_pad(hbank, nfft)
    H = np.fft.fft(hbankpad) / nfft # divide by nfft b/c FFTW's ifft does not do this

    xctype = np.result_type(xdtype, np.complex64) # cast to complex type with precision of x
    xpad = pyfftw.n_byte_align(np.zeros(nfft, xctype), 16)
    X = pyfftw.n_byte_align(np.zeros(nfft, xctype), 16)
    xfft = pyfftw.FFTW(xpad, X, threads=_THREADS)

    ytype = np.result_type(H.dtype, xctype)
    Y = pyfftw.n_byte_align_empty(H.shape, 16, ytype)
    y = pyfftw.n_byte_align_empty(H.shape, 16, ytype)
    ifft = pyfftw.FFTW(Y, y, direction='FFTW_BACKWARD', threads=_THREADS)

    @dopplerbank_dec(h, N, M, nfft=nfft, H=H)
    def shiftconv_fft(x):
        xpad[:M] = x
        xfft.execute() # input is xpad, output is X
        np.multiply(X, H, Y)
        ifft.execute() # input is Y, output is y

        yc = np.array(y)[:, :outlen] # need a copy, which np.array provides
        return yc

    return shiftconv_fft

def ShiftConvNumbaFFT(h, N, M, xdtype=np.complex_, powerof2=True):
    # implements Doppler filter:
    # y[n, p] = SUM_k (exp(-2*pi*j*n*(k - (L-1))/N) * h[k]) * x[p - k]
    #         = SUM_k (exp(-2*pi*j*n*k/N) * s*[k]) * x[p - (L-1) + k]
    L = len(h)
    outlen = M + L - 1
    nfft = outlen
    if powerof2:
        nfft = pow2(nfft)

    dopplermat = np.exp(-2*np.pi*1j*np.arange(N)[:, np.newaxis]*np.arange(L)/N)[:, ::-1]
    dopplermat.astype(np.result_type(h.dtype, np.complex64)) # cast to complex type with precision of h
    hbank = h*dopplermat
    # speed not critical here, just use numpy fft
    hbankpad = zero_pad(hbank, nfft)
    H = np.fft.fft(hbankpad) / nfft # divide by nfft b/c FFTW's ifft does not do this

    xcdtype = np.result_type(xdtype, np.complex64) # cast to complex type with precision of x
    xpad = pyfftw.n_byte_align(np.zeros(nfft, xcdtype), 16)
    X = pyfftw.n_byte_align(np.zeros(nfft, xcdtype), 16)
    xfft = pyfftw.FFTW(xpad, X, threads=_THREADS)

    ydtype = np.result_type(H.dtype, xcdtype)
    Y = pyfftw.n_byte_align_empty(H.shape, 16, ydtype)
    y = pyfftw.n_byte_align_empty(H.shape, 16, ydtype)
    ifft = pyfftw.FFTW(Y, y, direction='FFTW_BACKWARD', threads=_THREADS)
    
    xtype = numba.__getattribute__(str(np.dtype(xdtype)))
    
    #htype = numba.__getattribute__(str(H.dtype))
    #xctype = numba.__getattribute__(str(X.dtype))
    #ytype = numba.__getattribute__(str(Y.dtype))
    #@jit(argtypes=[htype[:, ::1], xctype[::1], ytype[:, ::1], xtype[::1]])
    #def fun(H, X, Y, x):
        #xpad[:M] = x
        #xfft.execute() # input is xpad, output is X
        #Y[:, :] = H*X # need expression optimized by numba but that writes into Y
        #ifft.execute() # input is Y, output is y

        #yc = np.array(y)[:, :outlen] # need a copy, which np.array provides
        #return yc
    
    #@dopplerbank_dec(h, N, M, nfft=nfft, H=H)
    #def shiftconv_numba_fft(x):
        #return fun(H, X, Y, x)

    @jit(argtypes=[xtype[::1]])
    def shiftconv_numba_fft(x):
        xpad[:M] = x
        xfft.execute() # input is xpad, output is X
        Y[:, :] = X*H # need expression optimized by numba but that writes into Y
        ifft.execute() # input is Y, output is y

        yc = np.array(y)[:, :outlen] # need a copy, which np.array provides
        return yc
    
    shiftconv_numba_fft = dopplerbank_dec(h, N, M, nfft=nfft, H=H)(shiftconv_numba_fft)

    return shiftconv_numba_fft

def ShiftConvSparseMod(h, N, M):
    """Doppler bank where the signal is downshift modulated before filtering."""
    # implements Doppler filter:
    # y[n, p] = SUM_k h[p - k] * (x[k] * exp(-2*pi*j*n*k/N))
    #         = SUM_k s*[k + (L-1) - p] * (x[k] * exp(-2*pi*j*n*k/N))
    L = len(h)

    hpad = np.hstack((np.zeros(M - 1, h.dtype), h, np.zeros(M - 1, h.dtype)))
    hmat = np.lib.stride_tricks.as_strided(hpad[(M - 1):], (M + L - 1, M),
                                           (hpad.itemsize, -hpad.itemsize))
    hmat = sp.sparse.csr_matrix(hmat.T)

    dftmat = np.exp(-2*np.pi*1j*np.arange(N)[:,np.newaxis]*np.arange(M)/N)

    @dopplerbank_dec(h, N, M, hmat=hmat, dftmat=dftmat)
    def shiftconv_sparse_mod(x):
        return np.asarray((dftmat*x)*hmat)

    return shiftconv_sparse_mod

def ShiftConvSparse(h, N, M):
    """Doppler bank where the filter is upshift modulated before filtering."""
    # implements Doppler filter:
    # y[n, p] = SUM_k exp(-2*pi*j*n*(k - (L-1))/N) * h[k] * x[p - k]
    #         = SUM_k exp(-2*pi*j*n*k/N) * s*[k] * x[p - (L-1) + k]
    filt = ShiftConvSparseMod(h, N, M)
    L = filt.L

    phasecorrect = np.exp(2*np.pi*1j*np.arange(N)[:, np.newaxis]*np.arange(-(L-1),M)/N)

    @dopplerbank_dec(h, N, M, hmat=filt.hmat, dftmat=filt.dftmat, phasecorrect=phasecorrect)
    def shiftconv_sparse(x):
        return phasecorrect*filt(x)

    return shiftconv_sparse


# ******** Doppler banks implemented by sweeping demodulation of the input and 
#          calculation of the spectrum for segments of the input               ********

def SweepSpectraCython(h, N, M, xdtype=np.complex_):
    # implements Doppler filter:
    # y[n, p] = SUM_k exp(-2*pi*j*n*(k - (L-1))/N) * (h[k] * x[p - k])
    #         = SUM_k exp(-2*pi*j*n*k/N) * (s*[k] * x[p - (L-1) + k])
    L = len(h)
    outlen = M + L - 1
    # when N < L, still need to take FFT with nfft >= L so we don't lose data
    # then subsample to get our N points that we desire
    step = L // N + 1
    nfft = N*step
    
    # ensure that h is C-contiguous as required by the Cython function
    h = np.asarray(h, order='C')

    demodpad = np.zeros((outlen, nfft), np.result_type(xdtype, h.dtype, np.complex64))
    demodpad = pyfftw.n_byte_align(demodpad, 16)
    y = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, y, threads=_THREADS)
    
    sweepspectra_cython = libdopplerbanks.SweepSpectraCython(h, demodpad, y, fft, 
                                                             step, N, M, xdtype)
    sweepspectra_cython = dopplerbank_dec(h, N, M)(sweepspectra_cython)
    
    return sweepspectra_cython

def SweepSpectraNumba(h, N, M, xdtype=np.complex_):
    # implements Doppler filter:
    # y[n, p] = SUM_k exp(-2*pi*j*n*(k - (L-1))/N) * (h[k] * x[p - k])
    #         = SUM_k exp(-2*pi*j*n*k/N) * (s*[k] * x[p - (L-1) + k])
    L = len(h)
    outlen = M + L - 1
    # when N < L, still need to take FFT with nfft >= L so we don't lose data
    # then subsample to get our N points that we desire
    step = L // N + 1
    nfft = N*step
    
    hrev = h[::-1]
    xpad = np.zeros(M + 2*(L - 1), xdtype) # x[0] at xpad[L - 1]

    demodpad = np.zeros((outlen, nfft), np.result_type(xdtype, h.dtype, np.complex64))
    demodpad = pyfftw.n_byte_align(demodpad, 16)
    y = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, y, threads=_THREADS)

    xtype = numba.__getattribute__(str(np.dtype(xdtype)))
    
    @jit(argtypes=[xtype[::1]])
    def sweepspectra_numba(x):
        xpad[(L - 1):outlen] = x
        for p in range(outlen):
            demodpad[p, :L] = hrev*xpad[p:(p + L)]
        fft.execute() # input is demodpad, output is y
        yc = np.array(y[:, ::step].T) # we need a copy, which np.array provides
        return yc
    
    sweepspectra_numba = dopplerbank_dec(h, N, M)(sweepspectra_numba)

    return sweepspectra_numba

def SweepSpectraStridedInput(h, N, M, xdtype=np.complex_):
    # implements Doppler filter:
    # y[n, p] = SUM_k exp(-2*pi*j*n*(k - (L-1))/N) * (h[k] * x[p - k])
    #         = SUM_k exp(-2*pi*j*n*k/N) * (s*[k] * x[p - (L-1) + k])
    L = len(h)
    outlen = M + L - 1
    # when N < L, still need to take FFT with nfft >= L so we don't lose data
    # then subsample to get our N points that we desire
    step = L // N + 1
    nfft = N*step
    
    hrev = h[::-1]
    xpad = np.zeros(M + 2*(L - 1), xdtype) # x[0] at xpad[L - 1]
    xshifted = np.lib.stride_tricks.as_strided(xpad,
                                               (outlen, L),
                                               (xpad.itemsize, xpad.itemsize))

    demodpad = np.zeros((outlen, nfft), np.result_type(xdtype, h.dtype, np.complex64))
    demodpad = pyfftw.n_byte_align(demodpad, 16)
    y = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, y, threads=_THREADS)

    @dopplerbank_dec(h, N, M)
    def sweepspectra_strided_input(x):
        xpad[(L - 1):outlen] = x
        np.multiply(xshifted, hrev, demodpad[:, :L])
        fft.execute() # input is demodpad, output is y
        yc = np.array(y[:, ::step].T) # we need a copy, which np.array provides
        return yc

    return sweepspectra_strided_input

def SweepSpectraStridedTapsMod(h, N, M, xdtype=np.complex_):
    """Doppler bank where the signal is downshift modulated before filtering."""
    # implements Doppler filter:
    # y[n, p] = SUM_k (h[p - k] * x[k]) * exp(-2*pi*j*n*k/N)
    #         = SUM_k (s*[k + (L-1) - p] * x[k]) * exp(-2*pi*j*n*k/N)
    L = len(h)
    # when N < M, still need to take FFT with nfft >= M so we don't lose data
    # then subsample to get our N points that we desire
    step = M // N + 1
    nfft = N*step

    hpad = np.hstack((np.zeros(M - 1, h.dtype), h, np.zeros(M - 1, h.dtype)))
    hmat = np.lib.stride_tricks.as_strided(hpad[(M - 1):], (M + L - 1, M),
                                           (hpad.itemsize, -hpad.itemsize))

    demodpad = np.zeros((M + L - 1, nfft), np.result_type(xdtype, h.dtype, np.complex64))
    demodpad = pyfftw.n_byte_align(demodpad, 16)
    y = pyfftw.n_byte_align(np.zeros_like(demodpad), 16)
    fft = pyfftw.FFTW(demodpad, y, threads=_THREADS)

    @dopplerbank_dec(h, N, M, hmat=hmat)
    def sweepspectra_strided_taps_mod(x):
        np.multiply(hmat, x, demodpad[:, :M])
        fft.execute() # input is demodpad, output is y
        yc = np.array(y[:, ::step].T) # need a copy, which np.array provides
        return yc

    return sweepspectra_strided_taps_mod

def SweepSpectraStridedTaps(h, N, M, xdtype=np.complex_):
    """Doppler bank where the filter is upshift modulated before filtering."""
    # implements Doppler filter:
    # y[n, p] = SUM_k exp(-2*pi*j*n*(k - (L-1))/N) * h[k] * x[p - k]
    #         = SUM_k exp(-2*pi*j*n*k/N) * s*[k] * x[p - (L-1) + k]
    filt = SweepSpectraStridedTapsMod(h, N, M, xdtype)
    L = filt.L

    phasecorrect = np.exp(2*np.pi*1j*np.arange(N)[:, np.newaxis]*np.arange(-(L-1),M)/N)

    @dopplerbank_dec(h, N, M, hmat=filt.hmat, phasecorrect=phasecorrect)
    def sweepspectra_strided_taps(x):
        return phasecorrect*filt(x)

    return sweepspectra_strided_taps