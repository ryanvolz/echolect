import numpy as np
import scipy as sp
import scipy.sparse
import scipy.fftpack
import pyfftw
import numba
from numba.decorators import jit, autojit
import multiprocessing

import libfilters
from util import convslice, pow2, zero_pad

_THREADS = multiprocessing.cpu_count()

def filter_dec(h, M, **kwargs):
    def decor(filt):
        filt.h = h
        filt.M = M
        L = len(h)
        filt.L = L

        for key, val in kwargs.iteritems():
            setattr(filt, key, val)

        filt.valid = convslice(L, M, 'valid')
        filt.same = convslice(L, M, 'same')
        filt.validsame = convslice(L, M, 'validsame')
        filt.nodelay = convslice(L, M, 'validsame')

        doc = """Apply filter to the input.

        Parameters
        ----------
        x : 1-D ndarray
            Values to be filtered.

        Returns
        -------
        y : 1-D ndarray
            Full output of the filter. Use filter attributes 'valid', 'same',
            or 'validsame' to slice into portions of the output.

        """

        if filt.__doc__ is not None:
            filt.__doc__ += doc
        else:
            filt.__doc__ = doc

        return filt

    return decor


# ******** filters that directly implement the convolution sum *********

def Conv(h, M):
    @filter_dec(h, M)
    def conv(x):
        return np.convolve(h, x, 'full')

    return conv

def CythonConv(h, M, xdtype):
    return filter_dec(h, M)(libfilters.Conv(h, M, xdtype))

#@autojit
#def NumbaConv(h, M, xdtype):
    #L = len(h)
    #outlen = M + L - 1
    
    #xdtype = np.dtype(xdtype)
    #outdtype = np.result_type(h.dtype, xdtype)

    #@jit('complex128[::1](complex128[::1])')
    #def numba_conv(x):
        #out = np.zeros(outlen, outdtype)
        #for m in range(M):
            #for l in range(L):
                #out[m + l] += h[l]*x[m]
        #return out

    #return filter_dec(h, M)(numba_conv)

def NumbaConv(h, M, xdtype):
    L = len(h)
    outlen = M + L - 1
    
    xdtype = np.dtype(xdtype)
    htype = numba.__getattribute__(str(h.dtype))
    xtype = numba.__getattribute__(str(xdtype))
    outdtype = np.result_type(h.dtype, xdtype)
    outtype = numba.__getattribute__(str(outdtype))

    @jit(restype=outtype[::1], argtypes=[htype[::1], xtype[::1]])
    def conv(h, x):
        out = np.zeros(outlen, outdtype)
        for m in range(M):
            for l in range(L):
                out[m + l] += h[l]*x[m]
        return out

    @filter_dec(h, M)
    def numba_conv(x):
        out = conv(h, x)
        return out

    return numba_conv

def SparseConv(h, M):
    L = len(h)

    hpad = np.hstack((np.zeros(M - 1, h.dtype), h, np.zeros(M - 1, h.dtype)))
    hmat = np.lib.stride_tricks.as_strided(hpad[(M - 1):], (M + L - 1, M),
                                            (hpad.itemsize, -hpad.itemsize))
    hmat = sp.sparse.csr_matrix(hmat)

    @filter_dec(h, M, hmat=hmat)
    def sparse_conv(x):
        return np.asarray(hmat*x)

    return sparse_conv

def StridedConv(h, M):
    L = len(h)

    hpad = np.hstack((np.zeros(M - 1, h.dtype), h, np.zeros(M - 1, h.dtype)))
    hmat = np.lib.stride_tricks.as_strided(hpad[(M - 1):], (M + L - 1, M),
                                           (hpad.itemsize, -hpad.itemsize))

    @filter_dec(h, M, hmat=hmat)
    def strided_conv(x):
        return np.sum(hmat*x, axis=-1)

    return strided_conv


# ******** filters that implement convolution via the FFT ********

def FFTPack(h, M, xdtype=np.complex_, powerof2=True):
    L = len(h)
    outlen = M + L - 1
    nfft = outlen
    if powerof2:
        nfft = pow2(nfft)

    H = sp.fftpack.fft(zero_pad(h, nfft))
    xpad = np.zeros(nfft, xdtype)

    @filter_dec(h, M, nfft=nfft, H=H)
    def fftpack(x):
        xpad[:M] = x
        X = sp.fftpack.fft(xpad)
        X *= H
        y = sp.fftpack.ifft(X)
        return y[:outlen]

    return fftpack

def FFTW(h, M, xdtype=np.complex_, powerof2=True):
    L = len(h)
    outlen = M + L - 1
    nfft = outlen
    if powerof2:
        nfft = pow2(nfft)

    outdtype = np.result_type(h.dtype, xdtype)
    fftdtype = np.result_type(outdtype, np.complex64) # output is always complex, promote using smallest

    # speed not critical here, just use numpy fft
    # cast to outdtype so we use same type of fft as when transforming x
    hpad = zero_pad(h, nfft).astype(outdtype)
    if np.iscomplexobj(hpad):
        H = np.fft.fft(hpad)
    else:
        H = np.fft.rfft(hpad)
    H = (H / nfft).astype(fftdtype) # divide by nfft b/c FFTW's ifft does not do this

    xpad = pyfftw.n_byte_align(np.zeros(nfft, outdtype), 16) # outdtype so same type fft as h->H
    X = pyfftw.n_byte_align(np.zeros(len(H), fftdtype), 16) # len(H) b/c rfft may be used
    xfft = pyfftw.FFTW(xpad, X, threads=_THREADS)

    y = pyfftw.n_byte_align_empty(nfft, 16, outdtype)
    ifft = pyfftw.FFTW(X, y, direction='FFTW_BACKWARD', threads=_THREADS)

    @filter_dec(h, M, nfft=nfft, H=H)
    def fftw(x):
        xpad[:M] = x
        xfft.execute() # input in xpad, result in X
        np.multiply(H, X, X)
        ifft.execute() # input in X, result in y
        yc = y[:outlen].copy()
        return yc

    return fftw

def NumbaFFTW(h, M, xdtype=np.complex_, powerof2=True):
    L = len(h)
    outlen = M + L - 1
    nfft = outlen
    if powerof2:
        nfft = pow2(nfft)

    outdtype = np.result_type(h.dtype, xdtype)
    fftdtype = np.result_type(outdtype, np.complex64) # output is always complex, promote using smallest

    # speed not critical here, just use numpy fft
    # cast to outdtype so we use same type of fft as when transforming x
    hpad = zero_pad(h, nfft).astype(outdtype)
    if np.iscomplexobj(hpad):
        H = np.fft.fft(hpad)
    else:
        H = np.fft.rfft(hpad)
    H = (H / nfft).astype(fftdtype) # divide by nfft b/c FFTW's ifft does not do this

    xpad = pyfftw.n_byte_align(np.zeros(nfft, outdtype), 16) # outdtype so same type fft as h->H
    X = pyfftw.n_byte_align(np.zeros(len(H), fftdtype), 16) # len(H) b/c rfft may be used
    xfft = pyfftw.FFTW(xpad, X, threads=_THREADS)

    y = pyfftw.n_byte_align_empty(nfft, 16, outdtype)
    ifft = pyfftw.FFTW(X, y, direction='FFTW_BACKWARD', threads=_THREADS)
    
    xtype = numba.__getattribute__(str(np.dtype(xdtype)))
    outtype = numba.__getattribute__(str(outdtype))
    ffttype = numba.__getattribute__(str(fftdtype))

    #@jit(restype=outtype[::1], 
         #argtypes=[outtype[::1], ffttype[::1], ffttype[::1], outtype[::1], xtype[::1]])
    #def filt(xpad, X, H, y, x):
        #xpad[:M] = x
        #xfft.execute() # input in xpad, result in X
        #X[:] = H*X
        #ifft.execute() # input in X, result in y
        #yc = y[:outlen].copy()
        #return yc
    
    #@filter_dec(h, M, nfft=nfft, H=H)
    #def numba_fftw(x):
        #return filt(xpad, X, H, y, x)
    
    @jit(argtypes=[xtype[::1]])
    def numba_fftw(x):
        xpad[:M] = x
        xfft.execute() # input in xpad, result in X
        X[:] = H*X # want expression that is optimized by numba but writes into X
        ifft.execute() # input in X, result in y
        yc = y[:outlen].copy()
        return yc
    
    numba_fftw = filter_dec(h, M, nfft=nfft, H=H)(numba_fftw)

    return numba_fftw

def NumpyFFT(h, M, xdtype=np.complex_, powerof2=True):
    L = len(h)
    outlen = M + L - 1
    nfft = outlen
    if powerof2:
        nfft = pow2(nfft)

    H = np.fft.fft(zero_pad(h, nfft))
    xpad = np.zeros(nfft, xdtype)

    @filter_dec(h, M, nfft=nfft, H=H)
    def numpy_fft(x):
        xpad[:M] = x
        X = np.fft.fft(xpad)
        X *= H
        y = np.fft.ifft(X)
        return y[:outlen]

    return numpy_fft