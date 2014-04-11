#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import division

import numpy as np
import scipy as sp
import scipy.interpolate

from echolect.filtering import filtering

__all__ = ['cubic_spline', 'cubic_polycoefs', 'cubic_pad', 
           'quintic_spline', 'quintic_polycoefs', 'quintic_pad',
           'polysum_interp', 'polysum_interp1d_factory', 'polysum_interp2d_factory',
           'cubic_filter', 'cubic', 'Cubic', 'Bicubic',
           'quintic_filter', 'quintic', 'Quintic', 'Biquintic']

#############################################################################
# Cubic
#############################################################################
# create piecewise polynomial by specifiying the cubic convolution spline
# values and its derivatives:
# u(-2) = 0, u'(-2) = 0
# u(-1) = 0, u'(-1) = 0.5
# u(0) = 1, u'(0) = 0
# u(1) = 0, u'(1) = -0.5
# u(2) = 0, u'(2) = 0
# u(x) = 0 for |x| > 2
cubic_spline = sp.interpolate.PiecewisePolynomial([-3, -2, -1, 0, 1, 2, 3],
                                                  [[0], [0, 0], [0, 0.5], [1, 0],
                                                   [0, -0.5], [0, 0], [0]],
                                                  [0, 3, 3, 3, 3, 0])

cubic_polycoefs = 1/2*np.array([[0, 2, 0, 0],
                                [-1, 0, 1, 0],
                                [2, -5, 4, -1],
                                [-1, 3, -3, 1]])

def cubic_pad(x, axis=-1):
    """Pads with boundary conditions appropriate for use with cubic interpolation filter."""
    xbc = np.concatenate((3*x.take((0,), axis) - 3*x.take((1,), axis) + x.take((2,), axis),
                          x,
                          3*x.take((-1,), axis) - 3*x.take((-2,), axis) + x.take((-3,), axis)
                          ), axis=axis)

    return xbc


#############################################################################
# Quintic
#############################################################################
# create piecewise polynomial by specifiying the quintic convolution spline
# values and its derivatives:
# u(-3)=0, u'(-3)=0, u''(-3)=0
# u(-2)=0, u'(-2)=-1/12, u''(-2)=-1/12
# u(-1)=0, u'(-1)=2/3, u''(-1)=4/3
# u(0)=1, u'(0)=0, u''(0)=-5/2
# u(1)=0, u'(1)=-2/3, u''(1)=4/3
# u(2)=0, u'(2)=1/12, u''(2)=-1/12
# u(3)=0, u'(3)=0, u''(3)=0
# u(x)=0 for |x| > 3
quintic_spline = sp.interpolate.PiecewisePolynomial([-4, -3, -2, -1, 0, 1, 2, 3, 4],
                                                    [[0], [0, 0, 0],
                                                     [0, -1/12, -1/12],
                                                     [0, 2/3, 4/3],
                                                     [1, 0, -5/2],
                                                     [0, -2/3, 4/3],
                                                     [0, 1/12, -1/12],
                                                     [0, 0, 0], [0]],
                                                    [0, 5, 5, 5, 5, 5, 5, 0])

quintic_polycoefs = 1/24*np.array([[0, 0, 24, 0, 0, 0],
                                   [2, -16, 0, 16, -2, 0],
                                   [-1, 16, -30, 16, -1, 0],
                                   [-9, 39, -70, 66, -33, 7],
                                   [13, -64, 126, -124, 61, -12],
                                   [-5, 25, -50, 50, -25, 5]])

def quintic_pad(x, axis=-1):
    """Pads with boundary conditions appropriate for use with quintic interpolation filter."""
    xbc = np.concatenate((15*x.take((0,), axis) - 40*x.take((1,), axis) + 45*x.take((2,), axis)
                              - 24*x.take((3,), axis) + 5*x.take((4,), axis),
                          5*x.take((0,), axis) - 10*x.take((1,), axis) + 10*x.take((2,), axis)
                              - 5*x.take((3,), axis) + x.take((4,), axis),
                          x,
                          5*x.take((-1,), axis) - 10*x.take((-2,), axis) + 10*x.take((-3,), axis)
                              - 5*x.take((-4,), axis) + x.take((-5,), axis),
                          15*x.take((-1,), axis) - 40*x.take((-2,), axis) + 45*x.take((-3,), axis)
                              - 24*x.take((-4,), axis) + 5*x.take((-5,), axis)
                          ), axis=axis)

    return xbc

#############################################################################
# General
#############################################################################
def polysum_interp(t, C, p):
    """Evaluate a sum of polynomials interpolant.

    Parameters
    ----------
    t : ndarray of any shape
        Values within [0, 1] at which to evaluate the interpolant.

    C : 2-d ndarray, with shape == (order + 1, npoints)
        Array giving the coefficients of the basis polynomials. The j'th
        column gives the polynomial coefficients for j'th data point, while
        the i'th row gives the coefficients for the term with order i.

    p : ndarray, with shape == t.shape + (npoints,)
        The points being interpolated, ordered in the last dimension with
        increasing index.

    e.g. for the cubic convolution interpolant and scalar t where
             f(t) =  pn1*(-t + 2t**2 - t**3)/2
                    + p0*(2 - 5t**2 + 3t**3)/2
                    + p1*(t + 4t**2 - 3t**3)/2
                    + p2*(-t**2 + t**3)/2
         interpolates on [0, 1] between the values p0 (at 0) and p1 (at 1)
         with bordering points pn1 (at -1) and p2 (at 2), one would use
             C = 1/2*np.array([[0, 2, 0, 0],
                               [-1, 0, 1, 0],
                               [2, -5, 4, -1],
                               [-1, 3, -3, 1]])
         and
             p = [pn1, p0, p1, p2].

    Returns
    -------
    vals : ndarray the same shape as t
        The values of the interpolant.

    """
    # p.shape == t.shape + (npoints,)
    t = np.asarray(t)[..., None]
    polyexp = np.arange(C.shape[0])
    T = np.power(t, polyexp) # t.shape x order+1
    polyvals = np.dot(T, C) # t.shape x npoints
    #polyvals = np.polynomial.polynomial.polyval(t.T, CubicInterp1d.C).T # t.shape x npoints
    vals = (p*polyvals).sum(-1)

    return vals

#############################################################################
# Factory
#############################################################################
def polysum_interp1d_factory(polycoefs, polyidx, optimal_pad_fun):
    #polycoefs_prime = np.polynomial.polynomial.polyder(polycoefs)
    Cps = []
    for c in polycoefs.T:
        Cps.append(np.polynomial.polynomial.polyder(c)[:, None])
    polycoefs_prime = np.concatenate(Cps, -1)
    
    class Interp1d(object):
        _C = polycoefs
        _Cp = polycoefs_prime
        
        def __init__(self, x, bc='optimal'):
            if bc is None or bc == 'edge':
                self.x = x
                self._cidx = polyidx
                self._idxmode = 'clip'
            elif bc == 'wrap':
                self.x = x
                self._cidx = polyidx
                self._idxmode = 'wrap'
            elif bc == 'optimal':
                self.x = optimal_pad_fun(x)
                added = len(self.x) - len(x)
                self._cidx = polyidx + added//2 # add added/2 to account for added B.C. entries
                self._idxmode = 'clip'
            elif bc == 'zero':
                self.x = np.zeros(len(x) + 2, dtype=x.dtype)
                self.x[1:-1] = x
                self._cidx = polyidx + 1 # add 1 to account for added B.C. entry
                self._idxmode = 'clip'
            else:
                raise ValueError("bc must be one of 'edge', 'optimal', 'wrap', or 'zero'")


        def __call__(self, i):
            return self.f(i)

        def _p_t(self, i):
            idx = np.floor(np.asarray(i)).astype(np.int)
            t = i - idx
            p = self.x.take(idx[:, None] + self._cidx, mode=self._idxmode)

            return p, t

        def f(self, i):
            p, t = self._p_t(i)
            return polysum_interp(t, self._C, p)

        def fp(self, i):
            p, t = self._p_t(i)
            return polysum_interp(t, self._Cp, p)

    return Interp1d

def polysum_interp2d_factory(polycoefs, polyidx, optimal_pad_fun):
    if isinstance(polycoefs, (tuple, list)):
        pcoefs = polycoefs[0], polycoefs[1]
    else:
        pcoefs = polycoefs, polycoefs
    #pcoefs_prime = tuple(np.polynomial.polynomial.polyder(p) for p in pcoefs)
    def polyder(C):
        Cps = []
        for c in C.T:
            Cps.append(np.polynomial.polynomial.polyder(c)[:, None])
        return np.concatenate(Cps, -1)
    pcoefs_prime = tuple(polyder(p) for p in pcoefs)

    if isinstance(polyidx, (tuple, list)):
        pidx = (polyidx[0][:, None], polyidx[1][None, :])
    else:
        pidx = (polyidx[:, None], polyidx[None, :])

    if isinstance(optimal_pad_fun, (tuple, list)):
        opadfuns = optimal_pad_fun[0], optimal_pad_fun[1]
    else:
        opadfuns = optimal_pad_fun, optimal_pad_fun
        
    class Interp2d(object):
        _iC, _jC = pcoefs
        _iCp, _jCp = pcoefs_prime
        
        def __init__(self, z, bc='optimal'):
            if bc is None or bc == 'edge':
                self.z = z
                self._cidx = pidx
                self._idxmode = 'clip'
            elif bc == 'wrap':
                self.z = z
                self._cidx = pidx
                self._idxmode = 'wrap'
            elif bc == 'optimal':
                self.z = opadfuns[1](opadfuns[0](z, 0), 1)
                iadded = self.z.shape[0] - z.shape[0]
                jadded = self.z.shape[1] - z.shape[1]
                self._cidx = (pidx[0] + iadded//2, pidx[1] + jadded//2) # add to account for added B.C. entries
                self._idxmode = 'clip'
            elif bc == 'zero':
                self.z = np.zeros((z.shape[0] + 2, z.shape[1] + 2), dtype=z.dtype)
                self.z[1:-1, 1:-1] = z
                self._cidx = (pidx[0] + 1, pidx[1] + 1) # add 1 to account for added B.C. entry
                self._idxmode = 'clip'
            else:
                raise ValueError("bc must be one of 'edge', 'optimal', 'wrap', or 'zero'")

        def __call__(self, i, j):
            return self.f(i, j)

        def _p_s_t(self, i, j):
            iidx = np.floor(np.asarray(i)).astype(np.int)
            s = i - iidx

            jidx = np.floor(np.asarray(j)).astype(np.int)
            t = j - jidx

            idx = np.ravel_multi_index((iidx[..., None, None] + self._cidx[0],
                                        jidx[..., None, None] + self._cidx[1]
                                       ), self.z.shape, mode=self._idxmode)

            p = self.z.take(idx)

            return p, s, t

        def f(self, i, j):
            p, s, t = self._p_s_t(i, j)

            # interpolate at t in j dimension (last)
            t = t[..., None] # so t broadcasts in the i dimension
            jvals = polysum_interp(t, self._jC, p)

            # interpolate at s in i dimension
            vals = polysum_interp(s, self._iC, jvals)

            return vals

        def fpi(self, i, j):
            p, s, t = self._p_s_t(i, j)

            # interpolate at t in j dimension (last)
            t = t[..., None] # so t broadcasts in the i dimension
            jvals = polysum_interp(t, self._jC, p)

            # interpolate at s in i dimension
            vals = polysum_interp(s, self._iCp, jvals)

            return vals

        def fpj(self, i, j):
            p, s, t = self._p_s_t(i, j)

            # interpolate at t in j dimension (last)
            t = t[..., None] # so t broadcasts in the i dimension
            jvals = polysum_interp(t, self._jCp, p)

            # interpolate at s in i dimension
            vals = polysum_interp(s, self._iC, jvals)

            return vals

        def grad(self, i, j):
            df_di = self.fpi(i, j)[..., None]
            df_dj = self.fpj(i, j)[..., None]

            g = np.concatenate((df_di, df_dj), -1) # broadcast(i,j) x 2

            return g

    return Interp2d

#############################################################################
# Specific
#############################################################################

def cubic_filter(n, offset=0):
    """Cubic interpolation filter for a sequence that has been upsampled by n.

    Following convolution, the values which interpolate the convolved sequence
    will start at index 2*n - 1. In other words, this filter has a delay of
    2*n - 1 samples from the non-causal interpolation filter.

    """

    # evaluate cubic convolution spline in within its domain of (-2, 2)
    # with offset from zero of offset/n and step size 1/n
    x = np.arange(-2*n + 1 - offset, 2*n)/float(n)
    h = cubic_spline(x)

    return h

def cubic(x, n, offset=0):
    # can't just convolve, need to set boundary conditions
    xbc = cubic_pad(x)
    xup = filtering.upsample(xbc, n)
    h = cubic_filter(n, offset)

    y = filtering.filter(h, xup)

    # delay of filter is 2*n - 1, so with b.c. padding of n, start at 3*n - 1
    start = 3*n - 1
    # stop at the last 'valid' (no implicit zero padding in conv) value
    stop = -len(h) + 1

    return y[start:stop]

Cubic = polysum_interp1d_factory(cubic_polycoefs,
                                 np.arange(-1, 3),
                                 cubic_pad)

Bicubic = polysum_interp2d_factory(cubic_polycoefs,
                                   np.arange(-1, 3),
                                   cubic_pad)

def quintic_filter(n, offset=0):
    """Quintic interpolation filter for a sequence that has been upsampled by n.

    Following convolution, the values which interpolate the convolved sequence
    will start at index 3*n - 1. In other words, this filter has a delay of
    3*n - 1 samples from the non-causal interpolation filter.

    """

    # evaluate quintic convolution spline in within its domain of (-3, 3)
    # with offset from zero of offset/n and step size 1/n
    x = np.arange(-3*n + 1 - offset, 3*n)/float(n)
    h = quintic_spline(x)

    return h

def quintic(x, n, offset=0):
    # can't just convolve, need to set boundary conditions
    xbc = quintic_pad(x)
    xup = filtering.upsample(xbc, n)
    h = quintic_filter(n, offset)

    y = filtering.filter(h, xup)

    # delay of filter is 3*n - 1, so with b.c. padding of 2*n, start at 5*n - 1
    start = 5*n - 1
    # stop at the last 'valid' (no implicit zero padding in conv) value
    stop = -len(h) + 1

    return y[start:stop]

Quintic = polysum_interp1d_factory(quintic_polycoefs,
                                   np.arange(-2, 4),
                                   quintic_pad)

Biquintic = polysum_interp2d_factory(quintic_polycoefs,
                                     np.arange(-2, 4),
                                     quintic_pad)