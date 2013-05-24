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
import scipy.optimize

import interpolators

__all__ = ['max', 'quadratic', 'quadratic_fixedskew', 'quadratic_lstsq', 
           'bicubic_hermite', 'biquintic_hermite']

def max(x):
    max_idx = np.argmax(x)
    max_idx = np.unravel_index(max_idx, x.shape)

    val = x[max_idx]

    return max_idx, val

def quadratic(x):
    max_idx = np.argmax(x)
    max_idx = np.unravel_index(max_idx, x.shape)

    K = np.array([-1, -1, 0, 1, 1, 0])
    L = np.array([-1, 1, 0, -1, 1, 1])
    # wrap the indices around the end because it either makes sense (Doppler)
    # or we're probably detecting noise anyway
    z = x[(max_idx[0] + K) % x.shape[0], (max_idx[1] + L) % x.shape[1]]
    A = np.matrix([K**2, K*L, L**2, K, L, np.ones(len(K))]).T

    C = np.linalg.solve(A, z)
    a, b, c, d, e, f = C

    i0 = (2*c*d - b*e)/(b**2 - 4*a*c)
    i1 = (2*a*e - b*d)/(b**2 - 4*a*c)
    val = f - a*i0**2 - b*i0*i1 - c*i1**2

    return (max_idx[0] + i0, max_idx[1] + i1), val

def quadratic_fixedskew(x, b=0):
    max_idx = np.argmax(x)
    max_idx = np.unravel_index(max_idx, x.shape)

    K = np.array([-1, 0, 0, 0, 1])
    L = np.array([0, -1, 0, 1, 0])
    # wrap the indices around the end because it either makes sense (Doppler)
    # or we're probably detecting noise anyway
    z = x[(max_idx[0] + K) % x.shape[0], (max_idx[1] + L) % x.shape[1]]
    z0 = z - b*K*L
    A = np.matrix([K**2, L**2, K, L, np.ones(len(K))]).T

    C = np.linalg.solve(A, z0)
    a, c, d, e, f = C

    i0 = (2*c*d - b*e)/(b**2 - 4*a*c)
    i1 = (2*a*e - b*d)/(b**2 - 4*a*c)
    val = f - a*i0**2 - b*i0*i1 - c*i1**2

    return (max_idx[0] + i0, max_idx[1] + i1), val

def quadratic_lstsq(x):
    max_idx = np.argmax(x)
    max_idx = np.unravel_index(max_idx, x.shape)

    idx0 = np.array([-1, 0, 1])
    idx1 = np.array([-1, 0, 1])
    K, L = np.meshgrid(idx0, idx1)
    K = K.ravel()
    L = L.ravel()
    # wrap the indices around the end because it either makes sense (Doppler)
    # or we're probably detecting noise anyway
    z = x[(max_idx[0] + K) % x.shape[0], (max_idx[1] + L) % x.shape[1]]
    A = np.matrix([K**2, K*L, L**2, K, L, np.ones(len(K))]).T

    C, _, _, _ = np.linalg.lstsq(A, z)
    a, b, c, d, e, _ = C
    f = x[max_idx] # lstsq fit for curvature, but peak value based on max value

    i0 = (2*c*d - b*e)/(b**2 - 4*a*c)
    i1 = (2*a*e - b*d)/(b**2 - 4*a*c)
    val = f - a*i0**2 - b*i0*i1 - c*i1**2

    return (max_idx[0] + i0, max_idx[1] + i1), val

def bicubic_hermite(x):
    # interpolate with bicubic Hermite spline using finite difference approximation
    # for derivatives, equivalent to cubic convolution interpolation
    max_idx = np.argmax(x)
    max_idx = np.unravel_index(max_idx, x.shape)

    # wrap the indices around the end because it either makes sense (Doppler)
    # or we're probably detecting noise anyway
    xinterp = interpolators.Bicubic(x, bc='wrap')
    f = lambda z: -xinterp.f(*z)
    fprime = lambda z: -xinterp.grad(*z)
    x0 = np.asarray(max_idx)
    xopt, fopt, d = sp.optimize.fmin_l_bfgs_b(f, x0, fprime,
                                              bounds=[(x0[0] - 1, x0[0] + 1),
                                                      (x0[1] - 1, x0[1] + 1)])

    return xopt, -fopt

def biquintic_hermite(x):
    # interpolate with biquintic Hermite spline using finite difference approximation
    # for derivatives, equivalent to cubic convolution interpolation
    max_idx = np.argmax(x)
    max_idx = np.unravel_index(max_idx, x.shape)

    # wrap the indices around the end because it either makes sense (Doppler)
    # or we're probably detecting noise anyway
    xinterp = interpolators.Biquintic(x, bc='wrap')
    f = lambda z: -xinterp.f(*z)
    fprime = lambda z: -xinterp.grad(*z)
    x0 = np.asarray(max_idx)
    xopt, fopt, d = sp.optimize.fmin_l_bfgs_b(f, x0, fprime,
                                              bounds=[(x0[0] - 1, x0[0] + 1),
                                                      (x0[1] - 1, x0[1] + 1)])

    return xopt, -fopt