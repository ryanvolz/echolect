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
import timeit

def pow2(n):
    return 2**(int(np.ceil(np.log2(n))))

def zero_pad(x, n):
    s = x.shape
    m = s[-1]
    if m == n:
        return x
    elif m > n:
        return x[..., 0:n]
    else:
        y = np.zeros(s[:-1] + (n,), dtype=x.dtype)
        y[..., 0:m] = x
        return y

def upsample(x, n, axis=0, phase=0):
    """Upsample x by inserting n-1 zeros between samples along the specified axis."""
    x = np.asarray(x)
    n = int(n)
    if phase < 0 or phase >= n:
        raise ValueError('phase must be between 0 and n-1')

    upshape = list(x.shape)
    upshape[axis] = n*x.shape[axis]
    y = np.zeros(upshape, x.dtype)
    
    idx = [slice(None)]*y.ndim
    idx[axis] = slice(phase, None, n)
    y[idx] = x

    return y

def downsample(x, n, axis=0, phase=0):
    """Downsample x by keeping every nth sample along the specified axis, starting with phase."""
    x = np.asarray(x)
    n = int(n)
    if phase < 0 or phase >= n:
        raise ValueError('phase must be between 0 and n-1')
    
    idx = [slice(None)]*x.ndim
    idx[axis] = slice(phase, None, n)

    return x[idx]

def time_filters(flist, x, number=100):
    times = []
    for filt in flist:
        timer = timeit.Timer(lambda: filt(x))
        times.append(min(timer.repeat(repeat=3, number=number)))

    return times

def convslice(L, M, mode='validsame'):
    smaller = min(L, M)
    bigger = max(L, M)
    if mode == 'valid':
        return slice(smaller - 1, bigger)
    elif mode == 'same':
        return slice((smaller - 1)//2, (smaller - 1)//2 + bigger)
    elif mode == 'validsame':
        return slice(smaller - 1, None)
    else:
        return slice(None)

def apply_filter_mode(filt, res, mode=None):
    if mode is None or mode == 'full':
        return res
    
    try:
        slc = getattr(filt, mode)
    except AttributeError:
        raise ValueError('Unknown mode')
    
    return res[..., slc]

def apply_to_2d(func1d, arr):
    if len(arr.shape) != 2:
        raise ValueError('arr must be 2-D')
    res = func1d(arr[0])
    if not isinstance(res, tuple):
        res = np.asarray(res)
        outshape = arr.shape[:1] + res.shape
        out = np.empty(outshape, res.dtype)
        out[0] = res
        for k in xrange(1, outshape[0]):
            row = arr[k]
            res = func1d(row)
            out[k] = res
    else:
        resarr = tuple(np.asarray(r) for r in res)
        try:
            # res is a namedtuple, keep same class
            out = res.__class__(*(np.empty(arr.shape[:1] + r.shape, r.dtype) for r in resarr))
        except:
            out = tuple(np.empty(arr.shape[:1] + r.shape, r.dtype) for r in resarr)
        for l, r in enumerate(res):
            out[l][0] = r
        for k in xrange(1, arr.shape[0]):
            row = arr[k]
            res = func1d(row)
            for l, r in enumerate(res):
                out[l][k] = r

    return out