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

# things I wish were in numpy
import numpy as np

__all__ = ['valargcmp', 'valargmax', 'valargmin']

def valargcmp(arr, axis=None, fun=np.argmax):
    args = fun(arr, axis)
    if axis is None:
        vals = arr.ravel()[args]
    else:
        idx = list(np.ix_(*[xrange(k) for k in args.shape]))
        idx.insert(axis, args)
        vals = arr[idx].squeeze()
    
    return vals, args

def valargmax(arr, axis=None):
    return valargcmp(arr, axis, np.argmax)

def valargmin(arr, axis=None):
    return valargcmp(arr, axis, np.argmin)