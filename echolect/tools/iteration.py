#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

# things I wish were in numpy
import numpy as np

__all__ = ['itarray']

def itarray(it, l):
    # creates a numpy array from a sequence of numpy arrays of known length
    # axis specifies the dimension that the sequence iterates over
    
    # get the first element in the sequence to determine shape and size
    a = it.next()
    # normal case with a just an ndarray
    if not isinstance(a, tuple):
        # create empty array to store the entire sequence
        all = np.empty((l,) + a.shape, dtype=a.dtype)
        # store what we got first
        all[0] = a
        
        # fill the rest of the array
        for k in xrange(1, l):
            all[k] = it.next()
    # special case with iterator returning tuples of ndarrays
    else:
        # create empty arrays to store the entire sequence
        all = tuple(np.empty((l,) + b.shape, dtype=b.dtype) for b in a)
        # store what we got first
        for kn, b in enumerate(a):
            all[kn][0] = b
        
        # fill the rest of each array
        for k in xrange(1, l):
            a = it.next()
            for kn, b in enumerate(a):
                all[kn][k] = b
    
    return all