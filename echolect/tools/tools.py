# things I wish were in numpy
import numpy as np

__all__ = ['valargcmp', 'valargmax', 'valargmin', 'itarray']

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