#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

__all__ = ['find_index', 'slice_by_value', 'wrap_check_start', 'wrap_check_stop']

def find_index(arr, val):
    """Find index n into sorted array arr so that arr[n] <= val < arr[n+1].

    Array arr must be sorted in ascending order.
    
    """
    n = np.searchsorted(arr, val, side='right') - 1

    if n == -1:
        raise ValueError('val comes before arr[0]')

    return n

def slice_by_value(arr, low=None, high=None):
    """Find slice into sorted array arr that includes values from low through high.

    Array arr must be sorted in ascending order.
    
    """
    if low is None or low <= arr[0]:
        start = None
    else:
        start = np.searchsorted(arr, low, side='left')

    if high is None or high >= arr[-1]:
        stop = None
    else:
        stop = np.searchsorted(arr, high, side='right')

    return slice(start, stop, 1)

def wrap_check_start(l, start):
    """Check that start index falls in [-l, l) and wrap negative values to l + start."""
    if (start < -l) or (start >= l):
        raise IndexError('start index out of range')
    if start < 0:
        start = start % l
    return start

def wrap_check_stop(l, stop):
    """Check that stop index falls in (-l, l] and wrap negative values to l + stop.
    
    For convenience, stop == 0 is assumed to be shorthand for stop == l.
    
    """
    if (stop <= -l) or (stop > l):
        raise IndexError('stop index out of range')
    elif stop <= 0:
        # let stop == 0 be shorthand for stop == l,
        # i.e. including all profiles to the end
        stop = (stop - 1) % l + 1
    return stop