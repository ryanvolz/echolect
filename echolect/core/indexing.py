import numpy as np

__all__ = ['find_index', 'slice_by_value']

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

def wrap_check_start_stop(l, start, stop=None):
    """Check start and stop indices and wrap negative indices to positive.
    
    Check that start index falls in [-l, l) and wrap negative values to l + start.
    Check that stop index falls in (-l, l] and wrap negative values to l + stop.
    
    If stop is None, stop = start + 1 is assumed.
    For convenience, stop == 0 is assumed to be shorthand for stop == l.
    
    """
    if (start < -l) or (start >= l):
        raise IndexError('start index out of range')
    if start < 0:
        start = start % l
    
    if stop is None:
        stop = start + 1
    elif (stop <= -l) or (stop > l):
        raise IndexError('stop index out of range')
    elif stop <= 0:
        # let stop == 0 be shorthand for stop == l,
        # i.e. including all profiles to the end
        stop = (stop - 1) % l + 1
    
    return start, stop