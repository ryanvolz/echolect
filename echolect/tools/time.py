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

__all__ = ['datetime_from_float', 'datetime_to_float', 'timestamp_strftime']

def datetime_from_float(t, unit, epoch=None):
    """Convert an array of floats to datetime64.
    
    It is assumed that the float represents time in seconds since the
    given datetime64 epoch. If epoch is None, np.datetime64(0) is used.
    
    """
    if epoch is None:
        epoch = np.datetime64(0, 's')
    factor = np.timedelta64(1, 's')/np.timedelta64(1, unit)
    dt = np.round(factor*t).astype('timedelta64[{0}]'.format(unit))
    return epoch + dt

def datetime_to_float(t, epoch=None):
    """Convert a datetime64 array to floating point.
    
    The floating point representation gives the time in seconds since
    the given datetime64 epoch. If epoch is None, np.datetime64(0) is used.
    
    """
    if epoch is None:
        epoch = np.datetime64(0, 's')
    return (t - epoch)/np.timedelta64(1, 's')

def timestamp_strftime(t, fstr):
    # replace %f with fractional digits, up to stored nanoseconds
    # (this is instead of the datetime strftime default of microseconds)
    # leading zeros included, trailing zeros removed
    frac = '{0:09}'.format(t.microsecond*1000 + t.nanosecond).rstrip('0')
    if frac == '':
        frac = '0'
    tsfstr = fstr.replace('%f', frac)
    # now delegate to strftime method
    return t.strftime(tsfstr)