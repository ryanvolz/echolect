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

from echolect.filtering import filtering

__all__ = ['autocorr', 'ambiguity']

def autocorr(code, nfreq=1):
    """Calculate autocorrelation of code for nfreq frequencies.
    
    If nfreq == 1, the result is a 1-D array with length that is 
    2*len(code) - 1. The peak value of sum(abs(code)**2) is located
    in the middle at index len(code) - 1.
    
    If nfreq > 1, the result is a 2-D array with the first index
    corresponding to frequency shift. The code is frequency shifted
    by normalized frequencies of range(nfreq)/nfreq and correlated
    with the baseband code. The result acorr[0] gives the 
    autocorrelation with 0 frequency shift, acorr[1] with 1/nfreq
    frequency shift, etc. These frequencies are the same as (and 
    are in the same order as) the FFT frequencies for an nfreq-
    length FFT.
    ****Thus, the peak value is at acorr[0, len(code) - 1]****
    
    To relocate the peak to the middle of the result, use
        np.fft.fftshift(acorr, axes=0)
    To relocate the peak to the [0, 0] entry, use
        np.fft.ifftshift(acorr, axes=1)
    
    """
    # special case because matched_doppler does not handle nfreq < len(code)
    if nfreq == 1:
        acorr = filtering.matched(code, code)
    else:
        acorr = filtering.matched_doppler(code, nfreq, code)
    
    return acorr

def ambiguity(code, nfreq=1):
    """Calculate the ambiguity function of code for nfreq frequencies.
    
    The ambiguity function is the square of the autocorrelation, 
    normalized so the peak value is 1.
    
    See autocorr for details.
    
    """
    acorr = autocorr(code, nfreq)
    # normalize so answer at zero delay, zero Doppler is 1
    b = len(code)
    if nfreq == 1:
        acorr = acorr / acorr[b - 1]
    else:
        acorr = acorr / acorr[0, b - 1]
    
    amb = acorr.real**2 + acorr.imag**2

    return amb