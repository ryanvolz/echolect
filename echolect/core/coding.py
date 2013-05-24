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
    # special case because matched_doppler does not handle nfreq < len(code)
    if nfreq == 1:
        acorr = filtering.matched(code, code)
    else:
        acorr = filtering.matched_doppler(code, nfreq, code)
    
    return acorr

def ambiguity(code, nfreq=1):
    acorr = autocorr(code, nfreq)
    # normalize so answer at zero delay, zero Doppler is 1
    b = len(code)
    if nfreq == 1:
        acorr = acorr / acorr[b - 1]
    else:
        acorr = acorr / acorr[0, b - 1]
    
    amb = acorr.real**2 + acorr.imag**2

    return amb