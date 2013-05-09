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