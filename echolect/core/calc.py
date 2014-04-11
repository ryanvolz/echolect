#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import scipy as sp
import scipy.stats
import scipy.constants
import bottleneck as bn

__all__ = ['med_pwr_est_factor', 'calc_power_factor', 'calc_rcs', 
           'calc_effective_area', 'calc_antenna_gain', 'calc_noise_cal_factor', 
           'altitude_from_range']

# factor with which to multiply median power of a complex voltage to robustly
# estimate the mean power, forming a MAD estimator
# sum of square of two independent gaussians is chisq(2), also equivalently exponential
noise_pwr_rv = sp.stats.chi2(2)
med_pwr_est_factor = noise_pwr_rv.mean()/noise_pwr_rv.median()

def calc_power_factor(noise_est, cal_est, cal_temp, bandwidth):
    return sp.constants.Bolzmann*bandwidth*cal_temp/(cal_est - noise_est)

def calc_rcs(rx_pwr, r, tx_pwr, gain_area_prod):
    """Calculate RCS, rx_pwr, r, and tx_pwr must broadcast appropriately."""
    return rx_pwr*((4*np.pi*r**2)**2)/tx_pwr/gain_area_prod

def calc_effective_area(gain, wavelength_m):
    return gain*wavelength_m**2/(4*np.pi)

def calc_antenna_gain(effective_area_msq, wavelength_m):
    return 4*np.pi*effective_area_msq/wavelength_m**2

def calc_noise_cal_factor(vlt, noise_slc, cal_slc, cal_temp, bandwidth):
    noise = vlt[:, noise_slc]
    cal = vlt[:, cal_slc]

    # assume noise is constant power over time window, robust MAD estimator
    noise_est = bn.median(noise.real**2 + noise.imag**2)*med_pwr_est_factor
    # cal temp is constant, so use all cal measurements to estimate power
    cal_est = bn.median(cal.real**2 + cal.imag**2)*med_pwr_est_factor
    pwr_factor = calc_power_factor(noise_est, cal_est, cal_temp, bandwidth)

    return noise_est, cal_est, pwr_factor

def altitude_from_range(range_m, azimuth_deg, elevation_deg, 
                        latitude_deg, base_altitude_m=0):
    """Calculate the altitude corresponding to a given range.
    
    Use beam azimuth and elevation and site location to calculate the
    altitude corresponding to given range, in meters.
    
    
    Arguments:
    
    range_m -- A number or 1-D numpy array of ranges in meters.
    
    Returns:
    
    A 1-D numpy array of altitudes in meters corresponding to the given 
        ranges.
    
    """
    # make sure range_m is a 1-D numpy array
    range_m = _np.array(range_m, copy = False, ndmin = 1).ravel()
    
    # set Earth spheroid constants (WGS84)
    a = 6378137.0 # Earth semi-major axis, meters
    b = 6356752.3142 # Earth semi-minor axis, meters
    esq = 1 - (b/a)**2 # Earth eccentricity squared
    
    # known values
    az = _math.radians(azimuth_deg) # beam azimuth angle
    el = _math.radians(elevation_deg) # beam elevation angle
    lat = _math.radians(latitude_deg) # site latitude
    h = base_altitude_m # site altitude
    
    # site position vector in IJK coordinates
    N = a / _math.sqrt(1 - esq*_math.sin(lat)**2)
    rho_vec = _np.matrix((   ((N + h)*_math.cos(lat), ),
                            (0, ),
                            ((N*(1 - esq) + h)*_math.sin(lat), ) ))
    
    # rotation matrix from SEZ to IJK
    rot_ang = _math.radians(-90) + lat
    rot = _np.matrix((   (_math.cos(rot_ang), 0, -_math.sin(rot_ang)),
                        (0, 1, 0),
                        (_math.sin(rot_ang), 0, _math.cos(rot_ang)) ))
    
    alt_m = _np.empty(range_m.shape[0])
    for k, r in enumerate(range_m):
        # position vector from site in SEZ coordinates
        r_vec = _np.matrix((  (-r*_math.cos(el)*_math.cos(az), ),
                                        (r*_math.cos(el)*_math.sin(az), ),
                                        (r*_math.sin(el), ) ))
        
        # rotate position vector from site into Earth-centered IJK coords
        r_vec = rot*r_vec
        
        # full position vector in IJK coordinates
        r_vec = rho_vec + r_vec
        X = r_vec.item(0)
        Y = r_vec.item(1)
        Z = r_vec.item(2)
        
        # apply Ferrari's solution to convert back to geodetic coordinates 
        # (partially, we want altitude only)
        R = _math.sqrt(X**2 + Y**2)
        Esq = a**2 - b**2
        F = 54.0*(b**2)*(Z**2)
        G = R**2 + (1- esq)*Z**2 - esq*Esq
        C = (esq**2)*F*(R**2) / _math.pow(G, 3)
        S = _math.pow(1 + C + _math.sqrt(C**2 + 2*C), 1./3.)
        P = F / ( 3*((S + 1/S + 1)**2)*(G**2) )
        Q = _math.sqrt(1 + 2*(esq**2)*P)
        R0 = (-(P*esq*R) / (1 + Q) + _math.sqrt(0.5*(a**2)*(1 + 1/Q) 
                - (P*(1 - esq)*Z**2) / (Q*(1 + Q)) - 0.5*P*(R**2)))
        U = _math.sqrt((R - esq*R0)**2 + Z**2)
        V = _math.sqrt((R - esq*R0)**2 + (1 - esq)*Z**2)
        alt = U*(1 - (b**2 / (a*V)))
        
        # alt = _math.sqrt(r**2 + a**2 - 2*r*a*_math.cos(el + _math.radians(90))) - a + h # simpler and almost as accurate (within meters)
        
        alt_m[k] = alt
    
    return alt_m