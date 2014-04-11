#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from scipy import constants
import h5py
import collections
import os
import glob

from echolect.core import subsectime
from echolect.core.indexing import find_index, slice_by_value

__all__ = ['map_files', 'find_pulse', 'voltage', 
           'read_voltage', 'read_frame_time', 'read_sample_time', 
           'read_misa_range', 'read_zenith_range', 
           'read_misa_power', 'read_zenith_power', 'read_power_time',
           'read_system_metadata', 'read_signal_metadata']

def map_files(fdir, ext='hdf5'):
    files = glob.glob(os.path.join(fdir, '*.' + ext))
    files.sort()
    files = np.asarray(files)

    times = np.zeros(len(files), 'datetime64[ns]')
    nframes = np.zeros(len(files), np.int64)
    for k, fpath in enumerate(files):
        with h5py.File(fpath, 'r') as h5file:
            secs = h5file['rf_signal/data/frame_timestamp_seconds'][0].astype('datetime64[s]')
            psecs = h5file['rf_signal/data/frame_timestamp_picoseconds'][0].astype('timedelta64[ps]')
            # convert picoseconds to nanoseconds (losing precision) because
            # a picosecond datetime64 cannot store useful times
            nsecs = psecs.astype('timedelta64[ns]')
            time = secs + nsecs
            times[k] = time
            nframes[k] = h5file['rf_signal/data/frame_timestamp_seconds'].shape[0]

    return files, times, nframes

def find_pulse(time, times, files):
    if time < times[0]:
        raise IndexError('time before beginning of data')
    fnum = find_index(times, time)

    with h5py.File(files[fnum], 'r') as f:
        t = read_frame_time(f)
    if time > t[-1]:
        raise IndexError('time after end of data')
    pnum = find_index(t, time)

    return fnum, pnum

class voltage(object):
    def __init__(self, h5file):
        self.real = h5file['rf_signal/data/real_samples']
        self.imag = h5file['rf_signal/data/imaginary_samples']
        self.shape = self.real.shape

    def __getitem__(self, key):
        real = self.real[key]
        imag = self.imag[key]
        vlt = np.empty(real.shape, np.complex64)
        vlt.real[:] = real
        vlt.imag[:] = imag

        return vlt

def read_voltage(h5file, key=slice(None)):
    vlt_o = voltage(h5file)
    return vlt_o[key]

def read_frame_time(h5file, slc=slice(None)):
    secs = h5file['rf_signal/data/frame_timestamp_seconds'][slc].astype('datetime64[s]')
    psecs = h5file['rf_signal/data/frame_timestamp_picoseconds'][slc].astype('timedelta64[ps]')
    # convert picoseconds to nanoseconds (losing precision) because
    # a picosecond datetime64 cannot store useful times
    nsecs = psecs.astype('timedelta64[ns]')
    t = secs + nsecs
    return t

def read_sample_time(h5file, slc=slice(None)):
    ns = h5file['rf_signal/data/real_samples'].shape[1]
    sigmeta = read_signal_metadata(h5file)
    ts = sigmeta.signal_sampling_period
    return np.arange(*slc.indices(ns))*ts

def read_misa_range(h5file, slc=slice(None)):
    sysmeta = read_system_metadata(h5file)
    td = sysmeta.misa_delay
    ts = read_sample_time(h5file, slc)
    tr = (ts - td)/np.timedelta64(1, 's') # total true delay in seconds (float)
    r = tr*constants.c/2

    return r

def read_zenith_range(h5file, slc=slice(None)):
    sysmeta = read_system_metadata(h5file)
    td = sysmeta.zenith_delay
    ts = read_sample_time(h5file, slc)
    tr = (ts - td)/np.timedelta64(1, 's') # total true delay in seconds (float)
    r = tr*constants.c/2

    return r

def read_misa_power(h5file):
    p = h5file['millstone_system_state/data/transmitter_power/misa_power']
    return p[:]

def read_zenith_power(h5file):
    p = h5file['millstone_system_state/data/transmitter_power/zenith_power']
    return p[:]

def read_power_time(h5file):
    secs = h5file['millstone_system_state/data/transmitter_power/utc_second'
                  ][:].astype('datetime64[s]')
    psecs = h5file['millstone_system_state/data/transmitter_power/utc_picosecond'
                   ][:].astype('timedelta64[ps]')
    # convert picoseconds to nanoseconds (losing precision) because
    # a picosecond datetime64 cannot store useful times
    nsecs = psecs.astype('timedelta64[ns]')
    t = secs + nsecs
    return t

def read_system_metadata(h5file):
    attrs = h5file['millstone_system_state/metadata'].attrs
    attrs_dict = dict(attrs)
    # delay and delay_error are floats in seconds, convert to timedelta with ns precision
    for attr in ['misa_delay', 'misa_delay_error', 'zenith_delay', 'zenith_delay_error']:
        secs = attrs[attr]
        t = np.round(1e9*secs).astype('timedelta64[ns]')
        attrs_dict[attr] = t
    mdo = collections.namedtuple('MHMetadata', attrs.keys())
    md = mdo(**attrs_dict)
    return md

def read_signal_metadata(h5file):
    attrs = h5file['rf_signal/metadata/signal'].attrs
    attrs_dict = dict(attrs)
    # signal_sampling_period is a float in nanoseconds, convert to timedelta
    ts = np.round(attrs['signal_sampling_period']).astype('timedelta64[ns]')
    attrs_dict['signal_sampling_period'] = ts
    mdo = collections.namedtuple('SignalMetadata', attrs.keys())
    md = mdo(**attrs_dict)
    return md