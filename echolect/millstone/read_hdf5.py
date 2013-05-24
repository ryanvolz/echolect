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

    times = []
    nframes = []
    for fpath in files:
        with h5py.File(fpath, 'r') as h5file:
            time = subsectime.SubSecTime.nofix(
                      h5file['rf_signal/data/frame_timestamp_seconds'][0],
                      h5file['rf_signal/data/frame_timestamp_picoseconds'][0],
                      1000000000000)
            times.append(time)
            nframes.append(h5file['rf_signal/data/frame_timestamp_seconds'].shape[0])
    times = np.asarray(times)
    nframes = np.asarray(nframes)

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
    t = np.vectorize(subsectime.SubSecTime.nofix)(
                     h5file['rf_signal/data/frame_timestamp_seconds'][slc],
                     h5file['rf_signal/data/frame_timestamp_picoseconds'][slc],
                     1000000000000)
    return t

def read_sample_time(h5file, slc=slice(None)):
    ns = h5file['rf_signal/data/real_samples'].shape[1]
    ts = subsectime.SubSecTimeDelta(
           0,
           1000*h5file['rf_signal/metadata/signal'].attrs['signal_sampling_period'],
           1000000000000) # signal_sampling_period is in nanoseconds
    return np.arange(*slc.indices(ns))*ts

def read_misa_range(h5file, slc=slice(None)):
    delay = subsectime.SubSecTimeDelta.from_seconds(
                h5file['millstone_system_state/metadata'].attrs['misa_delay'],
                1000000000000) # delay is in picoseconds
    ts = read_sample_time(h5file, slc)
    tr = (ts - delay).astype(np.float_)
    r = tr*constants.c/2

    return r

def read_zenith_range(h5file, slc=slice(None)):
    delay = subsectime.SubSecTimeDelta.from_seconds(
               h5file['millstone_system_state/metadata'].attrs['zenith_delay'],
               1000000000000) # delay is in picoseconds
    ts = read_sample_time(h5file, slc)
    tr = (ts - delay).astype(np.float_)
    r = tr*constants.c/2

    return r

def read_misa_power(h5file):
    p = h5file['millstone_system_state/data/transmitter_power/misa_power']
    return p[:]

def read_zenith_power(h5file):
    p = h5file['millstone_system_state/data/transmitter_power/zenith_power']
    return p[:]

def read_power_time(h5file):
    t = np.vectorize(subsectime.SubSecTime.nofix)(
     h5file['millstone_system_state/data/transmitter_power/utc_second'][:],
     h5file['millstone_system_state/data/transmitter_power/utc_picosecond'][:],
     1000000000000)
    return t

def read_system_metadata(h5file):
    attrs = h5file['millstone_system_state/metadata'].attrs
    mdo = collections.namedtuple('MHMetadata', attrs.keys())
    md = mdo(**attrs)
    return md

def read_signal_metadata(h5file):
    attrs = dict(h5file['rf_signal/metadata/signal'].attrs)
    attrs['signal_sampling_period'] = 1e-9*attrs['signal_sampling_period'] # convert from nanoseconds
    mdo = collections.namedtuple('SignalMetadata', attrs.keys())
    md = mdo(**attrs)
    return md