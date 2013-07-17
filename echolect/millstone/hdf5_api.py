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
import h5py
import pandas

from echolect.core import subsectime
from echolect.core.indexing import find_index, slice_by_value, wrap_check_start, wrap_check_stop
from .read_hdf5 import *

__all__ = ['VoltageReader']

class VoltageReader(object):                                                           
    def __init__(self, fdir, ant='zenith'):
        self._fdir = fdir
        self._ant = ant

        files, times, nframes = map_files(fdir)
        self._files = files
        self._times = times
        self._nframes = nframes
        framenums = np.cumsum(np.hstack(([0], nframes)))
        totframes = framenums[-1]
        self._framenums = framenums[:-1]
        self._frametimes = [None]*len(files)

        firstfile = h5py.File(files[0], 'r')
        vlt = voltage(firstfile)
        self.shape = (totframes,) + vlt.shape[1:]
        self.sysmeta = read_system_metadata(firstfile)
        self.sigmeta = read_signal_metadata(firstfile)
        t = read_frame_time(firstfile)
        if self._ant == 'zenith':
            r = read_zenith_range(firstfile)
        else:
            r = read_misa_range(firstfile)
        self.r = r
        firstfile.close()

        self.ts = subsectime.PicoTimeDelta.from_seconds(self.sigmeta.signal_sampling_period)
        self.pri = (t[1] - t[0]).make_special()
        self.t0 = self._times[0].make_special()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        kwargs = {}
        if isinstance(key, tuple) and len(key) == 2:
            framekey, samplekey = key
            if isinstance(samplekey, int):
                kwargs['slc'] = slice(samplekey, samplekey + 1)
            elif isinstance(samplekey, slice):
                kwargs['slc'] = samplekey
            else:
                raise TypeError('key must be an integer, slice, or a 2-tuple of those')
        else:
            framekey = key

        if isinstance(framekey, int):
            start = framekey
        elif isinstance(framekey, slice):
            start = framekey.start
            stop = framekey.stop
            step = framekey.step

            if start is None:
                start = 0
            if step is None:
                step = 1

            kwargs['stop'] = stop
            kwargs['step'] = step
        else:
            raise TypeError('key must be an integer, slice, or a 2-tuple of those')

        return self.byframe(start, **kwargs)
            

    def __iter__(self):
        return self.stepper()
    
    def _findbyframe(self, f):
        if (f < 0) or (f > self.shape[0]):
            raise IndexError('frame index out of range')
        filenum = find_index(self._framenums, f)
        framenum = f - self._framenums[filenum]
        
        return filenum, framenum
    
    def _findbytime(self, t, cache=False):
        if t < self._times[0]:
            raise IndexError('time before beginning of data')
        filenum = find_index(self._times, t)
        times = self._read_frame_time(filenum, cache=cache)
        if t > times[-1]:
            raise IndexError('time after end of data')
        framenum = find_index(times, t)
        
        return filenum, framenum
    
    def _findbypulse(self, p, cache=False):
        t = self.t0 + p*self.pri
        
        return self._findbytime(t, cache)
    
    def _read_frame_time(self, fnum, cache=False, f=None):
        t = self._frametimes[fnum]
        
        if t is None:
            if f is None:
                with h5py.File(self._files[fnum], 'r') as f:
                    t = read_frame_time(f)
            else:
                t = read_frame_time(f)
            
            if cache:
                self._frametimes[fnum] = t
        
        return t

    def _read_file_frames(self, fnum, frameslice, sampleslice):
        with h5py.File(self._files[fnum], 'r') as f:
            vlt = read_voltage(f, (frameslice, sampleslice))
            # do not cache time access because speed savings is insignificant
            # compared to time for reading voltage, but memory use
            # can become a problem if we end up reading the whole data set
            t = self._read_frame_time(fnum, cache=False, f=f)[frameslice]

        r = self.r[sampleslice]

        data = pandas.DataFrame(vlt, t, r)
        data.index.name = 'time'
        data.columns.name = 'range'
        return data
    
    def _read_frames(self, filenumstart, framestart, filenumstop, framestop, 
                     framestep, sampleslice):
        # get file number of last frame that we *include*
        if framestop == 0:
            # we want to end on last frame of previous file
            filenumend = filenumstop - 1
            framestop = self._nframes[filenumend]
        else:
            # end frame is same file as stop frame
            filenumend = filenumstop
            
        if filenumstart == filenumend:
            return self._read_file_frames(filenumstart, 
                                          slice(framestart, framestop, framestep), 
                                          sampleslice)
        
        strt = framestart
        ret = []
        for fnum in xrange(filenumstart, filenumend + 1):
            if fnum == filenumend:
                fslc = slice(strt, framestop, framestep)
            else:
                fslc = slice(strt, None, framestep)
                # set start for next file (possibly != 0) based on framestep
                # step - ((nframes - strt) % step) == (strt - nframes) % step
                strt = (strt - self._nframes[fnum]) % framestep

            ret.append(self._read_file_frames(fnum, fslc, sampleslice))
        return pandas.concat(ret)

    def byframe(self, start, stop=None, step=1, slc=slice(None), nframes=1):
        start = wrap_check_start(self.shape[0], start)
        if stop is None:
            stop = start + step*nframes
        else:
            stop = wrap_check_stop(self.shape[0], stop)
        
        fnumstart, fstart = self._findbyframe(start)
        fnumstop, fstop = self._findbyframe(stop)
        
        return self._read_frames(fnumstart, fstart, fnumstop, fstop, step, slc)

    def bytime(self, tstart, tend=None, step=1, slc=slice(None), nframes=1):
        fnumstart, fstart = self._findbytime(tstart, cache=True)

        if tend is None:
            stop = self._framenums[fnumstart] + fstart + step*nframes
            fnumstop, fstop = self._findbyframe(stop)
        else:
            # file and frame which INCLUDES tend
            fnumstop, fend = self._findbytime(tend, cache=True)
            # add 1 to get frame to stop on (so fend is included)
            fstop = fend + 1

        return self._read_frames(fnumstart, fstart, fnumstop, fstop, step, slc)

    def bypulse(self, pstart, pstop=None, step=1, slc=slice(None), nframes=1):
        fnumstart, fstart = self._findbypulse(pstart, cache=True)
        
        if pstop is None:
            stop = self._framenums[fnumstart] + fstart + step*nframes
            fnumstop, fstop = self._findbyframe(stop)
        else:
            # pstop - 1 because we want to find last frame to INCLUDE
            # (otherwise lookup would fail if pstop == (# of pulses in data) )
            fnumstop, fend = self._findbypulse(pstop - 1, cache=True)
            # add 1 to get frame to stop on (so fend is included)
            fstop = fend + 1
    
        return self._read_frames(fnumstart, fstart, fnumstop, fstop, step, slc)
    
    def findbytime(self, t):
        filenum, framenum = self._findbytime(t, cache=True)
        
        return self._framenums[filenum] + framenum
    
    def findbypulse(self, p):
        filenum, framenum = self._findbypulse(p, cache=True)
        
        return self._framenums[filenum] + framenum
    
    def lookuptime(self, frame):
        filenum, framenum = self._findbyframe(frame)
        
        t_file = self._read_frame_time(filenum, cache=True)
        t = t_file[framenum]
        
        return t
    
    def lookuppulse(self, frame):
        t = self.lookuptime(frame)
        p = (t - self.t0) // self.pri
        
        return p

    def stepper(self, start=0, stop=None, step=1, slc=slice(None), block_size=1,
                block_overlap=0, approx_read_size=1000):
        """approx_read_size is maximum number of frames to read from file at once, with
        actual amount satisfying (read_size - block_size) % (block_size - block_overlap) == 0"""
        if (start < -self.shape[0]) or (start >= self.shape[0]):
            raise IndexError('start index out of range')
        if start < 0:
            start = start % self.shape[0]

        if stop is None:
            stop = len(self)
        elif (stop <= -self.shape[0]) or (stop > self.shape[0]):
            raise IndexError('stop index out of range')
        elif stop <= 0:
            # let stop == 0 be shorthand for stop == self.shape[0],
            # i.e. including all frames to the end
            stop = (stop - 1) % self.shape[0] + 1

        block_step = block_size - block_overlap
        if block_size < approx_read_size:
            read_size = (approx_read_size - block_size)//block_step*block_step + block_size
        else:
            read_size = block_size

        read_step = step*(read_size - block_overlap)
        for fstart in xrange(start, stop, read_step):
            fstop = min(fstart + step*read_size, stop)
            data = self.byframe(fstart, fstop, step, slc)
            act_read_size = len(data)
            # only want to start a block with the final block_overlap frames if there is
            # subsequent data, hence the range stop of (act_read_size - block_overlap)
            for bstart in xrange(0, act_read_size - block_overlap, block_step):
                bend = min(bstart + block_size, act_read_size)
                yield data.irow(slice(bstart, bend))

    def rangeslice(self, low=None, high=None):
        return slice_by_value(self.r, low, high)