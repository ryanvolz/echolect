import numpy as np
import scipy as sp
import scipy.signal

from echolect.filtering import filtering
from echolect.estimators import interpolators

__all__ = ['complex_gaussian_noise', 'millstone']

def foo(tx):
    # for true Millstone simulation, would sample at 72 MHz, followed by a
    # cascade integration filter (CIC) to decimate partway down, followed by
    # a cascade-compensation FIR filter that decimates by 2, followed by a
    # a final FIR filter that decimates by 2 and meets the desired
    # passband/stopband requirements

    # until then, just use a lowpass filter
    n = 2
    txup = filtering.upsample(tx, n)
    print('Skipping low-pass filter!!!!!!')
    lowpass = np.zeros(101, dtype=txup.dtype) # need identity filter because of noise power manipulation below
    lowpass[50] = 1 # so filter delay is the same as lowpass
    #lowpass = sp.signal.remez(101, [0, 0.925/n, 1.075/n, 1], [1, 0], [1, 23], Hz=2)

    def run(noise, delay, freqshift, nsamples):
        # noise is the mean noise power
        # delay is number of samples (including fractional) which tx is to be delayed by
        # freqshift is the frequency shift in normalized frequency, -0.5 to 0.5

        # calculate delay amounts and upsample
        sampledelay, offset = divmod(n*delay, 1)
        h = interpolators.cubic_filter(n, offset)

        # interpolating digital signal as would be done in signal generator
        v = filtering.filter(h, txup) # integer delay of 2*n - 1 samples from filter

        # doppler shift, subtracting interpolation filter delay and offset so that
        # phase change at beginning of actual signal is 0
        w = np.exp(2*np.pi*1j*freqshift*(np.arange(len(v)) - (2*n - 1) - offset)/n)*v

        # pad signal to achieve desired delay AND ensure enough padding before and
        # after signal so that the filter does not exhibit edge effects
        x = np.zeros(n*nsamples + 2*(len(lowpass) - 1), w.dtype)
        # integer sample delay, accounting for delay already introduced by
        # interpolation filter and padding on either side by len(lowpass) - 1
        extradelay = int(sampledelay) - (2*n - 1) + (len(lowpass) - 1)
        if extradelay >= 0:
            xstart = extradelay # added delay
            wstart = 0
        else:
            xstart = 0
            wstart = -extradelay # reduced delay
        wend = len(x) + wstart - xstart # so view into w has at most same length as x[xstart:]
        wview = w[wstart:wend]
        xend = xstart + len(wview) # ensure view into x has same length as into w
        x[xstart:xend] = wview

        # noise
        # have to modify desired noise power by filter norm to account for how it reduces power
        # from reducing the bandwidth of the noise
        noise_factor = np.sqrt(noise)/np.linalg.norm(lowpass)
        samplenoise = noise_factor*(np.random.randn(len(x)) + 1j*np.random.randn(len(x)))/np.sqrt(2)
        y = x + samplenoise
        #return y[len(lowpass)-1:-len(lowpass):n] # values without filtering

        # receiver filtering
        filt = filtering.Filter(lowpass, len(y), y.dtype, measure=False)
        z = filt(y)
        z = z[filt.valid] # crop to "valid" filter results, undoing the extra padding we added earlier

        rx = z[50:-50:n] # remove filter delay and crop to nsamples

        return rx

    return run

def complex_gaussian_noise(shape, power, dtype=np.complex_):
    if isinstance(shape, int):
        shape = (shape,)
    noise = np.empty(shape, dtype=dtype)
    noise.real = np.sqrt(power/2.0)*np.random.randn(*shape)
    noise.imag = np.sqrt(power/2.0)*np.random.randn(*shape)
    return noise

def filter_block(taps):
    def apply_filter(x):
        filt = filtering.Filter(taps, len(x), x.dtype, measure=False)
        y = filt(x)
        y = y[filt.valid] # crop to "valid" filter results, undoing the extra padding we added earlier
        y = y # remove filter delay and crop to nsamples
        return y
    
    return apply_filter

def millstone(tx):
    return foo(tx)