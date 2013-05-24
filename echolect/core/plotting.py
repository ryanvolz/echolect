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

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import subsectime

__all__ = ['rtiplot', 'implot', 'arrayticks', 'timeticks_helper', 
           'timeticks_array', 'timeticks']

def rtiplot(z, t, r, **kwargs):
    kwargs['xistime'] = True
    return implot(z, t, r, **kwargs)

def implot(z, x, y, xlabel=None, ylabel=None, title=None, colorbar=True,
           clabel=None, exact_ticks=True, xbins=10, ybins=10,
           xistime=False, yistime=False, **kwargs):
    imshowkwargs = dict(aspect='auto', interpolation='nearest', origin='lower')

    if exact_ticks:
        extent = (-0.5, x.shape[0] - 0.5,
                  -0.5, y.shape[0] - 0.5)
    else:
        xstart = float(x[0])
        xend = float(x[-1])
        xstep = (xend - xstart)/(x.shape[0] - 1)
        ystart = float(y[0])
        yend = float(y[-1])
        ystep = (yend - ystart)/(y.shape[0] - 1)
        extent = (xstart - xstep/2, xend + xstep/2,
                  ystart - ystep/2, yend + ystep/2)

    imshowkwargs.update(extent=extent)
    imshowkwargs.update(kwargs)

    img = plt.imshow(z.T, **imshowkwargs)
    ax = img.axes

    if colorbar:
        # add a colorbar that resizes with the image
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size=.125, pad=.1)
        cb = plt.colorbar(img, cax=cax)
        img.set_colorbar(cb, cax)
        if clabel is not None:
            cb.set_label(clabel)

    # title and labels
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # format and locate the ticks
    if exact_ticks:
        if xistime:
            timeticks_array(ax.xaxis, x, xbins)
        else:
            arrayticks(ax.xaxis, x, xbins)
        if yistime:
            timeticks_array(ax.yaxis, y, ybins)
        else:
            arrayticks(ax.yaxis, y, ybins)
    else:
        if xistime:
            timeticks(ax.xaxis, x[0], x[-1], xbins)
        if yistime:
            timeticks(ax.yaxis, y[0], y[-1], ybins)

    # make current axes ax (to make sure it is not cax)
    plt.sca(ax)
    # tight layout to make sure all labels fit, etc.
    plt.tight_layout()

    return img

def arrayticks(axis, arr, nbins=10):
    def tickformatter(x, pos=None):
        try:
            idx = int(round(x))
            val = arr[idx]
        except IndexError:
            s = ''
        else:
            if isinstance(val, float):
                s = '{0:.3f}'.format(val).rstrip('0').rstrip('.')
            else:
                s = str(val)
            if pos is None:
                s = s + ' ({0})'.format(idx)
        return s
    axis.set_major_formatter(mpl.ticker.FuncFormatter(tickformatter))
    axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=nbins, integer=True))

def timeticks_helper(ts, te):
    # get common string to label time axis
    tts = ts.to_datetime().timetuple()
    tte = te.to_datetime().timetuple()
    # compare year
    if tts[0] != tte[0]:
        tlabel = ''
        sfun = lambda ttick: str(ttick)
    # compare month
    elif tts[1] != tte[1]:
        tlabel = str(tts[0])
        sfun = lambda ttick: ttick.strftime('%b %d, %H:%M:%S.%f'
                                                      ).rstrip('0').rstrip('.')
    # compare day of month
    elif tts[2] != tte[2]:
        tlabel = ts.strftime('%B %Y')
        sfun = lambda ttick: ttick.strftime('%d, %H:%M:%S.%f'
                                                      ).rstrip('0').rstrip('.')
    # compare hour
    elif tts[3] != tte[3]:
        tlabel = ts.strftime('%b %d %Y')
        sfun = lambda ttick: ttick.strftime('%H:%M:%S.%f'
                                                      ).rstrip('0').rstrip('.')
    # compare minute
    elif tts[4] != tte[4]:
        tlabel = ts.strftime('%b %d %Y, %H:xx')
        sfun = lambda ttick: ttick.strftime('%M:%S.%f').rstrip('0').rstrip('.')
    # compare second
    elif tts[5] != tte[5]:
        tlabel = ts.strftime('%b %d %Y, %H:%M:xx')
        sfun = lambda ttick: ttick.strftime('%S.%f').rstrip('0').rstrip('.')
    else:
        tlabel = ts.strftime('%b %d %Y, %H:%M:%S.xx')
        sfun = lambda ttick: ttick.strftime('%f')

    return tlabel, sfun

def timeticks_array(axis, arr, nbins=10):
    tlabel, sfun = timeticks_helper(arr[0], arr[-1])
    currlabel = axis.get_label_text()
    if currlabel != '':
        tlabel = tlabel + '\n' + currlabel
    axis.set_label_text(tlabel)

    def tickformatter(x, pos=None):
        idx = int(round(x))
        try:
            val = arr[idx]
        except IndexError:
            s = ''
        else:
            s = sfun(val)
            if pos is None:
                s = s + ' ({0})'.format(idx)
        return s
    
    axis.set_major_formatter(mpl.ticker.FuncFormatter(tickformatter))
    axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=nbins, integer=True))

    # rotate x-axis tick labels so they can be read and fit together
    for label in axis.get_ticklabels():
        label.set_ha('left')
        label.set_rotation(-45)

def timeticks(axis, ts, te, nbins=10):
    tlabel, sfun = timeticks_helper(ts, te)
    currlabel = axis.get_label_text()
    if currlabel != '':
        tlabel = tlabel + '\n' + currlabel
    axis.set_label_text(tlabel)

    def tickformatter(x, pos=None):
        ttick = subsectime.MicroTime.from_seconds(x)
        s = sfun(ttick)
        return s

    axis.set_major_formatter(mpl.ticker.FuncFormatter(tickformatter))
    axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=nbins, integer=False))

    # rotate x-axis tick labels so they can be read and fit together
    for label in axis.get_ticklabels():
        label.set_ha('left')
        label.set_rotation(-45)