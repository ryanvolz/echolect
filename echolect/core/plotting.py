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
from mpl_toolkits import axes_grid1

from . import subsectime

__all__ = ['rtiplot', 'implot', 'colorbar', 'make_axes_fixed', 
           'arrayticks', 'timeticks_helper', 'timeticks_array', 'timeticks']

def rtiplot(z, t, r, **kwargs):
    kwargs['xistime'] = True
    return implot(z, t, r, **kwargs)

def implot(z, x, y, xlabel=None, ylabel=None, title=None, cbar=True,
           clabel=None, exact_ticks=True, xbins=10, ybins=10,
           xistime=False, yistime=False, ax=None, **kwargs):
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

    if ax is None:
        ax = plt.gca()
    img = ax.imshow(z.T, **imshowkwargs)

    if cbar:
        cb = colorbar(img, position='right', size=0.125, pad=0.1, label=clabel)

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

    return img

def colorbar(img, position, size, pad=None, label=None, **kwargs):
    # add a colorbar that resizes with the image
    ax = img.axes
    fig = ax.get_figure()
    # delete any existing colorbar
    if hasattr(ax, 'colorbar'):
        oldcb, oldcax, oldloc = ax.colorbar
        # delete from figure
        fig.delaxes(oldcax)
        # restore axes to original divider
        ax.set_axes_locator(oldloc)
        del ax.colorbar, oldcb, oldcax, oldloc
        
    loc = ax.get_axes_locator()
    if loc is None:
        # create axes divider that follows size of original axes (the subplot's area)
        # colorbar will be added to this divider to fit in the original axes area
        axdiv = axes_grid1.axes_divider.AxesDivider(ax)
    else:
        origdiv = loc._axes_divider
        # make new axes divider to add colorbar to (since we can't presume to modify original)
        hsize = axes_grid1.Size.AddList(origdiv.get_horizontal()[loc._nx:loc._nx1])
        vsize = axes_grid1.Size.AddList(origdiv.get_vertical()[loc._ny:loc._ny1])
        axdiv = axes_grid1.axes_divider.AxesDivider(ax,
                                                    xref=hsize, 
                                                    yref=vsize)
        axdiv.set_aspect(origdiv.get_aspect())
        axdiv.set_anchor(origdiv.get_anchor())
        axdiv.set_locator(loc)
    # place the axes in the new divider so that it is sized to make room for colorbar
    axloc = axdiv.new_locator(0, 0)
    ax.set_axes_locator(axloc)
    cax = axdiv.append_axes(position, size=size, pad=pad)
    cb = fig.colorbar(img, cax=cax, ax=ax, **kwargs)
    # add colorbar reference to image
    img.set_colorbar(cb, cax)
    # add colorbar reference to axes (so we can remove it if called again, see above)
    ax.colorbar = (cb, cax, loc)
    if label is not None:
        cb.set_label(label)
    
    # make current axes ax (to make sure it is not cax)
    fig.sca(ax)
    
    return cb

def make_axes_fixed(ax, xinches, yinches):
    div = axes_grid1.axes_divider.AxesDivider(ax,
                                              xref=axes_grid1.Size.Fixed(xinches),
                                              yref=axes_grid1.Size.Fixed(yinches))
    origloc = ax.get_axes_locator()
    if origloc is not None:
        div.set_locator(origloc)
    loc = div.new_locator(0, 0)
    ax.set_axes_locator(loc)

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