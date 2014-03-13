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
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import pandas

from echolect.tools.time import datetime_from_float, datetime_to_float, timestamp_strftime

__all__ = ['rtiplot', 'implot', 'colorbar', 'make_axes_fixed', 
           'arrayticks', 'timeticks_helper', 'timeticks_array', 'timeticks']

def rtiplot(z, t, r, **kwargs):
    kwargs['xistime'] = True
    return implot(z, t, r, **kwargs)

def implot(z, x, y, xlabel=None, ylabel=None, title=None, 
           exact_ticks=True, xbins=10, ybins=10, 
           xistime=False, yistime=False, 
           cbar=True, clabel=None, cposition='right', 
           csize=0.125, cpad=0.1, cbins=None, 
           ax=None, pixelaspect=None, 
           **kwargs):
    imshowkwargs = dict(aspect='auto', interpolation=None, origin='lower')
    
    # asarray needed to convert pandas' DatetimeIndex to datetime64
    if xistime:
        x = np.asarray(x)
    if yistime:
        y = np.asarray(y)

    if exact_ticks:
        extent = (-0.5, x.shape[0] - 0.5,
                  -0.5, y.shape[0] - 0.5)
    else:
        if xistime:
            # use day of the given first time as epoch
            xepoch = x[0].astype('datetime64[D]')
            x_float = datetime_to_float(x, epoch=xepoch)
            xstart = x_float[0]
            xend = x_float[-1]
        else:
            xstart = x[0]
            xend = x[-1]
        xstep = (xend - xstart)/(x.shape[0] - 1)
        if yistime:
            # use day of the given first time as epoch
            yepoch = y[0].astype('datetime64[D]')
            y_float = datetime_to_float(y, epoch=yepoch)
            ystart = y_float[0]
            yend = y_float[-1]
        else:
            ystart = y[0]
            yend = y[-1]
        ystep = (yend - ystart)/(y.shape[0] - 1)
        extent = (xstart - xstep/2.0, xend + xstep/2.0,
                  ystart - ystep/2.0, yend + ystep/2.0)
    imshowkwargs.update(extent=extent)

    if pixelaspect is not None:
        box_aspect = abs((extent[1] - extent[0])/(extent[3] - extent[2]))
        arr_aspect = float(z.shape[0])/z.shape[1]
        aspect = box_aspect/arr_aspect/pixelaspect
        imshowkwargs.update(aspect=aspect)
    
    imshowkwargs.update(kwargs)

    if ax is None:
        ax = plt.gca()
    img = ax.imshow(z.T, **imshowkwargs)

    if cbar:
        cb = colorbar(img, position=cposition, size=csize, pad=cpad, 
                      label=clabel, bins=cbins)

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
            timeticks(ax.xaxis, x[0], x[-1], xepoch, xbins)
        else:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=xbins, integer=False))
        if yistime:
            timeticks(ax.yaxis, y[0], y[-1], yepoch, ybins)
        else:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=ybins, integer=False))

    return img

def colorbar(img, position='right', size=0.125, pad=0.1, label=None, bins=None, 
             **kwargs):
    # add a colorbar that resizes with the image
    ax = img.axes
    fig = ax.get_figure()
    # delete any existing colorbar
    if img.colorbar is not None:
        oldcb = img.colorbar
        oldcax = oldcb.ax
        # delete colorbar axes from figure
        fig.delaxes(oldcax)
        # restore axes to original divider
        if hasattr(img, 'axesloc'):
            origloc = img.axesloc
            ax.set_axes_locator(origloc)
        # delete colorbar reference
        img.colorbar = None
        del oldcb, oldcax, origloc
    
    # save original locator as attribute (so we can delete colorbar, see above)
    origloc = ax.get_axes_locator()
    img.axesloc = origloc
    # make axes locatable so we can use the resulting divider to add a colorbar
    axdiv = make_axes_locatable(ax)
    
    # create colorbar and its axes
    cax = axdiv.append_axes(position, size=size, pad=pad)
    if position in ('bottom', 'top'):
        orientation = 'horizontal'
    else:
        orientation = 'vertical'
    cb = fig.colorbar(img, cax=cax, ax=ax, orientation=orientation, **kwargs)
    # add colorbar reference to image
    img.colorbar = cb
    if label is not None:
        cb.set_label(label)
    
    # adjust number of tick bins if desired
    if bins is not None:
        tickloc = mpl.ticker.MaxNLocator(nbins=bins, integer=False)
        if position in ('bottom', 'top'):
            cax.xaxis.set_major_locator(tickloc)
        elif position in ('left', 'right'):
            cax.yaxis.set_major_locator(tickloc)
    
    # make current axes ax (to make sure it is not cax)
    fig.sca(ax)
    
    return cb

def make_axes_fixed(ax, xinches, yinches):
    # make a fixed size divider, located using existing locator if necessary
    div = axes_grid1.axes_divider.AxesDivider(ax,
                                              xref=axes_grid1.Size.Fixed(xinches),
                                              yref=axes_grid1.Size.Fixed(yinches))
    origloc = ax.get_axes_locator()
    if origloc is not None:
        div.set_locator(origloc)
    
    # place the axes in the new divider
    loc = div.new_locator(0, 0)
    ax.set_axes_locator(loc)
    
    return div

def make_axes_locatable(ax):
    # custom make_axes_locatable to fix:
    #  - case when axes is already locatable and we want to work within 
    #    existing divider
    #  - case when axes has a specified aspect ratio other than 1 or auto

    origloc = ax.get_axes_locator()
    if origloc is None:
        # create axes divider that follows size of original axes (the subplot's area)
        
        # default AxesDivider has relative lengths in data units,
        # i.e. if the x-axis goes from x0 to x1, then the horizontal size of
        # the axes has a relative length of (x1 - x0)
        
        # when the axes' aspect ratio is set, however, the default axes divider
        # scales the divider size so that aspect ratio is fixed at 1 regardless
        # of the specified aspect ratio
        
        # in order to make aspect ratios other than 1 work, we need to scale
        # the relative length for the y-axis by the aspect ratio
        
        # set relative length for x-axis based on data units of ax
        hsize = axes_grid1.Size.AxesX(ax)
        
        # set relative length for y-axis based on aspect-scaled data units
        aspect = ax.get_aspect()
        if aspect == 'equal':
            aspect = 1
        
        if aspect == 'auto':
            vsize = axes_grid1.Size.AxesY(ax)
        else:
            vsize = axes_grid1.Size.AxesY(ax, aspect=aspect)
        
        div = axes_grid1.axes_divider.AxesDivider(ax, xref=hsize, yref=vsize)
    else:
        origdiv = origloc._axes_divider
        # make new axes divider (since we can't presume to modify original)
        hsize = axes_grid1.Size.AddList(
                    origdiv.get_horizontal()[origloc._nx:origloc._nx1])
        vsize = axes_grid1.Size.AddList(
                    origdiv.get_vertical()[origloc._ny:origloc._ny1])
        div = axes_grid1.axes_divider.AxesDivider(ax, xref=hsize, yref=vsize)
        div.set_aspect(origdiv.get_aspect())
        div.set_anchor(origdiv.get_anchor())
        div.set_locator(origloc)
    
    # place the axes in the new divider
    loc = div.new_locator(0, 0)
    ax.set_axes_locator(loc)
    
    return div

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
    tts = ts.timetuple()
    tte = te.timetuple()
    # compare year
    if tts[0] != tte[0]:
        tlabel = ''
        sfun = lambda ttick: timestamp_strftime(
                              ttick,
                              '%Y-%m-%d %H:%M:%S.%f').rstrip('0').rstrip('.')
    # compare month
    elif tts[1] != tte[1]:
        tlabel = str(tts[0])
        sfun = lambda ttick: timestamp_strftime(
                              ttick,
                              '%b %d, %H:%M:%S.%f').rstrip('0').rstrip('.')
    # compare day of month
    elif tts[2] != tte[2]:
        tlabel = timestamp_strftime(ts, '%B %Y')
        sfun = lambda ttick: timestamp_strftime(
                              ttick, 
                              '%d, %H:%M:%S.%f').rstrip('0').rstrip('.')
    # compare hour
    elif tts[3] != tte[3]:
        tlabel = timestamp_strftime(ts, '%b %d %Y')
        sfun = lambda ttick: timestamp_strftime(
                              ttick,
                              '%H:%M:%S.%f').rstrip('0').rstrip('.')
    # compare minute
    elif tts[4] != tte[4]:
        tlabel = timestamp_strftime(ts, '%b %d %Y, %H:xx')
        sfun = lambda ttick: timestamp_strftime(
                              ttick,
                              '%M:%S.%f').rstrip('0').rstrip('.')
    # compare second
    elif tts[5] != tte[5]:
        tlabel = timestamp_strftime(ts, '%b %d %Y, %H:%M:xx')
        sfun = lambda ttick: timestamp_strftime(
                              ttick,
                              '%S.%f').rstrip('0').rstrip('.')
    else:
        tlabel = timestamp_strftime(ts, '%b %d %Y, %H:%M:%S.xx')
        sfun = lambda ttick: timestamp_strftime(
                              ttick,
                              '%f')

    return tlabel, sfun

def timeticks_array(axis, arr, nbins=10):
    # convert time array to pandas DatetimeIndex, 
    # which returns Timestamp objects when indexed
    arr_idx = pandas.DatetimeIndex(arr)
    
    tlabel, sfun = timeticks_helper(arr_idx[0], arr_idx[-1])
    currlabel = axis.get_label_text()
    if currlabel != '':
        tlabel = tlabel + '\n' + currlabel
    axis.set_label_text(tlabel)

    def tickformatter(x, pos=None):
        idx = int(round(x))
        try:
            val = arr_idx[idx]
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

def timeticks(axis, ts, te, floatepoch, nbins=10):
    # convert ts and te to Timestamp objects
    ts = pandas.Timestamp(ts)
    te = pandas.Timestamp(te)
    
    tlabel, sfun = timeticks_helper(ts, te)
    currlabel = axis.get_label_text()
    if currlabel != '':
        tlabel = tlabel + '\n' + currlabel
    axis.set_label_text(tlabel)

    def tickformatter(x, pos=None):
        ttick = pandas.Timestamp(datetime_from_float(x, 'ns', epoch=floatepoch).item())
        s = sfun(ttick)
        return s

    axis.set_major_formatter(mpl.ticker.FuncFormatter(tickformatter))
    axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=nbins, integer=False))

    # rotate x-axis tick labels so they can be read and fit together
    for label in axis.get_ticklabels():
        label.set_ha('left')
        label.set_rotation(-45)