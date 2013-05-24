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
import glumpy
import threading
import time

__all__ = ['Video']

class Video(object):
    def __init__(self, gen, npulses=1000, vmin=0, vmax=30, winsize=(1450, 800)):
        self.gen = gen
        self.npulses = npulses

        self.figure = glumpy.Figure(winsize)
        self.figure.event('on_mouse_motion')(self.on_mouse_motion)
        self.figure.event('on_mouse_scroll')(self.on_mouse_scroll)
        self.figure.event('on_key_press')(self.on_key_press)
        self.figure.timer(30.0)(self.draw) # 30 works, more seems to result in skips

        self.pause = [False, False]
        self.stop = False
        self.loc_scale = [0, 0, 1]

        # initialize image
        dat = self.gen.next()
        self.Z = np.zeros((len(dat), npulses), dtype=np.float32)
        self.Z[:, 0] = dat
        self.I = glumpy.Image(self.Z, interpolation='nearest',
                              colormap=glumpy.colormap.IceAndFire, origin='lower',
                              vmin=vmin, vmax=vmax)

    def on_mouse_motion(self, x, y, dx, dy):
        zoom = self.loc_scale[2]
        x = x/float(self.figure.width)
        y = y/float(self.figure.height)
        x = min(max(x,0),1)
        y = min(max(y,0),1)
        self.loc_scale[0] = x*self.figure.width*(1-zoom)
        self.loc_scale[1] = y*self.figure.height*(1-zoom)
        self.figure.redraw()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        zoom = self.loc_scale[2]
        if scroll_y > 0:
            zoom *= 1.25
        elif scroll_y < 0:
            zoom /= 1.25
        self.loc_scale[2] = min(max(zoom, 1), 20)
        self.on_mouse_motion(x, y, 0, 0)

    def draw(self, dt):
        self.figure.clear()
        self.I.update()
        x,y,s = self.loc_scale
        self.I.draw(x, y, 0, s*self.figure.width, s*self.figure.height)
        self.figure.redraw()

    def on_key_press(self, key, modifiers):
        if key == glumpy.window.key.P or key == glumpy.window.key.SPACE:
            self.pause[0] = not self.pause[0]
            self.pause[1] = False
            return True
        if key == glumpy.window.key.Q or key == glumpy.window.key.ESCAPE:
            self.stop = True

    def _plottingthread(self):
        time.sleep(1)
        block_size = self.npulses
        w = self.Z.shape[1]
        for pulse_num, dat in enumerate(self.gen):
            bn = (pulse_num + 1) % w # offset by 1 because we already read first pulse in __init__
            self.Z[:, bn] = np.flipud(dat)

            while self.pause[0]:
                if not self.pause[1]:
                    print pulse_num + 1 # offset by 1 because we already read first pulse in __init__
                    self.pause[1] = True
                time.sleep(1)
            
            if self.stop:
                break

    def play(self):
        t = threading.Thread(target=self._plottingthread)
        t.start()
        self.figure.show()
        t.join()
