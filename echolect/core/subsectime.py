#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

"""A module for working with dates/times that have precise sub-second resolution.


Exported Classes:

FixedTimezone -- Fixed timezone defined by timezone offset and DST flag.

SubSecTimeDelta -- Class representing relative times with sub-second resolution.

SubSecTime -- Class representing absolute times with sub-second resolution.


@author: Ryan Volz

"""
from __future__ import division as _division
import datetime as _datetime
import calendar as _calendar
import warnings as _warnings
import math as _math
import re as _re

##TODO
# __slots__ to save space per object?


class FixedTimezone(_datetime.tzinfo):
    """Fixed timezone defined by timezone offset and DST flag."""

    def __init__(self, tz_offset=0, DST=False, name=None):
        """Initialize fixed offset timezone.


        Arguments:

        tz_offset -- Timezone hours from UTC (+ for East, - for West).

        DST -- Boolean indicating whether Daylight Savings Time offset
            should be included.

        name -- String giving the timezone name.

        """
        self._offset = _datetime.timedelta(hours=tz_offset + DST)
        self._DST = DST
        self._name = name

    def utcoffset(self, dt):
        return self._offset

    def tzname(self, dt):
        return self._name

    def dst(self, dt):
        return _datetime.timedelta(hours=self._DST)


class SubSecTimeDelta(object):
    """A class for representing relative times with sub-second resolution.


    Public Attributes:

    seconds -- Integer giving the number of seconds.

    subseconds -- Integer giving the number of subseconds.


    Public Methods:

    total_seconds -- Return the time as a float in seconds.

    total_subseconds -- Return the total time as an integer in subseconds.

    """
    def __new__(cls, secs=0, ssecs=0, factor=1000000):
        """Initialize SubSecTimeDelta from seconds and subseconds integers.


        Arguments:

        secs -- Integer giving the number of seconds.

        ssecs -- Integer giving the number of subseconds.

        factor -- Integer giving conversion factor from subseconds to seconds.

        The total time is given by (secs + ssecs/factor) seconds.

        """
        self = cls.nofix(int(secs), int(ssecs), int(factor))

        # make sure ssecs is valid (absolute value less than factor)
        secs, ssecs = divmod(self._ssecs, self._factor)
        secs = secs + self._secs

        # make nsecs lie between 0 and factor-1 (inclusive) if secs > 0
        # and between -(factor-1) and 0 (inclusive) if secs < 0
        # (nsecs is already >= 0 from the divmod above)
        if secs < 0 and ssecs > 0:
            secs += 1
            ssecs -= self._factor

        self._secs = secs
        self._ssecs = ssecs

        return self

    @classmethod
    def nofix(cls, secs=0, ssecs=0, factor=1000000):
        self = object.__new__(cls)

        self._secs = secs
        self._ssecs = ssecs
        self._factor = factor

        return self

    @classmethod
    def from_seconds(cls, seconds, factor=1000000):
        factor = int(factor)
        if seconds >= 0:
            secs = int(seconds)
            ssecs = int(round((seconds % 1)*factor))
            return cls.nofix(secs=secs, ssecs=ssecs, factor=factor)
        else:
            secs = -int(-seconds)
            ssecs = -int(round((-seconds % 1)*factor))
            return cls.nofix(secs=secs, ssecs=ssecs, factor=factor)

    @property
    def seconds(self):
        """Get seconds."""
        return self._secs

    @property
    def subseconds(self):
        """Get subseconds."""
        return self._ssecs

    @property
    def factor(self):
        """Get subsecond factor."""
        return self._factor

    def total_seconds(self):
        """Return the time as a float in seconds."""
        return self._secs + self._ssecs/self._factor

    def total_subseconds(self):
        """Return the total time as an integer in subseconds."""
        return self._ssecs + self._secs*self._factor

    def make_special(self):
        factor = self._factor
        if factor == 1000:
            return MilliTimeDelta.nofix(self._secs, self._ssecs, factor)
        elif factor == 1000000:
            return MicroTimeDelta.nofix(self._secs, self._ssecs, factor)
        elif factor == 1000000000:
            return NanoTimeDelta.nofix(self._secs, self._ssecs, factor)
        elif factor == 1000000000000:
            return PicoTimeDelta.nofix(self._secs, self._ssecs, factor)
        return self

    def change_factor(self, factor):
        """Change the subsecond factor to the one provided."""
        if factor == self._factor:
            return self
        changed = SubSecTimeDelta(self._secs, self._ssecs*factor//self._factor,
                                  factor)
        return changed

    def equalize_factors(self, other):
        """Equalize factors for two SubSecTimeDelta objects to their maximum."""
        if self._factor == other._factor:
            return self, other
        max_factor = max(self._factor, other._factor)
        return self.change_factor(max_factor), other.change_factor(max_factor)

    def fractional_digits(self, precision=12):
        dec_exp = _math.log10(self._factor)
        if dec_exp.is_integer():
            fstr = '{0:0' + str(int(dec_exp)) + '}'
            return fstr.format(self._ssecs).rstrip('0')
        else:
            fstr = '{0:.' + str(precision) + 'f}'
            fracstring = fstr.format(float(self._ssecs)/self._factor
                                    ).split('.')[1].rstrip('0')
        if fracstring == '':
            fracstring = '0'
        return fracstring

    def __repr__(self):
        return '{0}({1}, {2}, {3})'.format(self.__class__.__name__, self._secs,
                                           self._ssecs, self._factor)

    def __str__(self):
        s = str(self._secs) + '.' + self.fractional_digits()
        # remove decimal point if no fractional digits
        s = s.rstrip('0').rstrip('.')
        return s

    def __float__(self):
        return self.total_seconds()

    def __add__(self, other):
        if isinstance(other, SubSecTimeDelta):
            s, o = self.equalize_factors(other)
            secs = s._secs + o._secs
            ssecs = s._ssecs + o._ssecs
            if isinstance(other, SubSecTime):
                return type(other)(secs, ssecs, o._factor)
            return type(self)(secs, ssecs, s._factor)
        elif isinstance(other, (int, long, float)):
            # assume other represents seconds
            return self + SubSecTimeDelta.from_seconds(other, factor=self._factor)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        return SubSecTimeDelta.__add__(self, -other)

    def __rsub__(self, other):
        return SubSecTimeDelta.__add__(-self, other)

    def __pos__(self):
        return self

    def __neg__(self):
        return type(self).nofix(-self._secs, -self._ssecs, self._factor)

    def __abs__(self):
        return type(self).nofix(abs(self._secs), abs(self._ssecs), self._factor)

    def __mul__(self, other):
        if isinstance(other, (int, long)):
            return type(self)(self._secs*other, self._ssecs*other, self._factor)
        if isinstance(other, float):
            a, b = other.as_integer_ratio()
            return self * a / b
        return NotImplemented

    __rmul__ = __mul__

    def __divmod__(self, other):
        if isinstance(other, SubSecTimeDelta):
            s, o = self.equalize_factors(other)
            div, rem = divmod(s.total_subseconds(),
                              o.total_subseconds())
            mod = type(other).nofix(0, rem, o.factor)
            return div, mod
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, SubSecTimeDelta):
            s, o = self.equalize_factors(other)
            return s.total_subseconds() // o.total_subseconds()
        elif isinstance(other, (int, long)):
            return type(self)(0, self.total_subseconds() // other,
                              self._factor)
        return NotImplemented

    def __mod__(self, other):
        if isinstance(other, SubSecTimeDelta):
            s, o = self.equalize_factors(other)
            return type(other)(0, (s.total_subseconds()
                                   % o.total_subseconds()), o._factor)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, SubSecTimeDelta):
            s, o = self.equalize_factors(other)
            return s.total_subseconds() / o.total_subseconds()
        elif isinstance(other, (int, long, float)):
            return type(self)(0, self.total_subseconds() / other,
                              self._factor)
        return NotImplemented

    __div__ = __truediv__

    def __cmp__(self, other):
        if isinstance(other, SubSecTimeDelta):
            s, o = self.equalize_factors(other)
            return cmp((s._secs, s._ssecs), (o._secs, o._ssecs))
        elif isinstance(other, (int, long, float)):
            # assume other represents seconds
            return cmp(self, type(self).from_seconds(other, self._factor))
        return NotImplemented

    def __hash__(self):
        return hash((self._secs, self._ssecs, self._factor))


class SubSecTime(SubSecTimeDelta):
    """A class for representing absolute times with subsecond resolution.

    The time represents the seconds and subseconds in UTC from epoch.

    Derived from SubSecTimeDelta, so provides all of its functionality.


    Additional methods:

    from_datetime -- Create a SubSecTime object from a datetime object.

    to_datetime -- Convert SubSecTime object to a datetime object.

    Addition or subtraction with a SubSecTimeDelta produces a new SubSecTime
    object. Otherwise, operations are defined as with SubSecTimeDelta.

    """
    _fromstring_compiled_re = None

    @classmethod
    def from_string(cls, timestr):
        """Create SubSecTime from str in strftime format '%Y-%m-%d %H:%M:%S.%f'.

        Converting a SubSecTime to and from a string is possible so that
        sst == SubSecTime.from_string(str(sst)).

        """
        if cls._fromstring_compiled_re is None:
            pat = '(?P<Y>\d\d\d\d)' \
                  + '-(?P<m>1[0-2]|0[1-9]|[1-9])' \
                  + '-(?P<d>3[0-1]|[1-2]\d|0[1-9]|[1-9]| [1-9])' \
                  + '\s+(?P<H>2[0-3]|[0-1]\d|\d)' \
                  + ':(?P<M>[0-5]\d|\d)' \
                  + ':(?P<S>6[0-1]|[0-5]\d|\d)' \
                  + '\.(?P<f>[0-9]+)'
            cls._fromstring_compiled_re = _re.compile(pat, _re.IGNORECASE)

        v = cls._fromstring_compiled_re.match(timestr).groupdict()
        secsdt = _datetime.datetime(int(v['Y']), int(v['m']), int(v['d']),
                                    int(v['H']), int(v['M']), int(v['S']))
        secs = _calendar.timegm(secsdt.utctimetuple())
        ssecs = int(v['f'])
        digits = len(v['f'])
        factor = 10**digits

        return cls(secs, ssecs, factor)

    @classmethod
    def from_datetime(cls, dt):
        """Create a SubSecTime object from a datetime object.


        Arguments:

        dt -- Datetime object to be converted to a SubSecTime object.

        Returns:

        A SubSecTime object giving the time from epoch in UTC.

        """
        secs = _calendar.timegm(dt.utctimetuple())
        ssecs = dt.microsecond
        factor = 1000000
        return cls(secs, ssecs, factor)

    def to_datetime(self, tz=None):
        """Convert SubSecTime object to a datetime object.

        Precision may be lost in the conversion since datetimes are only
        accurate to microseconds.


        Arguments:

        tz -- Timezone given by a _datetime.tzinfo object within which the
            datetime will be given.
            If None, timezone representing UTC will be used.


        Returns:

        A datetime object representing the same time as the SubSecTime object.

        """
        if tz is None:
            tz = FixedTimezone(tz_offset=0, DST=0, name='UTC')

        return _datetime.datetime.fromtimestamp(self.total_seconds(), tz)

    def make_special(self):
        factor = self._factor
        if factor == 1000:
            return MilliTime.nofix(self._secs, self._ssecs, factor)
        elif factor == 1000000:
            return MicroTime.nofix(self._secs, self._ssecs, factor)
        elif factor == 1000000000:
            return NanoTime.nofix(self._secs, self._ssecs, factor)
        elif factor == 1000000000000:
            return PicoTime.nofix(self._secs, self._ssecs, factor)
        return self

    def change_factor(self, factor):
        """Change the subsecond factor to the one provided."""
        if factor == self._factor:
            return self
        changed = SubSecTime(self._secs, self._ssecs*factor//self._factor,
                             factor)
        return changed

    def strftime(self, fstr, precision=12):
        """Like datetime strftime, but '%f' is replaced with fractional
        subsecond digits.

        """
        dtfstr = fstr.replace('%f', self.fractional_digits(precision))
        return self.to_datetime().strftime(dtfstr)

    def __str__(self):
        return self.strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0').rstrip('.')

    def __add__(self, other):
        if isinstance(other, SubSecTime):
            return NotImplemented
        return SubSecTimeDelta.__add__(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, SubSecTime):
            # need to handle this specially to get SubSecTimeDelta as output
            s, o = self.equalize_factors(other)
            secs = s._secs - o._secs
            ssecs = s._ssecs - o._ssecs
            deltaclass = o.__class__.__bases__[0] # to keep specialized type
            return deltaclass(secs, ssecs, s._factor)
        return SubSecTimeDelta.__sub__(self, other)


class MilliTimeDelta(SubSecTimeDelta):
    # needed to set default factor
    def __new__(cls, secs=0, ssecs=0, factor=1000):
        return SubSecTimeDelta.__new__(cls, secs, ssecs, factor)

    @classmethod
    def nofix(cls, secs=0, ssecs=0, factor=1000):
        if factor != 1000:
            _warnings.warn('Converting subseconds to new factor', RuntimeWarning)
            ssecs = (ssecs*1000) // factor
            if (factor > 1000) and (((ssecs*1000) % factor) != 0):
                _warnings.warn('Precision lost in conversion to new subsecond factor',
                               RuntimeWarning)
        return super(MilliTimeDelta, cls).nofix(secs, ssecs, 1000)

    @classmethod
    def from_seconds(cls, seconds):
        factor = 1000
        return super(MilliTimeDelta, cls).from_seconds(seconds, factor)

    @property
    def milliseconds(self):
        """Get milliseconds."""
        return self.subseconds

    def total_milliseconds(self):
        """Return the total time as an integer in milliseconds."""
        return self.total_subseconds()

    def __repr__(self):
        return '{0}({1}, {2})'.format(self.__class__.__name__, self._secs,
                                      self._ssecs)


class MilliTime(MilliTimeDelta, SubSecTime):
    pass


class MicroTimeDelta(SubSecTimeDelta):
    # needed to set default factor
    def __new__(cls, secs=0, ssecs=0, factor=1000000):
        return SubSecTimeDelta.__new__(cls, secs, ssecs, factor)

    @classmethod
    def nofix(cls, secs=0, ssecs=0, factor=1000000):
        if factor != 1000000:
            _warnings.warn('Converting subseconds to new factor', RuntimeWarning)
            ssecs = (ssecs*1000000) // factor
            if (factor > 1000000) and (((ssecs*1000000) % factor) != 0):
                _warnings.warn('Precision lost in conversion to new subsecond factor',
                               RuntimeWarning)
        return super(MicroTimeDelta, cls).nofix(secs, ssecs, 1000000)

    @classmethod
    def from_seconds(cls, seconds):
        factor = 1000000
        return super(MicroTimeDelta, cls).from_seconds(seconds, factor)

    @property
    def microseconds(self):
        """Get microseconds."""
        return self.subseconds

    def total_microseconds(self):
        """Return the total time as an integer in microseconds."""
        return self.total_subseconds()

    def __repr__(self):
        return '{0}({1}, {2})'.format(self.__class__.__name__, self._secs,
                                      self._ssecs)


class MicroTime(MicroTimeDelta, SubSecTime):
    pass


class NanoTimeDelta(SubSecTimeDelta):
    # needed to set default factor
    def __new__(cls, secs=0, ssecs=0, factor=1000000000):
        return SubSecTimeDelta.__new__(cls, secs, ssecs, factor)

    @classmethod
    def nofix(cls, secs=0, ssecs=0, factor=1000000000):
        if factor != 1000000000:
            _warnings.warn('Converting subseconds to new factor', RuntimeWarning)
            ssecs = (ssecs*1000000000) // factor
            if (factor > 1000000000) and (((ssecs*1000000000) % factor) != 0):
                _warnings.warn('Precision lost in conversion to new subsecond factor',
                               RuntimeWarning)
        return super(NanoTimeDelta, cls).nofix(secs,
                                               ssecs,
                                               1000000000)

    @classmethod
    def from_seconds(cls, seconds):
        factor = 1000000000
        return super(NanoTimeDelta, cls).from_seconds(seconds, factor)

    @property
    def nanoseconds(self):
        """Get nanoseconds."""
        return self.subseconds

    def total_nanoseconds(self):
        """Return the total time as an integer in nanoseconds."""
        return self.total_subseconds()

    def __repr__(self):
        return '{0}({1}, {2})'.format(self.__class__.__name__, self._secs,
                                      self._ssecs)


class NanoTime(NanoTimeDelta, SubSecTime):
    pass


class PicoTimeDelta(SubSecTimeDelta):
    # needed to set default factor
    def __new__(cls, secs=0, ssecs=0, factor=1000000000000):
        return SubSecTimeDelta.__new__(cls, secs, ssecs, factor)

    @classmethod
    def nofix(cls, secs=0, ssecs=0, factor=1000000000000):
        if factor != 1000000000000:
            _warnings.warn('Converting subseconds to new factor', RuntimeWarning)
            ssecs = (ssecs*1000000000000) // factor
            if (factor > 1000000000000) and (((ssecs*1000000000000) % factor) != 0):
                _warnings.warn('Precision lost in conversion to new subsecond factor',
                               RuntimeWarning)
        return super(PicoTimeDelta, cls).nofix(secs,
                                               ssecs,
                                               1000000000000)

    @classmethod
    def from_seconds(cls, seconds):
        factor = 1000000000000
        return super(PicoTimeDelta, cls).from_seconds(seconds, factor)

    @property
    def picoseconds(self):
        """Get picoseconds."""
        return self.subseconds

    def total_picoseconds(self):
        """Return the total time as an integer in picoseconds."""
        return self.total_subseconds()

    def __repr__(self):
        return '{0}({1}, {2})'.format(self.__class__.__name__, self._secs,
                                      self._ssecs)


class PicoTime(PicoTimeDelta, SubSecTime):
    pass
