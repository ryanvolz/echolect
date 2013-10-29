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

#*Each Jicamarca raw data file is structured as blocks of data preceded by a header.
#*Each file starts off with a longer "first header", followed by a block of data,
# followed by a basic header, data, basic header, etc.:
# ----First header = Basic header + Long header
# ----Data block
# ----Basic header
# ----Data block
# ----Basic header
# ----Data block
# ----...
#*Each block of data consists of a number of "profiles" (data for a single pulse), 
# each of which contains data from each of the sample "channels". The samples are 
# arranged hierarchically in real/imaginary, then by channel, then by delay time (range):
# Data block----Profile 0----Delay 0----Channel 0----Real
#                                                ----Imag
#                                   ----Channel 1----Real
#                                                ----Imag
#                                         ...
#                                   ----Channel i----Real
#                                                ----Imag
#                        ----Delay 1----Channel 0----Real
#                                         ...
#                        ----Delay k
#                                         ...
#           ----Profile 1----Delay 0----Channel 0----Real
#                                         ...
#           ----Profile m
#*The basic headers give the number of the following block and the time at which
# the block begins.
#*The long header gives system parameters and describes the number of profiles, delays, 
# and channels per block of data.
#*Detailed structures for the headers can be deduced from the functions for reading them
# given below.

# define flags that appear in headers with their interpretation/meaning
nDataType = dict(RAWDATA = np.uint32(0x00000000),
                 SPECTRA = np.uint32(0x00000001),
                 DIFF2_PROCESS_WIN = np.uint32(0x00001000),
                 DIFF3_PROCESS_WIN = np.uint32(0x00010000), 
                 DIFF4_PROCESS_WIN = np.uint32(0x00011000),
                 SAVE_INCOH_INT_TIME_AVER = np.uint32(0x00100000))

RCfunction = dict(NONE = 0,
                  FLIP = 1,
                  CODE = 2,
                  SAMPLING = 3,
                  LIN6DIV256 = 4,
                  SYNCHRO = 5)

nCodeType = dict(NONE = 0,
                 USERDEFINE = 1,
                 BARKER2 = 2,
                 BARKER3 = 3,
                 BARKER4 = 4,
                 BARKER5 = 5,
                 BARKER7 = 6,
                 BARKER11 = 7,
                 BARKER13 = 8,
                 AC128 = 9,
                 COMPLEMENTARYCODE2 = 10,
                 COMPLEMENTARYCODE4 = 11,
                 COMPLEMENTARYCODE8 = 12,
                 COMPLEMENTARYCODE16 = 13,
                 COMPLEMENTARYCODE32 = 14,
                 COMPLEMENTARYCODE64 = 15,
                 COMPLEMENTARYCODE128 = 16,
                 CODE_BINARY28 = 17)

PROCFLAG = dict(COHERENT_INTEGRATION = np.uint32(0x00000001),
                DECODE_DATA = np.uint32(0x00000002),
                SPECTRA_CALC = np.uint32(0x00000004),
                INCOHERENT_INTEGRATION = np.uint32(0x00000008),
                POST_COHERENT_INTEGRATION = np.uint32(0x00000010),
                SHIFT_FFT_DATA = np.uint32(0x00000020),
                DATATYPE_CHAR = np.uint32(0x00000040),
                DATATYPE_SHORT = np.uint32(0x00000080),
                DATATYPE_LONG = np.uint32(0x00000100),
                DATATYPE_INT64 = np.uint32(0x00000200),
                DATATYPE_FLOAT = np.uint32(0x00000400),
                DATATYPE_DOUBLE = np.uint32(0x00000800),
                DATAARRANGE_CONTIGUOUS_CH = np.uint32(0x00001000),
                DATAARRANGE_CONTIGUOUS_H = np.uint32(0x00002000),
                DATAARRANGE_CONTIGUOUS_P = np.uint32(0x00004000),
                SAVE_CHANNELS_DC = np.uint32(0x00008000),
                DEFLIP_DATA = np.uint32(0x00010000),
                DEFINE_PROCESS_CODE = np.uint32(0x00020000),
                ACQ_SYS_NATALIA = np.uint32(0x00040000),
                ACQ_SYS_ECHOTEK = np.uint32(0x00080000),
                ACQ_SYS_ADRXD = np.uint32(0x000C0000),
                ACQ_SYS_JULIA = np.uint32(0x00100000),
                ACQ_SYS_XXXXXX = np.uint32(0x00140000),
                EXP_NAME_ESP = np.uint32(0x00200000),
                CHANNEL_NAMES_ESP = np.uint32(0x00400000),
                OPERATION_MASK = np.uint32(0x0000003F),
                DATATYPE_MASK = np.uint32(0x00000FC0),
                DATAARRANGE_MASK = np.uint32(0x00007000),
                ACQ_SYS_MASK = np.uint32(0x001C0000))

DINFLAG = dict(RC_L4_TXA_REF = np.uint32(0x00000001),
               RC_L4_TXB_REF = np.uint32(0x00000002),
               RC_L4_TXX_REF = np.uint32(0x00000003),
               RC_L5_TXA_REF = np.uint32(0x00000004),
               RC_L5_TXB_REF = np.uint32(0x00000008),
               RC_L5_TXX_REF = np.uint32(0x0000000C),
               RC_L6_TXA_REF = np.uint32(0x00000010),
               RC_L6_TXB_REF = np.uint32(0x00000020),
               RC_L6_TXX_REF = np.uint32(0x00000030),
               RC_L7_TXA_REF = np.uint32(0x00000040),
               RC_L7_TXB_REF = np.uint32(0x00000080),
               RC_L7_TXX_REF = np.uint32(0x000000C0),
               RC_RESERVED_REF = np.uint32(0x00000000),
               RC_MIDDLE_OF_SUB_BAUD_REF = np.uint32(0x00000100),
               RC_MIDDLE_OF_TX_REF = np.uint32(0x00000200),
               RC_BEGIN_OF_TX_REF = np.uint32(0x00000300),
               RC_SYNC_DELAY_ESP = np.uint32(0x00000400),
               RC_PULSE_AFTER_WINDOW = np.uint32(0x00000800),
               RC_CTRL_SWITCH1 = np.uint32(0x00001000),
               RC_CTRL_SWITCH2 = np.uint32(0x00002000),
               RC_CLK_DIV_ESP = np.uint32(0x00004000),
               RC_RANGE_TR_DYNAMIC = np.uint32(0x00008000),
               RC_RANGE_TXA_DYNAMIC = np.uint32(0x00010000),
               RC_RANGE_TXB_DYNAMIC = np.uint32(0x00020000),
               RC_SYNC_DIV_ESP = np.uint32(0x00040000),
               RC_EXT_SYNC_DELAY_ESP = np.uint32(0x00080000))

def read_basic_header(f):
    """Read/parse a basic header from the given file object at its current position."""
    basic_header_t = np.dtype([('nHeaderLength','<u4'),
                               ('nHeaderVER','<u2'),
                               ('nDataCurrentBlock','<u4'),
                               ('time_sec','<u4'),
                               ('time_msec','<u2'),
                               ('timezone','<i2'),
                               ('dstflag','<i2'),
                               ('nErrorCount','<u4')])
    # fromfile will silently not return the requested number of items if EOF is reached
    basic_header = np.fromfile(f, basic_header_t, 1)
    try:
        basic_header = basic_header.item() # make sure we got entry, convert to python types
    except ValueError:
        raise EOFError('End of file reached. Could not read header.')
    basic_header_dict = dict(zip(basic_header_t.names, basic_header))
    return(basic_header_dict)

def read_system_header(f):
    """Read/parse a system header from the given file object at its current position."""
    system_header_t = np.dtype([('nHeader_Sys_Length','<u4'),
                                ('nSamples','<u4'),
                                ('nProfiles','<u4'),
                                ('nChannels','<u4'),
                                ('nADCResolution','<u4'),
                                ('nPCIDIOBusWidth','<u4'),])
    # fromfile will silently not return the requested number of items if EOF is reached
    system_header = np.fromfile(f, system_header_t, 1)
    try:
        system_header = system_header.item() # make sure we got entry, convert to python types
    except ValueError:
        raise EOFError('End of file reached. Could not read header.')
    system_header_dict = dict(zip(system_header_t.names, system_header))
    return(system_header_dict)

def read_rc_header(f):
    """Read/parse an rc header from the given file object at its current position."""
    # read the static part
    start_pos = f.tell()
    rc_static_header_t = np.dtype([('nHeader_RC_Length','<u4'),
                                   ('nEspType','<u4'),
                                   ('nNTX','<u4'),
                                   ('fIPP','<f4'),
                                   ('fTxA','<f4'),
                                   ('fTxB','<f4'),
                                   ('nNum_Windows','<u4'),
                                   ('nNum_Taus','<u4'),
                                   ('nCodeType','<u4'),
                                   ('nL6_Function','<u4'),
                                   ('nL5_Function','<u4'),
                                   ('fCLOCK','<f4'),
                                   ('nPrePulseBefore','<u4'),
                                   ('nPrePulseAfter','<u4'),
                                   ('sRango_TR','<a16'),
                                   ('nDinFlags','<u4'),
                                   ('sRango_TxA','<a20'),
                                   ('sRango_TxB','<a20')])
    # fromfile will silently not return the requested number of items if EOF is reached
    rc_static_header = np.fromfile(f, rc_static_header_t, 1)
    try:
        rc_static_header = rc_static_header.item() # make sure we got entry, convert to python types
    except ValueError:
        raise EOFError('End of file reached. Could not read header.')
    rc_header = dict(zip(rc_static_header_t.names, rc_static_header))

    # evaluate the controller flags
    rc_header['nDinFlags'] = dict(((k, rc_header['nDinFlags'] & v == v)
                                   for k,v in DINFLAG.iteritems()))

    # read the dynamic part
    sampwin_t = np.dtype([('sRC_fH0','<f4'),
                          ('sRC_fDH','<f4'),
                          ('sRC_nNSA','<u4')])
    sampwin = np.fromfile(f, sampwin_t, rc_header['nNum_Windows'])
    if len(sampwin) != rc_header['nNum_Windows']:
        raise EOFError('End of file reached. Could not read header.')
    rc_header['sRC_fH0'] = sampwin['sRC_fH0'].astype(np.float_)
    rc_header['sRC_fDH'] = sampwin['sRC_fDH'].astype(np.float_)
    rc_header['sRC_nNSA'] = sampwin['sRC_nNSA'].astype(np.int_)

    if rc_header['nNum_Taus'] != 0:
        rc_header['sfTau'] = np.fromfile(f, '<f4', rc_header['nNum_Taus']).astype(np.float_)
        if len(rc_header['sfTau']) != rc_header['nNum_Taus']:
            raise EOFError('End of file reached. Could not read header.')

    if rc_header['nCodeType'] != nCodeType['NONE']:
        try:
            num_codes = np.fromfile(f, '<u4', 1).item()
            num_bauds = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        size_bauds = num_bauds//32 + 1
        codes = np.fromfile(f, '<u4', num_codes*size_bauds).astype(np.int_)
        if len(codes) != num_codes*size_bauds:
            raise EOFError('End of file reached. Could not read header.')
        codes = codes.reshape(size_bauds, num_codes).transpose()
        rc_header['nNum_Codes'] = num_codes
        rc_header['nNum_Bauds'] = num_bauds
        rc_header['snCode'] = codes

    if rc_header['nL5_Function'] == RCfunction['FLIP']:
        try:
            rc_header['nFLIP1'] = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')

    if rc_header['nL6_Function'] == RCfunction['FLIP']:
        try:
            rc_header['nFLIP2'] = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')

    if rc_header['nL5_Function'] == RCfunction['SAMPLING']:
        try:
            rc_header['nL5_Num_Windows'] = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        L5sampwin_t = np.dtype([('sL5_fH0','<f4'),
                                ('sL5_fDH','<f4'),
                                ('sL5_nNSA','<u4')])
        L5sampwin = np.fromfile(f, L5sampwin_t, rc_header['nL5_Num_Windows'])
        if len(L5sampwin) != rc_header['nL5_Num_Windows']:
            raise EOFError('End of file reached. Could not read header.')
        rc_header['sL5_fH0'] = L5sampwin['sL5_fH0'].astype(np.float_)
        rc_header['sL5_fDH'] = L5sampwin['sL5_fDH'].astype(np.float_)
        rc_header['sL5_nNSA'] = L5sampwin['sL5_nNSA'].astype(np.int_)
    elif rc_header['nL5_Function'] == RCfunction['CODE']:
        try:
            num_codes = np.fromfile(f, '<u4', 1).item()
            num_bauds = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        size_bauds = num_bauds//32 + 1
        codes = np.fromfile(f, '<u4', num_codes*size_bauds).astype(np.int_)
        if len(codes) != num_codes*size_bauds:
            raise EOFError('End of file reached. Could not read header.')
        codes = codes.reshape(num_codes, size_bauds)
        rc_header['nL5_Num_Codes'] = num_codes
        rc_header['nL5_Num_Bauds'] = num_bauds
        rc_header['sL5_nCode'] = codes

    if rc_header['nL6_Function'] == RCfunction['SAMPLING']:
        try:
            rc_header['nL6_Num_Windows'] = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        L6sampwin_t = np.dtype([('sL6_fH0','<f4'),
                                ('sL6_fDH','<f4'),
                                ('sL6_nNSA','<u4')])
        L6sampwin = np.fromfile(f, L6sampwin_t, rc_header['nL6_Num_Windows'])
        if len(L6sampwin) != rc_header['nL6_Num_Windows']:
            raise EOFError('End of file reached. Could not read header.')
        rc_header['sL6_fH0'] = L6sampwin['sL6_fH0'].astype(np.float_)
        rc_header['sL6_fDH'] = L6sampwin['sL6_fDH'].astype(np.float_)
        rc_header['sL6_nNSA'] = L6sampwin['sL6_nNSA'].astype(np.int_)
    elif rc_header['nL6_Function'] == RCfunction['CODE']:
        try:
            num_codes = np.fromfile(f, '<u4', 1).item()
            num_bauds = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        size_bauds = num_bauds//32 + 1
        codes = np.fromfile(f, '<u4', num_codes*size_bauds).astype(np.int_)
        if len(codes) != num_codes*size_bauds:
            raise EOFError('End of file reached. Could not read header.')
        codes = codes.reshape(num_codes, size_bauds)
        rc_header['nL6_Num_Codes'] = num_codes
        rc_header['nL6_Num_Bauds'] = num_bauds
        rc_header['sL6_nCode'] = codes

    if rc_header['nDinFlags']['RC_SYNC_DELAY_ESP']:
        try:
            rc_header['nSynchro_Delay'] = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
    if rc_header['nDinFlags']['RC_SYNC_DIV_ESP']:
        try:
            rc_header['nExt_Synchro_Divisor'] = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
    if rc_header['nDinFlags']['RC_CLK_DIV_ESP']:
        try:
            rc_header['nExt_Clk_Divisor'] = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
    if rc_header['nDinFlags']['RC_EXT_SYNC_DELAY_ESP']:
        try:
            rc_header['nExt_Synchro_Delay'] = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
    if rc_header['nDinFlags']['RC_RANGE_TR_DYNAMIC']:
        try:
            range_len = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        range = np.fromfile(f, '<a1', range_len + 1)
        if len(range) != (range_len + 1):
            raise EOFError('End of file reached. Could not read header.')
        rc_header['nTR_RangeLen'] = range_len
        rc_header['nTR_Range'] = range.tostring()
    if rc_header['nDinFlags']['RC_RANGE_TXA_DYNAMIC']:
        try:
            range_len = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        range = np.fromfile(f, '<a1', range_len + 1)
        if len(range) != (range_len + 1):
            raise EOFError('End of file reached. Could not read header.')
        rc_header['nTXA_RangeLen'] = range_len
        rc_header['nTXA_Range'] = range.tostring()
    if rc_header['nDinFlags']['RC_RANGE_TXB_DYNAMIC']:
        try:
            range_len = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        range = np.fromfile(f, '<a1', range_len + 1)
        if len(range) != (range_len + 1):
            raise EOFError('End of file reached. Could not read header.')
        rc_header['nTXB_RangeLen'] = range_len
        rc_header['nTXB_Range'] = range.tostring()

    # verify that we read the correct amount of data
    end_pos = f.tell()
    #assert ((end_pos - start_pos) == rc_header['nHeader_RC_Length']), 'RC Header Parsed Incorrectly'
    f.seek(start_pos + rc_header['nHeader_RC_Length'])

    return(rc_header)

def read_proc_header(f):
    """Read/parse a proc header from the given file object at its current position."""
    # read the static part
    start_pos = f.tell()
    proc_static_header_t = np.dtype([('nHeader_PP_Length','<u4'),
                                     ('nDataType','<u4'),
                                     ('nSizeOfDataBlock','<u4'),
                                     ('nProfilesperBlock','<u4'),
                                     ('nDataBlocksperFile','<u4'),
                                     ('nData_Windows','<u4'),
                                     ('nProcessFlags','<u4'),
                                     ('nCoherentIntegrations','<u4'),
                                     ('nIncoherentIntegrations','<u4'),
                                     ('nTotalSpectra','<u4')])
    # fromfile will silently not return the requested number of items if EOF is reached
    proc_static_header = np.fromfile(f, proc_static_header_t, 1)
    try:
        proc_static_header = proc_static_header.item() # make sure we got entry, convert to python types
    except ValueError:
        raise EOFError('End of file reached. Could not read header.')
    proc_header = dict(zip(proc_static_header_t.names, proc_static_header))

    # evaluate the processing flags
    proc_header['nProcessFlags'] = dict(((k, proc_header['nProcessFlags'] & v == v)
                                         for k,v in PROCFLAG.iteritems()))

    # read the dynamic part
    sampwin_t = np.dtype([('sfH0','<f4'),
                          ('sfDH','<f4'),
                          ('snNSA','<u4')])
    sampwin = np.fromfile(f, sampwin_t, proc_header['nData_Windows'])
    if len(sampwin) != proc_header['nData_Windows']:
            raise EOFError('End of file reached. Could not read header.')
    proc_header['sfH0'] = sampwin['sfH0'].astype(np.float_)
    proc_header['sfDH'] = sampwin['sfDH'].astype(np.float_)
    proc_header['snNSA'] = sampwin['snNSA'].astype(np.int_)

    if proc_header['nTotalSpectra'] != 0:
        proc_header['nSpectraCombinations'] = np.fromfile(self._cur_file,
                                    '<u2', proc_header['nTotalSpectra']).astype(np.int_)
        if len(proc_header['nSpectraCombinations']) != proc_header['nTotalSpectra']:
            raise EOFError('End of file reached. Could not read header.')

    if proc_header['nProcessFlags']['DEFINE_PROCESS_CODE']:
        try:
            num_codes = np.fromfile(f, '<u4', 1).item()
            num_bauds = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        size_bauds = num_bauds//32 + 1
        codes = np.fromfile(f, '<u4', num_codes*size_bauds).astype(np.int_)
        if len(codes) != num_codes*size_bauds:
            raise EOFError('End of file reached. Could not read header.')
        codes = codes.reshape(num_codes, size_bauds)
        proc_header['nProcessCodes'] = num_codes
        proc_header['nProcessBauds'] = num_bauds
        proc_header['sfProcessCode'] = codes

    if proc_header['nProcessFlags']['EXP_NAME_ESP']:
        try:
            name_len = np.fromfile(f, '<u4', 1).item()
        except ValueError:
            raise EOFError('End of file reached. Could not read header.')
        name = np.fromfile(f, '<a1', name_len + 1)
        if len(name) != (name_len + 1):
            raise EOFError('End of file reached. Could not read header.')
        proc_header['nExp_NameLen'] = name_len
        proc_header['sExp_Range'] = name.tostring()

    # verify that we read the correct amount of data
    end_pos = f.tell()
    #assert ((end_pos - start_pos) == proc_header['nHeader_PP_Length']), 'PROC Header Parsed Incorrectly'
    f.seek(start_pos + proc_header['nHeader_PP_Length'])

    return(proc_header)

def read_first_header(f):
    """Read/parse a 'first header' from the given file object at its current position."""
    # get the basic header that makes up the beginning of the first header
    basic_header = read_basic_header(f)

    # get the long header that makes up the rest of the first header
    # beginning with the system header
    system_header = read_system_header(f)
    # now the radar controller header
    rc_header = read_rc_header(f)
    # finally the processing header
    proc_header = read_proc_header(f)

    # merge them all into the first header
    first_header = dict(basic_header.items() + system_header.items() +
                        rc_header.items() + proc_header.items())

    return(first_header)

def parse_dtype(first_header):
    if first_header['nProcessFlags']['DATATYPE_CHAR']:
        raw_dtype = np.dtype([('real', '<i1'), ('imag', '<i1')])
        dtype = np.dtype(np.complex64)
    elif first_header['nProcessFlags']['DATATYPE_SHORT']:
        raw_dtype = np.dtype([('real', '<i2'), ('imag', '<i2')])
        dtype = np.dtype(np.complex64)
    elif first_header['nProcessFlags']['DATATYPE_LONG']:
        raw_dtype = np.dtype([('real', '<i4'), ('imag', '<i4')])
        dtype = np.dtype(np.complex128)
    elif first_header['nProcessFlags']['DATATYPE_INT64']:
        raw_dtype = np.dtype([('real', '<i8'), ('imag', '<i8')])
        dtype = np.dtype(np.complex128)
    elif first_header['nProcessFlags']['DATATYPE_FLOAT']:
        raw_dtype = np.dtype([('real', '<f4'), ('imag', '<f4')])
        dtype = np.dtype(np.complex64)
    elif first_header['nProcessFlags']['DATATYPE_DOUBLE']:
        raw._dtype = np.dtype([('real', '<f8'), ('imag', '<f8')])
        dtype = np.dtype(np.complex128)
    else:
        raw_dtype = np.dtype([('real', '<i2'), ('imag', '<i2')])
        dtype = np.dtype(np.complex64)
        
    return raw_dtype, dtype

def parse_time(header):
    secs = np.datetime64(header['time_sec'], 's')
    msecs = np.timedelta64(header['time_msec'], 'ms')
    time = secs + msecs
    return time

def parse_block_shape(first_header):
    nprofiles = int(first_header['nProfilesperBlock']) # convert to smaller int if possible
    nsamples = first_header['snNSA'][0] # [0] needed b/c we want first (only) "window"
    nchannels = int(first_header['nChannels']) # convert to smaller int if possible
    return nprofiles, nsamples, nchannels

def parse_ts(first_header):
    dr_km = first_header['sfDH']
    ts_s = 2*dr_km*1e3/3e8 # Jicamarca's km units for dr assume c = 3e8 m/s
    ts = np.round(1e9*ts_s).astype('timedelta64[ns]')
    
    if len(ts) == 1:
        return ts[0]
    return ts

def parse_ipp(first_header):
    ipp_km = first_header['fIPP']
    ipp_s = 2*ipp_km*1e3/3e8 # Jicamarca's km units for IPP assume c = 3e8 m/s
    ipp = np.round(1e9*ipp_s).astype('timedelta64[ns]')
    return ipp

def parse_range_index(first_header):
    ngates_win = first_header['snNSA']
    r0_win = first_header['sfH0']*1e3*constants.c/3e8 # m, correct for assumed c = 3e8 m/s
    dr_win = first_header['sfDH']*1e3*constants.c/3e8 # m, correct for assumed c = 3e8 m/s
    
    r = []
    for ngates, r0, dr in zip(ngates_win, r0_win, dr_win):
        r.append(r0 + np.arange(ngates)*dr)
    
    r = np.hstack(r)
    return r

def parse_pulse_codes(first_header):
    codes_decimal = first_header['snCode']
    
    codes = []
    for kcode in xrange(codes_decimal.shape[0]):
        code = []
        for kpart in xrange(codes_decimal.shape[1]):
            code_decimal = codes_decimal[kcode, kpart]
            code.append(np.asarray(list(np.binary_repr(code_decimal)), dtype=np.int8))
        codes.append(np.hstack(code))
    
    return codes

def read_data_block(f, first_header):
    raw_dtype, dtype = parse_dtype(first_header)
    block_shape = parse_block_shape(first_header)
    
    block = np.fromfile(f, raw_dtype, np.product(block_shape))
    try:
        block = block.reshape(block_shape)
    except ValueError: # reshape fails, we didn't get the number of samples we expected
        raise EOFError('End of file reached. Could not read block.')
    
    vlt = np.empty(block_shape, dtype=dtype)
    vlt.real = block['real']
    vlt.imag = block['imag']
    
    return vlt