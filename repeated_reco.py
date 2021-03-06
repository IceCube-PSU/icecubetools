#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /storage/home/jll1062/build/pingusoft/trunk
"""
HybridReco run with various MultiNest parameters for accuracy comparisons.
"""

# TODO: Time remaining considering number of events in an I3 file is > 1

# TODO: Not tested on PINGU yet (see esp. geometry, srt_pulse_name, and
# segment_length)

# TODO: Use sqlite as more advanced jobs queue, where filesystem is not read
# every time but the sqlite db is updated manually by some process, but then
# worker threads have to tell the sqlite db when they've completed processing a
# file and that file is flagged as "done" and won't be re-processed

from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Mapping, OrderedDict
from copy import deepcopy
from fcntl import flock, LOCK_EX, LOCK_NB
from functools import partial
import getpass
from glob import glob
import grp
from itertools import product
import operator
from os import environ, getpid, remove
from os.path import abspath, basename, dirname, getsize, isdir, isfile, join
import pwd
import random
import re
import signal
import socket
import sys
import time

from dateutil.parser import parse as date_parse
from dateutil.tz import tzlocal
import numpy as np

# Justin's personal scripts (from ~jll1062/mypy/bin)
from genericUtils import (expand, hrlist2list, list2hrlist, mkdir, chown_and_chmod,
                          timediffstamp, timestamp, wstderr, wstdout)
from smartFormat import lowPrec


__all__ = ['EXTENSION', 'LOCK_SUFFIX', 'LOCK_SEP', 'LOCK_FMT',
           'LOCK_ACQ_TIMEOUT', 'RECO_RE', 'NUM_LIVEPOINTS', 'GROUP', 'GID',
           'MODE',
           'TOLERANCES', 'FIT_FIELD_SUFFIX',
           'MN_DEFAULT_KW', 'RECOS', 'MIN_RECO_TIME',
           'get_process_info', 'EventCounter', 'construct_reco_name',
           'recos_from_path', 'path_from_recos', 'parse_args', 'main']


RECOS_SET = 2
DRAGON_L5_CRITERIA = '''(
    frame.Has('IC86_Dunkman_L3') and frame['IC86_Dunkman_L3']
    and frame.Has('IC86_Dunkman_L4') and (frame['IC86_Dunkman_L4']['result'] == 1)
    and frame.Has('IC86_Dunkman_L5') and (frame['IC86_Dunkman_L5']['bdt_score'] >= 0.2)
)'''

EXTENSION = '.i3.bz2'

LOCK_SUFFIX = '.lock'
LOCK_SEP = ' = '
LOCK_FMT = '%s' + LOCK_SEP + '%s\n'
LOCK_ACQ_TIMEOUT = 0.01 # sec

RECO_RE = re.compile(r'_recos([\s0-9,\-]+)')

GROUP = 'dfc13_collab'
GID = None
try:
    GID = grp.getgrnam(GROUP).gr_gid
except KeyError:
    GID = pwd.getpwnam(getpass.getuser()).pw_gid
MODE = 0o666


def get_process_info():
    """Get metadata bout the running process.

    Returns
    -------
    info : dict
        Keys are 'hostname', 'ip_address', 'pid', and 'user'

    """
    info = OrderedDict()
    info['hostname'] = socket.gethostname()
    info['ip_address'] = socket.gethostbyname(info['hostname'])
    info['pid'] = getpid()
    # getpass.getuser doesn't return what I want when running in a PBS job...
    if 'USER' in environ:
        info['user'] = environ['USER']
    else:
        info['user'] = getpass.getuser()
    return info


def construct_reco_name(dims, numlive, tol, trial):
    """Construct a canonical name for the HybridReco/MultiNest reconstruction
    defined by a few particular parameters that MultiNest takes.

    Note that the resulting name is Python-friendly, replacing
    * '.' with 'd'
    * '+' with 'p'
    * '-' with 'm'

    Parameters
    ----------
    dims : int
    numlive : int
    tol : float
    trial : int

    Returns
    -------
    reco_name : string

    """
    mn_name = 'MN%dD' % dims
    numlive_name = 'nlive%d' % numlive
    tol_name = 'tol' + lowPrec(tol)
    trial_name = 'trial%d' % trial
    reco_name = '_'.join([mn_name, numlive_name, tol_name, trial_name])
    reco_name = reco_name.replace('.', 'd')
    reco_name = reco_name.replace('+', 'p')
    reco_name = reco_name.replace('-', 'm')
    return reco_name


FIT_FIELD_SUFFIX = '_FitParams'
MN_CONFIG_PREFIX = 'MN_Full_'
MN_DEFAULT_KW = dict(
    config_prefix=MN_CONFIG_PREFIX,
    segment_length=7, # meters
    has_mc_truth=True,
    fit_cascade_direction=False,
    #input_pulses=args.srt_pulse_name,
    usecoszen=True,
    mmodal=True,
    consteff=False,
    #numlive=75,
    efr=1.0,
    #tol=1.1,
    #base_geometry='deepcore',
    track_zenith_bounds=[-1, 1],
    cascade_zenith_bounds=[-1, 1],
    show_feedback=0,
    #time_limit=time_limit, # sec
    store_llhp_values=True,
    raw_output_base_name='./MN-default',
    write_raw_output_files=0,
    #If=lambda f: f.Has('Cuts_V5.1_Step1') and f['Cuts_V5.1_Step1'].value
)

RECOS = []

if RECOS_SET == 1:
    NUM_LIVEPOINTS = [1000, 10000]
    TOLERANCES = [1e-2]

    # High-resolution MultiNest runs
    HIRES_TRIALS = 1
    TIME_LIMIT_FACTOR = 1000
    for _numlive, _tol, _trial in product(NUM_LIVEPOINTS, TOLERANCES,
                                          list(range(HIRES_TRIALS))):
        _time_limit = 60 * int(np.round(np.clip(
            _numlive*2/3 + 84,
            a_min=10,
            a_max=22*60
        ))) * TIME_LIMIT_FACTOR
        _reco_name = construct_reco_name(dims=8, numlive=_numlive, tol=_tol,
                                         trial=_trial)

        _kwargs = deepcopy(MN_DEFAULT_KW)
        _kwargs['prefix'] = '%s_' % _reco_name
        _kwargs['time_limit'] = _time_limit
        _kwargs['numlive'] = _numlive
        _kwargs['tol'] = _tol
        RECOS.append(
            dict(name=_reco_name, time_limit=_time_limit, kwargs=_kwargs)
        )
    # "Standard" HybridReco settings from PINGU, re-run 10 times
    for _trial in range(10):
        _numlive = 75
        _tol = 1.1
        _time_limit = 150 * 60 * TIME_LIMIT_FACTOR
        _reco_name = construct_reco_name(dims=8, numlive=_numlive, tol=_tol,
                                         trial=_trial)
        _kwargs = deepcopy(MN_DEFAULT_KW)
        _kwargs['prefix'] = '%s_' % _reco_name
        _kwargs['time_limit'] = _time_limit
        _kwargs['numlive'] = _numlive
        _kwargs['tol'] = _tol
        RECOS.append(
            dict(name=_reco_name, time_limit=_time_limit, kwargs=_kwargs)
        )
    # 50 livepoints, repeated 20 times
    for _trial in range(20):
        _numlive = 50
        _tol = 0.01
        _time_limit = 100 * 60 * TIME_LIMIT_FACTOR
        _reco_name = construct_reco_name(dims=8, numlive=_numlive, tol=_tol,
                                         trial=_trial)
        _kwargs = deepcopy(MN_DEFAULT_KW)
        _kwargs['prefix'] = '%s_' % _reco_name
        _kwargs['time_limit'] = _time_limit
        _kwargs['numlive'] = _numlive
        _kwargs['tol'] = _tol
        RECOS.append(
            dict(name=_reco_name, time_limit=_time_limit, kwargs=_kwargs)
        )
    # 25 livepoints, repeated 20 times
    for _trial in range(20):
        _numlive = 25
        _tol = 0.01
        _time_limit = 50 * 60 * TIME_LIMIT_FACTOR
        _reco_name = construct_reco_name(dims=8, numlive=_numlive, tol=_tol,
                                         trial=_trial)
        _kwargs = deepcopy(MN_DEFAULT_KW)
        _kwargs['prefix'] = '%s_' % _reco_name
        _kwargs['time_limit'] = _time_limit
        _kwargs['numlive'] = _numlive
        _kwargs['tol'] = _tol
        RECOS.append(
            dict(name=_reco_name, time_limit=_time_limit, kwargs=_kwargs)
        )
    # 10 livepoints, repeated 20 times
    for _trial in range(20):
        _numlive = 10
        _tol = 0.01
        _time_limit = 30 * 60 * TIME_LIMIT_FACTOR
        _reco_name = construct_reco_name(dims=8, numlive=_numlive, tol=_tol,
                                         trial=_trial)
        _kwargs = deepcopy(MN_DEFAULT_KW)
        _kwargs['prefix'] = '%s_' % _reco_name
        _kwargs['time_limit'] = _time_limit
        _kwargs['numlive'] = _numlive
        _kwargs['tol'] = _tol
        RECOS.append(
            dict(name=_reco_name, time_limit=_time_limit, kwargs=_kwargs)
        )

elif RECOS_SET == 2:
    NUM_LIVEPOINTS = [50, 75]
    TOLERANCES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    TRIALS = 1
    _tr = list(range(TRIALS))
    for _trial, _numlive, _tol in product(_tr, NUM_LIVEPOINTS, TOLERANCES):
        _time_limit = 479 * 60 # 479 min = 7h 59min = 28740 sec
        _reco_name = construct_reco_name(dims=8, numlive=_numlive, tol=_tol,
                                         trial=_trial)
        _kwargs = deepcopy(MN_DEFAULT_KW)
        _kwargs['prefix'] = '%s_' % _reco_name
        _kwargs['time_limit'] = _time_limit
        _kwargs['numlive'] = _numlive
        _kwargs['tol'] = _tol
        RECOS.append(
            dict(name=_reco_name, time_limit=_time_limit, kwargs=_kwargs)
        )

    # We should run multiple recos with nlive={50, 75} and
    # tol={1e-3, 1e-2, 1e-1} and also run a bunch of nlive=25 with
    # tol={1e-3, 1e-2, 1e-1} repeated versions of the above (recos 12 on) and
    # attach these instead
    RECOS = RECOS[:12]

    NUM_LIVEPOINTS = [25]
    TOLERANCES = [1e-3, 1e-2, 1e-1]
    TRIALS = 1
    _tr = list(range(TRIALS))
    for _trial, _numlive, _tol in product(_tr, NUM_LIVEPOINTS, TOLERANCES):
        _time_limit = 479 * 60 # 479 min = 7h 59min = 28740 sec
        _reco_name = construct_reco_name(dims=8, numlive=_numlive, tol=_tol,
                                         trial=_trial)
        _kwargs = deepcopy(MN_DEFAULT_KW)
        _kwargs['prefix'] = '%s_' % _reco_name
        _kwargs['time_limit'] = _time_limit
        _kwargs['numlive'] = _numlive
        _kwargs['tol'] = _tol
        RECOS.append(
            dict(name=_reco_name, time_limit=_time_limit, kwargs=_kwargs)
        )

    NUM_LIVEPOINTS = [25, 50, 75]
    TOLERANCES = [1e-3, 1e-2, 1e-1]
    # By now we've already done one trial of each (trial0), so do 9 more and
    # start here with trial1
    TRIALS = 9
    _tr = list(range(1, 1 + TRIALS))
    for _trial, _numlive, _tol in product(_tr, NUM_LIVEPOINTS, TOLERANCES):
        _time_limit = 479 * 60 # 479 min = 7h 59min = 28740 sec
        _reco_name = construct_reco_name(dims=8, numlive=_numlive, tol=_tol,
                                         trial=_trial)
        _kwargs = deepcopy(MN_DEFAULT_KW)
        _kwargs['prefix'] = '%s_' % _reco_name
        _kwargs['time_limit'] = _time_limit
        _kwargs['numlive'] = _numlive
        _kwargs['tol'] = _tol
        RECOS.append(
            dict(name=_reco_name, time_limit=_time_limit, kwargs=_kwargs)
        )


MIN_RECO_TIME = min([r['time_limit'] for r in RECOS])


def if_proto(frame, reco_base, eval_criteria=None):
    """Prototype function that skips running an IceTray module or segment on a
    frame if field_name = `reco_base` + FIT_FIELD_SUFFIX is already in the
    frame.

    Note that this function must be used with `partial` to populate values for
    `reco_base`; this function cannot be used directly as an argument to an
    IceTray module's `If` kwarg (as only a single argument, `frame`, is
    provided to that function by the IceTray)

    """
    from icecube import dataclasses, dataio, icetray, multinest_icetray # pylint: disable=unused-variable, import-error

    field_name = reco_base + FIT_FIELD_SUFFIX

    eval_result = True
    if eval_criteria is not None:
        eval_result = eval(eval_criteria) # pylint: disable=eval-used

    if not eval_result:
        sys.stdout.write('[NOT RUN] Not running reco %s (event did not pass'
                         ' `eval_criteria`)\n' % field_name)
        sys.stdout.flush()
        return False

    if not frame.Has(field_name):
        sys.stdout.write('[RUN    ] Running missing reco %s\n' % field_name)
        sys.stdout.flush()
        return True

    #if run_if_previous_timed_out and frame[field_name].has_reached_time_limit:
    #    sys.stdout.write('[RUN    ] Running timed-out reco %s\n' % field_name)
    #    sys.stdout.flush()
    #    return True

    sys.stdout.write('[NOT RUN] Not running reco %s\n' % field_name)
    sys.stdout.flush()
    return False


# Do not repeat a reconstruction already performed on a given event; the 'If'
# function ensures this isn't done.
for reco in RECOS:
    if_func = partial(
        if_proto,
        reco_base=reco['name'],
        eval_criteria=DRAGON_L5_CRITERIA,
        #run_if_previous_timed_out=True
    )
    reco['kwargs']['If'] = if_func


def recos_from_path(filepath):
    """Parse filepath for recos (reported to have been) run on the file.

    Parameters
    ----------
    filepath : string

    Returns
    -------
    recos : list of integers

    """
    filename = basename(filepath)
    recos = [hrlist2list(x) for x in RECO_RE.findall(filename)]
    recos = sorted(reduce(operator.add, recos, []))
    return recos


def path_from_recos(orig_path, recos, ext=EXTENSION):
    """Construct a new filename given original path/filename and a list of
    reconstructions that have been run on the file.

    Parameters
    ----------
    orig_path : string
        Original filename or path to modify in order to construct a new
        filename/path

    recos : sequence of zero or more integers
        Which reconstructions were run on the file

    ext : string
        Extension of original file path; used also in new path

    Returns
    -------
    new_path : string

    """
    assert orig_path.endswith(ext)

    # Strip the extension
    orig_path = orig_path[:-len(ext)]

    # Construct a concise string indicating recos run (or '' if none were run)
    if recos:
        reco_str = '_recos' + list2hrlist(sorted(recos))
    else:
        reco_str = ''

    # Put it all together
    return RECO_RE.sub('', orig_path) + reco_str + ext


def acquire_lock(lock_path, lock_info=None):
    """Acquire a lock on the file at `lock_path` and record `lock_info` to
    that file.

    Parameters
    ----------
    lock_path : string
    lock_info : None or Mapping

    Returns
    -------
    lock_f : file object
        This holds an exlcusive lock; close the file or use fcntl.flock to
        release the lock.

    Raises
    ------
    IOError: [Errno 11] Resource temporarily unavailable
        The lock is held by a different process on the file. Note that the
        same process can re-acquire a lock infinitely many times (but there
        is no lock counter, so the first file descriptor to be closed or
        explicitly release the lock also releases the lock for all other
        instaces within the process).

    ValueError: I/O operation on closed file
        This might be the case if the file has disappeared between opening it
        and actually acquiring the exclusive lock.

    Notes
    -----
    See
        https://loonytek.com/2015/01/15/advisory-file-locking-differences-between-posix-and-bsd-locks
    for more info about locks. Note that this function uses flock, i.e.
    POSIX--not BSD--locking. This means that it should work even with an NFS
    filesystem, although there are other tradeoffs as well. And locking is
    "cooperative," so another process can simply ignore the `flock` locking
    protocol altogether and read/write/delete the file.

    """
    lock_acq_timeout_time = time.time() + LOCK_ACQ_TIMEOUT
    lock_f = file(lock_path, 'a')
    lock_acquired = False
    while time.time() <= lock_acq_timeout_time:
        try:
            flock(lock_f, LOCK_EX | LOCK_NB)
        except IOError, err:
            if err.errno == 11:
                wstdout('.')
                time.sleep(random.random()*LOCK_ACQ_TIMEOUT/100)
                continue
            else:
                raise
        else:
            lock_acquired = True

    if not lock_acquired:
        exc = IOError('[Errno 11] Resource temporarily unavailable')
        exc.errno = 11
        raise exc

    if lock_info is not None:
        assert isinstance(lock_info, Mapping)
        # Write info out to the lock through a new, write-able file
        # descriptor; note that the lock is still held by the `lock`
        # file descriptor.
        with file(lock_path, 'w') as lock_w:
            for k, v in lock_info.items():
                lock_w.write(LOCK_FMT % (k, v))
            try:
                chown_and_chmod(lock_w, gid=GID, mode=MODE)
            except OSError, err:
                # errno 1 : operation not permitted (allowing this)
                if err.errno != 1:
                    raise

    return lock_f


def read_lockfile(path):
    """Read the contents of a lockfile and convert into an OrderedDict"""
    with file(path, 'r') as f:
        lines = f.readlines()
    lock_info = OrderedDict()
    for line in lines:
        k, v = line.split(LOCK_SEP)
        if k in ['acquired_at', 'expires_at']:
            dt = date_parse(v)
            # Timestamps are all in UTC, but `time.time()` (used within the
            # script) uses localtime, so convert to local timezone then convert
            # to integer seconds since epoch
            v = int(dt.astimezone(tzlocal()).strftime('%s'))
        if k in ['pid']:
            v = int(v)
        lock_info[k] = v
    return lock_info


def cleanup_lock_f(lock_f, force_remove=False):
    """Remove a lock file and release the lock held on it (in that order, to
    ensure another process doesn't create a new lockfile after releasing lock
    and prior to this function removing the file). If that fails, though,
    release the lock and remove the file.

    Parameters
    ----------
    lock_f : open file object, or None
        If None, this simply returns without an exception.

    force_remove : bool
        Remove the lock file even if we don't hold a lock on it. WARNING! This
        is unsafe behavior in multi-threaded/multi-processing situations.

    Raises
    ------
    AssertionError
        If lock_f is not a file object

    ValueError
        If lock_f is a closed file object and `force_remove` is False

    """
    if lock_f is None:
        return

    assert isinstance(lock_f, file)

    if lock_f.closed:
        if force_remove:
            try:
                remove(lock_f.name)
            except OSError, err:
                # OSError.errno of 2 means file doesn't exist, which is fine;
                # otherwise, raise the exception since the file could not be
                # deleted.
                if err.errno != 2:
                    raise
        else:
            raise ValueError(
                'Lock file is already closed; refusing to remove file.'
            )
    else:
        removed = False
        retry = 0
        for retry in range(10):
            if removed:
                break
            try:
                remove(lock_f.name)
            except OSError, err:
                if err.errno == 2:
                    removed = True
                    break
                elif err.errno == 5:
                    removed = False
            retry += 1
            time.sleep(0.01)

        lock_f.close()
        if not removed:
            try:
                remove(lock_f.name)
            except OSError, err:
                # errno: 2=no file; 16=Device or resource busy
                if err.errno not in [2, 16]:
                    raise


class EventCounter(object):
    """
    Determine whether or not to process an event based on the event number in
    the I3 file(s) being processed. Intended for the `process_event` method to
    be used as an IceTray module.

    Parameters
    ----------
    srt_pulse_name : string
        Name of the pulse series to look for in the frame. This is used as a
        proxy to identify a frame as an event that should be counted.

    skip : int >= 0
        Number of events to skip over. If 0, no events will be skipped.
    n_events : int
        Total number of events to process, starting from `skip`. If <= 0,
        *all* events will be processed starting from `skip`.

    """
    def __init__(self, srt_pulse_name, skip, n_events):
        self.srt_pulse_name = srt_pulse_name
        self.skip = skip
        self.n_events = n_events
        self.event_number = -1
        self.events_run = []

    def process_event(self, *args, **kwargs):
        """Pass this method as an IceTray module."""
        if 'frame' in kwargs.keys():
            frame = kwargs['frame']
        elif not args:
            frame = None
        elif len(args) == 1:
            frame = args[0]
        elif len(args) > 1:
            raise ValueError('Got %d frames, can only handle 1.'
                             % len(args))

        if frame is None:
            return False

        if frame.Has(self.srt_pulse_name):
            self.event_number += 1
        else:
            return False

        if (self.event_number < self.skip
                and not (self.skip == 0 and self.event_number == -1)):
            return False

        if self.n_events > 0 and self.event_number >= self.skip + self.n_events:
            return False

        wstdout('> Processing an event; total events processed will be %5d\n'
                % (self.event_number + 1))
        self.events_run.append(self.event_number)

        return True


class FileLister(object):
    """List file(s), ignoring any that are locked.

    Note that either `infile` or `indir` must be specified, but not both.

    Parameters
    ----------
    infile : string
        File to return (at most once).
    indir : string
        Directory to search for unlocked files.

    """
    def __init__(self, infile=None, indir=None):
        self.infile = infile
        self.indir = indir
        self.used_file = False
        if self.infile is not None:
            assert self.indir is None
            self.mode = 'infile'
            self.files = [infile]
        elif self.indir is not None:
            self.mode = 'indir'
            self.files = glob(join(self.indir, '*' + EXTENSION))
        else:
            raise ValueError('Either `infile` or `indir` must not be None.')
        #random.shuffle(self.files)
        self.next_file = None

    def get_next_file(self):
        """Retrieve the next file.

        Returns
        -------
        next_file : string

        """
        if self.mode == 'infile':
            if self.used_file:
                self.next_file = None
            else:
                self.next_file = self.infile
                self.used_file = True
        elif self.mode == 'indir':
            self.next_file = self._get_file_from_dir()

        return self.next_file

    def _get_file_from_dir(self):
        while self.files:
            f = self.files.pop()
            return f
        return None


def parse_args(descr=__doc__):
    """Parse command line arguments"""

    parser = ArgumentParser(
        description=descr,
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--infile',
        default=None,
        help='''Path to the input file. If specified, do not specify --indir'''
    )
    parser.add_argument(
        '--indir',
        default=None,
        help='''Path to the input directory, from which all I3 files will be
        processed. If specified, to not specify --infile'''
    )
    parser.add_argument(
        '--outdir',
        required=True,
        help='''Output directory; must not be same as indir (or infile's
        directory), as the source file may be corrupted or removed if an error
        is encountered.''',
    )
    parser.add_argument(
        '--gcd',
        required=True,
        help='Path to GCD file',
    )
    parser.add_argument(
        '--skip',
        type=int, default=0,
        help='''Number of events to skip. Note that an "event" is defined as a
        frame containing the SRT_PULSE_NAME.''',
    )
    parser.add_argument(
        '--n-events',
        type=int, default=0,
        help='''Total number of "events" to process (n-events <= 0 processes
        all events in the file(s) starting from from --skip). Note that an
        event is defined as a frame containing the SRT_PULSE_NAME.''',
    )
    parser.add_argument(
        '--recos',
        type=str.lower, default='all',
        help='''Reco numbers to run. Specify "all" or a human-readable list,
        like "0-2,5" (which would perform steps 0, 1, 2, and 5). Note that
        indexing begins at 0.''',
    )
    parser.add_argument(
        '--detector',
        required=True, type=str.lower, choices=['deepcore', 'pingu'],
        help='''Detector for which the MC or data was produced (this selects an
        appropriate SRT_PULSE_NAME)'''
    )
    parser.add_argument(
        '--minutes-remaining',
        type=float, default=np.inf,
        help='''Minutes remaining in a job to run a reco; only those
        reconstructions with time limits less than this will run. Specify <= 0
        for no limit.'''
    )
    parser.add_argument(
        '--hours-remaining',
        type=float, default=np.inf,
        help='''Hours remaining in a job to run a reco; only those
        reconstructions with time limits less than this will run. Specify <= 0
        for no limit.'''
    )
    parser.add_argument(
        '--use-locks',
        action='store_true',
        help='''EXPERIMENTAL (and flaky): Use file locking to protect files
        from being processed by multiple separate processes.'''
    )

    args = parser.parse_args()

    assert args.skip >= 0

    if args.recos == 'all':
        args.requested = list(range(len(RECOS)))
    else:
        args.requested = hrlist2list(args.recos)

    num_inspecs = 0
    if args.infile is not None:
        args.infile = abspath(expand(args.infile))
        if not isfile(args.infile):
            raise IOError('`infile` "%s" is not a file.' % args.infile)
        num_inspecs += 1

    if args.indir is not None:
        args.indir = abspath(expand(args.indir))
        assert isdir(args.indir)
        num_inspecs += 1

    if num_inspecs != 1:
        raise ValueError(
            'Either --infile or --indir must be specified but not both.'
        )

    if args.infile is not None:
        indir = abspath(dirname(expand(args.infile)))
    else:
        indir = abspath(expand(args.indir))

    args.outdir = abspath(expand(args.outdir))
    if args.outdir == indir:
        raise ValueError(
            'Outdir cannot be same as indir (or if infile is specified,'
            ' directory in which infile resides'
        )

    mkdir(args.outdir, warn=False)
    assert isdir(args.outdir)

    args.gcd = expand(args.gcd)
    assert isfile(args.gcd)

    if args.detector == 'pingu':
        args.srt_pulse_name = 'newSRT_TW_Cleaned_WavedeformPulses'
        args.geometry = 'pingu'
    elif args.detector == 'deepcore':
        args.srt_pulse_name = 'SRTTWOfflinePulsesDC'
        args.geometry = 'deepcore'

    if np.isinf(args.minutes_remaining):
        if np.isinf(args.hours_remaining):
            args.seconds_remaining = np.inf
        else:
            args.seconds_remaining = args.hours_remaining * 3600
    else:
        if not np.isinf(args.hours_remaining):
            assert args.minutes_remaining == args.hours_remaining*60
        args.seconds_remaining = args.minutes_remaining * 60

    if args.seconds_remaining <= 0:
        args.seconds_remaining = np.inf

    args.seconds_remaining = int(np.ceil(np.clip(args.seconds_remaining,
                                                 a_min=0, a_max=31556926)))

    return args


def main():
    """Main"""
    start_time_sec = time.time()
    args = parse_args()

    def _sigint_handler(signal, frame): # pylint: disable=unused-argument, redefined-outer-name
        wstderr('='*79 + '\n')
        wstderr('*** CAUGHT CTL-C (sigint) *** ... attempting to cleanup!\n')
        wstderr('='*79 + '\n')
        raise KeyboardInterrupt

    # Import IceCube things now
    from I3Tray import I3Tray # pylint: disable=import-error
    from icecube import dataclasses, dataio, icetray, multinest_icetray # pylint: disable=unused-variable, import-error
    from cluster import get_spline_tables

    lock_info = get_process_info()

    wstdout('='*79 + '\n')
    for d in [vars(args), lock_info]:
        wstdout('\n')
        wstdout(
            '\n'.join([(('%20s'%k) + ' = %s'%d[k]) for k in sorted(d.keys())])
        )
        wstdout('\n'*2)

    file_lister = FileLister(infile=args.infile, indir=args.indir)
    event_counter = EventCounter(srt_pulse_name=args.srt_pulse_name,
                                 skip=args.skip, n_events=args.n_events)
    expiration = time.time() + args.seconds_remaining
    expiration_timestamp = timestamp(at=expiration, utc=True)

    while True:
        infile_path = file_lister.get_next_file()

        if infile_path is None:
            wstdout('> No more files that can be processed. Quitting.\n')
            break

        # NOTE: cannot run on a file that has _all_ recos already run, since
        # output file cannot be same as input file (which it will have same
        # name, since the name is derived from recos run / etc.)

        already_run = recos_from_path(infile_path)
        # NOTE: now skipping a reco is determined ONLY by the "If" kwarg, and
        # not by the filename at all (swap the comment on the next line for the
        # line below to change behavior back)
        #recos_not_run_yet = sorted(set(args.requested) - set(already_run))
        recos_not_run_yet = sorted(set(args.requested))

        if not recos_not_run_yet:
            wstdout('> Nothing more to be done on file. Moving on. ("%s")\n'
                    % infile_path)
            continue

        # See if file still exists
        if not isfile(infile_path):
            wstdout('> File no longer exists. Moving on. ("%s")\n'
                    % infile_path)
            continue

        # Skip if empty input files
        if getsize(infile_path) == 0:
            wstdout('> Input file is 0-length. Moving on. ("%s")\n'
                    % infile_path)
            continue

        # NOTE: commenting out the following and forcing an extremely long
        # timeout to allow all recos to run (of which many won't have to,
        # becuase they've already been run). Uncomment the following three
        # lines and comment out the "time_remaining =" line below to change the
        # behavior back when most or all recos have to be run

        #time_remaining = np.ceil(
        #    args.seconds_remaining - (time.time() - start_time_sec)
        #)
        time_remaining = 3600 * 24 * 10000

        # See if any reco at all fits in the remaining time
        if time_remaining <= MIN_RECO_TIME:
            wstdout('Not enough time to run *any* reco. Quitting.\n')
            break

        # See if any of the recos needing to be run on *this* file fit in the
        # remaining time; register all `reco_num`s that can be run
        recos_to_run = []
        after_proc_time_remaining = time_remaining
        for reco_num in recos_not_run_yet:
            time_limit = RECOS[reco_num]['time_limit']
            if time_limit > after_proc_time_remaining:
                continue
            recos_to_run.append(reco_num)
            after_proc_time_remaining -= time_limit

        time_to_run_processing = time_remaining - after_proc_time_remaining
        # Give the lock an extra minute beyond the strict time to run
        expiration = time.time() + time_to_run_processing + 60
        expiration_timestamp = timestamp(at=expiration, utc=True)

        if not recos_to_run:
            wstdout('Not enough time to run any remaining reco on file. Moving'
                    ' on. ("%s")\n' % infile_path)
            continue

        infile_lock_f, outfile_lock_f = None, None
        infile_lock_path = infile_path + LOCK_SUFFIX
        outfile_lock_path = None
        allrecos = set(recos_to_run).union(already_run)
        outfile_name = basename(
            path_from_recos(orig_path=infile_path, recos=allrecos)
        )
        outfile_path = abspath(expand(join(args.outdir, outfile_name)))

        #print('args.outdir: "%s", outfile_name: "%s", outfile_path: "%s"'
        #      % (args.outdir, outfile_name, outfile_path))
        #break # debug

        outfile_lock_path = outfile_path + LOCK_SUFFIX

        if outfile_name == infile_path or outfile_path == infile_path:
            wstdout(
                'Outfile is same as infile, which will lead to removal of'
                ' infile. Path = "%s" ; Moving on to next input file.\n'
                % infile_path
            )
            continue

        lock_info['acquired_at'] = timestamp(utc=True)
        lock_info['expires_at'] = expiration_timestamp
        lock_info['infile'] = infile_path
        lock_info['outfile'] = outfile_path

        if isfile(outfile_path):
            wstdout('> Outfile path exists; will overwrite if both infile and'
                    ' outfile locks can be obtained! ...\n'
                    '>     "%s"\n' % outfile_path)

        # NOTE:
        # Create lockfiles (if they don't exist) for each of the infile and
        # outfile, and try to acquire exclusive locks on these before
        # working with either the infile or outfile.
        #
        # Also: write info to the lockfiles to know when it's okay to clean
        # each up manually. Note that the `flock` will be removed by the OS
        # as soon as the lock file is closed or when this process dies.
        lock_info['type'] = 'infile_lock'
        try:
            if args.use_locks:
                infile_lock_f = acquire_lock(infile_lock_path, lock_info)
        except IOError:
            wstdout(
                '> infile lock failed to be obtained.'
                '    "%s"\n'
                % infile_lock_path
            )
            infile_lock_f = None
            continue

        lock_info['type'] = 'outfile_lock'
        try:
            if args.use_locks:
                outfile_lock_f = acquire_lock(outfile_lock_path, lock_info)
        except IOError:
            wstdout(
                'ERROR: outfile lock failed to be obtained.'
                ' Cleaning up infile lock and moving on.\n'
                '    "%s" (infile lock)\n'
                '    "%s" (outfile lock)\n'
                % (infile_lock_path, outfile_lock_path)
            )
            cleanup_lock_f(infile_lock_f)
            infile_lock_f = None
            continue

        try:
            remove(outfile_path)
        except OSError, err:
            # OSError.errno == 2 => "No such file or directory", which is OK
            # since the point of `remove` is to make sure the path doesn't
            # exist; otherwise, we can't go on since the output file exists but
            # apparently cannot be overwritten
            if err.errno != 2:
                wstdout(
                    '> ERROR: obtained locks but outfile path exists and'
                    ' cannot be removed. Cleaning up locks and moving on.\n'
                    '>     "%s" (outfile path)\n'
                    '>     "%s" (infile_lock_path)\n'
                    '>     "%s" (outfile_lock_path)\n'
                    % (outfile_path, infile_lock_path, outfile_lock_path)
                )
                cleanup_lock_f(infile_lock_f)
                infile_lock_f = None
                cleanup_lock_f(outfile_lock_f)
                outfile_lock_f = None
                continue
        except Exception:
            cleanup_lock_f(infile_lock_f)
            infile_lock_f = None
            cleanup_lock_f(outfile_lock_f)
            outfile_lock_f = None
            raise

        try:
            tray = I3Tray()
            files = []
            files.append(args.gcd)
            files.append(infile_path)

            icetray.set_log_level(icetray.I3LogLevel.LOG_WARN)

            tray.AddModule('I3Reader', 'reader', FileNameList=files)

            # Module for selecting events for processing
            tray.AddModule(event_counter.process_event, 'process_event')

            spline_tables = get_spline_tables()

            tray.AddSegment(
                multinest_icetray.MultiNestConfigureLikelihood, 'MultiNestConf',
                config_prefix=MN_CONFIG_PREFIX,
                cascade_spline_abs=spline_tables['cscd_amplitude'],
                cascade_spline_prob=spline_tables['cscd_timing'],
                track_spline_abs=spline_tables['trk_amplitude'],
                track_spline_prob=spline_tables['trk_timing'],
                bad_doms_name='BadDomsList',
                input_pulses=args.srt_pulse_name,
                wave_form_range='WaveformRange',
                use_ball_speedup=True,
                ball_radius=150.0,
            )

            for reco_num in recos_to_run:
                reco_name = RECOS[reco_num]['name']
                time_limit = RECOS[reco_num]['time_limit']
                kwargs = RECOS[reco_num]['kwargs']

                wstdout(
                    '> Setting up tray to run reco #%3d, %s (time limit %s)\n'
                    % (reco_num, reco_name, timediffstamp(time_limit))
                )

                tray.AddSegment(
                    multinest_icetray.MultiNestReco, reco_name,
                    input_pulses=args.srt_pulse_name,
                    base_geometry=args.geometry,
                    **kwargs
                )

            tray.AddModule(
                'I3Writer', 'EventWriter',
                Filename=outfile_path,
                Streams=[icetray.I3Frame.Physics, icetray.I3Frame.DAQ],
                DropOrphanStreams=[icetray.I3Frame.DAQ]
            )

            tray.AddModule('TrashCan', 'Done')

            wstdout('> ' + '-'*77 + '\n')
            wstdout('> Will run reco(s) %s on infile\n'
                    '>     "%s"\n' % (list2hrlist(recos_to_run), infile_path))
            wstdout('> Results will be output to file\n'
                    '>     "%s"\n' % outfile_path)
            wstdout('> Time now: %s\n' % timestamp(utc=True))
            wstdout(
                '> Est. time remaining after running %d reco(s): %s\n'
                % (len(recos_to_run),
                   timediffstamp(after_proc_time_remaining, hms_always=True))
            )
            wstdout('> ' + '-'*77 + '\n')

            signal.signal(signal.SIGINT, _sigint_handler)

            try:
                tray.Execute()
                tray.Finish()
            except:
                wstdout('ERROR executing IceTray; removing outfile\n'
                        '    "%s"\n' % outfile_path)
                remove(outfile_path)
                raise
            else:
                # Write a file indicating that the output file is ready for
                # further processing
                readyfile_path = outfile_path + '.ready'
                with open(readyfile_path, 'w'):
                    pass
                try:
                    chown_and_chmod(readyfile_path, gid=GID, mode=MODE)
                except OSError, err:
                    # errno 1 : operation not permitted (allowing this)
                    if err.errno != 1:
                        raise

                # If the directive was to process all events in the file,
                # remove the original file as this is completely superseded by
                # `outfile` (i.e., `outfile` is a superset of `infile_path`).
                if args.skip == 0 and args.n_events <= 0:
                    wstdout('> SUCCESS; removing now-obsolete infile\n'
                            '>     "%s"\n' % infile_path)
                    remove(infile_path)
                try:
                    chown_and_chmod(outfile_path, gid=GID, mode=MODE)
                except OSError, err:
                    # errno 1 : operation not permitted (allowing this)
                    if err.errno != 1:
                        raise

        finally:
            cleanup_lock_f(infile_lock_f)
            infile_lock_f = None
            cleanup_lock_f(outfile_lock_f)
            outfile_lock_f = None
            del tray

        dt = time.time() - start_time_sec
        time_remaining = np.ceil(
            args.seconds_remaining - (dt)
        )
        wstdout('> Total time elapsed : %s\n'
                % timediffstamp(dt, hms_always=True))
        wstdout('> Time remaining     : %s\n\n'
                % timediffstamp(time_remaining, hms_always=True))

main.__doc__ = __doc__


if __name__ == '__main__':
    main()
