#!/usr/bin/env python

"""
Python utilities for working with IceCube (I3) files
"""

from __future__ import absolute_import, division, with_statement

import copy
import os
import re
import sys
import textwrap
from traceback import format_exception

import numpy as np

from genericUtils import findFiles

try:
    from justinTSV import readJustinTSV
except ImportError:
    pass


__all__ = ['SOURCE_I3_RE', 'pushEventsCriteria', 'pushSelectEvents',
           'pushEventsByCriteria', 'get_keys', 'countEvents', 'Split',
           'merge', 'countEventsInAllI3Files', 'getEventPhysicsFrames',
           'flattenFrameData', 'flattenFrame', 'passAllCuts']


SOURCE_I3_RE = re.compile(r'(.*)\.(i3)(\.bz2){0,1}$', re.IGNORECASE)


def wstdout(s):
    """write and flush string `s` to stdout"""
    sys.stdout.write(s)
    sys.stdout.flush()


def wstderr(s):
    """write and flush string `s` to stderr"""
    sys.stdout.write(s)
    sys.stdout.flush()


def mkdir(d, mode=0o750, warn=True):
    """Make directory, recursively if parent directories do not exist."""
    d = os.path.expandvars(os.path.expanduser(d))
    if warn and os.path.isdir(d):
        wstderr('Directory already exists: "%s"\n' % d)

    try:
        os.makedirs(os.path.expandvars(os.path.expanduser(d)), mode=mode)
    except OSError as err:
        if err.errno != 17:
            raise err
    else:
        wstdout('Created directory: "%s"\n' % d)


def expand(p):
    """Shortcut to exapnd user dir ("~") and vars for a path."""
    return os.path.expanduser(os.path.expandvars(p))


def pushEventsCriteria(input_i3, output_i3, criteria, debug=False):
    """Grabs events meeting a certain criteria from input file and pushes them
    to the output file.

    Parameters
    ----------
    input_i3, output_i3 : open dataio.I3File objects

    criteria : list of dicts
        List of dictionaries of the following format:
        { (key, (field, operator, value)) }

    debug : bool

    Notes
    -----
    where the following will be performed (return of True passes an event on):
        operator(frame[key].field, value)

    Valid basic operators, like ==, >=, etc. can be found in the Python
    operator module; otherwise, this function must take two arguments and
    return either True or False.

    To do an operation like operator.contains(list, value), you'll have to
    write a custom function that takes the list of acceptable values as the
    second argument and the value to be found as the first argument.

    Note that all specified criteria are AND-ed together to determine if
    an event will be kept.

    """
    from icecube import icetray # pylint: disable=import-error

    if isinstance(criteria, dict):
        criteria = [criteria]

    skip = -1
    storeFrame = False
    while input_i3.more():
        frame = input_i3.pop_frame()
        if frame.Stop == icetray.I3Frame.DAQ:
            skip += 1

            # Default to keep this and its associated Physics (p) frame
            storeFrame = True

            ## If there are no remaining events to grab, might as well quit!
            #if len(eventIDs_remaining) == 0:
            #    break

            # Debug mode stops after first 50 Q frames
            if debug > 0 and skip >= 50:
                break

            # Check all criteria for this frame
            for criterion in criteria:
                for (key, (field, op, value)) in criterion.iteritems():
                    framekey = frame[key]
                    storeFrame &= op(framekey.get(field), value)

        # Store whatever frame we're on, if the storeFrame flag is set...
        if storeFrame:
            output_i3.push(frame)

    del frame


def pushSelectEvents(input_i3, output_i3, events_list, debug=False):
    """Grab events meeting a certain criteria from input file and push them to
    the output file.

    Parameters
    ----------
    input_i3, output_i3 : open dataio.I3File objects (the latter in write mode)
    events_list : list of dicts
        List of dictionaries with either 'skip', 'eventID', or both, to
        uniquely identify an event
    debug : bool

    """
    from icecube import icetray # pylint: disable=import-error

    eventIDs = [event['eventid'] for event in events_list]
    eventSkips = [event['skip'] for event in events_list]

    eventIDs_remaining = copy.deepcopy(eventIDs)

    skip = -1
    storeFrame = False
    while input_i3.more():
        frame = input_i3.pop_frame()
        if frame.Stop == icetray.I3Frame.DAQ:
            skip += 1

            # Default to not keeping this and its associated Physics
            #   frame (p-frame)
            storeFrame = False

            # If there are no remaining events to grab, might as well quit!
            if not eventIDs_remaining:
                break

            # Debug mode stops after first 50 Q frames
            if (debug > 0) and (skip >= 50):
                break

            header = frame['I3EventHeader']
            if header.event_id in eventIDs:
                idx = eventIDs.index(header.event_id)
                if skip == eventSkips[idx]:
                    storeFrame = True
                    wstdout(
                        '    * run_id=' + str(header.run_id) +
                        ' sub_run_id=' + str(header.sub_run_id) +
                        ' event_id=' + str(header.event_id) + '\n'
                    )
                else:
                    wstderr(
                        'Skip mismatch! Skip specd:'
                        + str(eventSkips[idx])
                        + ' but loop skip='+str(skip) + '\n'
                    )
                eventIDs_remaining.remove(header.event_id)

        # Store whatever frame we're on, if the storeFrame flag is set...
        if storeFrame:
            output_i3.push(frame)

    del frame


def pushEventsByCriteria(input_i3, output_i3, criteria_list):
    """Grab events from an input file and push those events (and all their
    associated P, Q, and other types of frames) that meet all specified
    criteria to the output file.

    Parameters
    ----------
    input_i3, output_i3 : open dataio.I3File objects (the latter in write mode)
    criteria_list : list of callables
        Each callable a function that operate on a frame, each of which returns
        True (to keep) or False (to throw away). *All* criteria must be met for
        *any one frame* in the sequence for the entire sequence to be written
        to the output I3 file.

    """
    from icecube import icetray # pylint: disable=import-error

    frameSequenceBuffer = []
    storeSequenceFlags = []
    while input_i3.more():
        frame = input_i3.pop_frame()

        if frame.Stop == icetray.I3Frame.DAQ:
            # Store old sequence to file if any of the physics frames in the
            # sequence matched all of the criteria
            if True in storeSequenceFlags:
                for f in frameSequenceBuffer:
                    output_i3.push(f)

            # Reset the flags and clear the buffer, as a DAQ frame indicates
            # the start of a new sequence of frames that will all be associated
            # with this DAQ frame
            storeSequenceFlags = []
            frameSequenceBuffer = []

        # Store the new frame to the buffer, whatever kind of frame it is
        frameSequenceBuffer.append(frame)

        # Apply criteria only to physics frames
        if frame.Stop == icetray.I3Frame.Physics:
            storeFlag = True
            for criteria in criteria_list:
                storeFlag = storeFlag and criteria(frame)
            storeSequenceFlags.append(storeFlag)


def get_keys(i3fname):
    """Return the complete set of keys present in any frame in the I3 file.

    Parameters
    ----------
    i3fname : string
        Path to I3 file

    Returns
    -------
    keys : list of strings
        All unique keys from any frame in the file

    """
    from icecube import dataio, icetray # pylint: disable=import-error

    input_i3 = dataio.I3File(i3fname, 'r')

    keys = set()
    try:
        while input_i3.more():
            i3frame = input_i3.pop_frame()
            if i3frame.Stop not in [icetray.I3Frame.DAQ,
                                    icetray.I3Frame.Physics]:
                continue
            keys = keys.union(i3frame.keys())
        del i3frame
    finally:
        input_i3.close()

    return sorted(keys)


def countEvents(i3fname, debug=False):
    """Count the number of events (number of Q frames, actually) in an i3 file

    Parameters
    ----------
    i3fname : string
    debug : bool

    """
    from icecube import dataio, icetray # pylint: disable=import-error

    input_i3 = dataio.I3File(i3fname, 'r')
    try:
        frameCount = 0
        while input_i3.more():
            frame = input_i3.pop_frame()
            if frame.Stop == icetray.I3Frame.DAQ:
                frameCount += 1
                #header = frame['I3EventHeader']

            # Debug mode stops after first 50 Q frames
            if debug > 0 and frameCount >= 50:
                break
        del frame
    finally:
        input_i3.close()

    return frameCount


class Split(object):
    """Split `n_total` events from `infile` into separate files, each
    containing at most `n_per_file` events.


    Parameters
    ----------
    infile : string
        Input filename/path

    n_per_file : int
        Maximum number of events per output file

    n_total : None or int
        Consume a total of this many events. If <= 0 or None, all events will
        be consumed from the source file.

    outdir : None or string
        Directory into which to place the output files. If None, output
        directory is the same as the directory containing the input file.

    keep_criteria : None or string
        Criteria for choosing the event for splitting out; events that fail to
        meet the criteria will not count towards n-total or n-per-file. The
        variable `frame` is available to the keep_criteria, which is `eval`'ed
        to yield whether to keep (True) the event or discard it (False)

    """
    def __init__(self, infile, n_per_file=1, n_total=None, outdir=None,
                 keep_criteria=None):
        self.infile_path = expand(infile)
        self.n_per_file = n_per_file
        self.n_total = n_total if n_total is not None else np.inf
        self.outdir = expand(outdir)
        wstdout('outdir = "%s"\n' % self.outdir)
        if self.outdir:
            mkdir(self.outdir, warn=False)
        else:
            self.outdir = os.path.dirname(self.infile_path)
        self.keep_criteria = keep_criteria.strip()
        if self.keep_criteria:
            wstdout('Keep criteria:\n>>> %s\n' % self.keep_criteria)
        mkdir(self.outdir, warn=False)
        self.event_number = -1
        self.all_event_number = -1
        self.all_frame_number = -1
        self.events_written = 0
        self.frame_queue = []
        self.finished = False
        self.outfilepath = None

        basename, ext = self.infile_path, None
        while '.' in basename:
            basename, ext = os.path.splitext(basename)
            if ext.lower() == '.i3':
                break
        self.basename = os.path.basename(basename)

    def split(self):
        """Perform the splitting operation"""
        from icecube import dataclasses, dataio, icetray # pylint: disable=import-error
        if self.finished:
            raise Exception('Already performed operation.')

        infile = dataio.I3File(self.infile_path, 'r')
        self.event_number = -1
        self.all_event_number = -1
        self.all_frame_number = -1
        self.events_written = 0
        try:
            while infile.more():
                frame = infile.pop_frame()
                self.all_frame_number += 1
                stop = frame.Stop
                if stop == icetray.I3Frame.DAQ:
                    wstdout('q')
                    self._increment_event()
                    if self.events_written >= self.n_total:
                        wstderr('\nhit n_total\n')
                        break
                    self.frame_queue.append(frame)
                elif stop == icetray.I3Frame.Physics:
                    keep = True
                    if self.keep_criteria:
                        keep = eval(self.keep_criteria) # pylint: disable=eval-used

                    if keep:
                        wstdout('p')
                        self.frame_queue.append(frame)
                    else:
                        wstdout('x')
                        last_frame = self.frame_queue[-1]
                        if last_frame.Stop == icetray.I3Frame.DAQ:
                            self.frame_queue.pop()
                            self.event_number -= 1
                else:
                    wstdout('\nWARNING! Not keeping frame with Stop = "%s"\n'
                            % stop)
        except Exception:
            raise
        else:
            self.finished = True
        finally:
            infile.close()
            self._write_outfile()

    def _increment_event(self):
        self.event_number += 1
        self.all_event_number += 1
        if (self.event_number > 0) and (self.event_number % self.n_per_file == 0):
            self._write_outfile()

    def _write_outfile(self):
        from icecube import dataio # pylint: disable=import-error

        num_frames = len(self.frame_queue)
        if num_frames == 0:
            #wstderr('\nNo frames to write!\n')
            return
        self.events_written += self.n_per_file
        quotient = (self.event_number - 1) // self.n_per_file
        self.outfilepath = os.path.join(
            self.outdir,
            '%s_split%d.i3.bz2' % (self.basename, quotient)
        )
        #wstdout('\nWriting %d frames to %s\n' % (num_frames, self.outfilepath))
        outfile = dataio.I3File(self.outfilepath, 'w')
        try:
            for _ in range(num_frames):
                frame = self.frame_queue.pop(0)
                outfile.push(frame)
            wstdout('>')
            #wstdout('%s\n' % self.outfilepath)
        finally:
            outfile.close()


def merge(frames):
    """Produce a sequence of 2 frames: The first Q frame encountered (if there
    is one), and the merger of the P frames encountered.

    If I3 file(s) are provided, the files are scanned until the first P frame
    is encountered. The first Q frame encountered, if it is prior to the first
    P frame, is used as the output Q frame if no previous Q frame has been
    seen in the `frames` argument.

    Parameters
    ----------
    frames : I3Frame, string path to i3 file, or sequence thereof

    Returns
    -------
    merged_frames : list of I3Frame
        If a Q frame was encountered, that is first in the list followed by the
        merged P frame. If only P frames were encountered, just the merged P
        frame will be in the `frames` list.

    """
    from icecube import dataio, icetray # pylint: disable=import-error

    if isinstance(frames, (basestring, icetray.I3Frame)):
        frames = [frames]

    q_frame = None
    p_frame = None
    for frame in frames:
        if isinstance(frame, basestring):
            i3file_path = expand(frame)
            try:
                i3file = dataio.I3File(i3file_path)
            except:
                exc_str = ''.join(format_exception(*sys.exc_info()))
                wstdout('ERROR! Could not open file (moving on): "%s"\n%s\n'
                        % (i3file_path, exc_str))
                continue
            try:
                frame_num = 0
                while i3file.more():
                    try:
                        frame = i3file.pop_frame()
                    except:
                        exc_str = ''.join(format_exception(*sys.exc_info()))
                        wstdout(
                            'ERROR! Could not pop frame %d from file "%s".'
                            ' Moving on to next file.\n%s\n'
                            % (frame_num, i3file_path, exc_str)
                        )
                        break
                    else:
                        frame_num += 1

                    if frame.Stop == icetray.I3Frame.DAQ:
                        if q_frame is None:
                            q_frame = frame
                    elif frame.Stop == icetray.I3Frame.Physics:
                        if p_frame is None:
                            p_frame = frame
                        else:
                            p_frame.merge(frame)
                        break
            except:
                exc_str = ''.join(format_exception(*sys.exc_info()))
                wstdout('ERROR! Failure working with file "%s"\n%s\n'
                        % (i3file_path, exc_str))
            finally:
                i3file.close()

        elif isinstance(frame, icetray.I3Frame):
            if frame.Stop == icetray.I3Frame.DAQ:
                if q_frame is None:
                    q_frame = frame
            elif frame.Stop == icetray.I3Frame.Physics:
                if p_frame is None:
                    p_frame = frame
                else:
                    p_frame.merge(frame)

    merged_frames = []
    if q_frame is not None:
        merged_frames.append(q_frame)
    if p_frame is not None:
        merged_frames.append(p_frame)

    return merged_frames


def countEventsInAllI3Files(rootdir='.', recurse=False):
    """Count events (Q-frames) in all I3 files in a directory (and optionally
    recursing into sub-directories).

    Parameters
    ----------
    rootdir : string
    recurse : bool

    """
    digits = 12
    f_iter = findFiles(
        rootdir,
        regex=r'.*i3(\.tar){0,1}(\.bz2){0,1}',
        recurse=recurse
    )
    total_count = 0
    f_count = []
    wstdout(('%'+str(digits)+'s   %s\n') % ('Event count', 'File path'))
    wstdout(('%'+str(digits)+'s   %s\n') % ('-'*digits, '-'*(80-digits-3)))

    for f_path, _, _ in f_iter:
        count = countEvents(f_path)
        total_count += count
        f_count.append((f_path, count))
        wstdout(('%'+str(digits)+'d   %s\n') % (count, f_path))

    wstdout(('-'*digits+'\n'))
    wstdout(
        ('%'+str(digits)+'s total DAQ (Q) frames contained in %d files\n')
        % (total_count, len(f_count))
    )


def getEventPhysicsFrames(i3fname, idx=0, debug=False):
    """Grabs event's physics frame(s) in a list, by event index (i.e., DAQ
    frame index, starting at 0 for first DAQ fraome in the I3 file).

    Parameters
    ----------
    i3fname : string
    idx : int
    debug : bool

    """
    from icecube import dataio, icetray # pylint: disable=import-error

    skip = -1
    keptFrames = []
    storeFrame = False
    input_i3 = dataio.I3File(i3fname, 'r')
    try:
        while input_i3.more():
            frame = input_i3.pop_frame()
            if frame.Stop == icetray.I3Frame.DAQ:
                skip += 1

                # If storeFrame is set, then job is already done since
                # we've reached the next DAQ frame
                if storeFrame:
                    break

                # Default to not keeping this and its associated Physics
                # frame (p-frame)
                storeFrame = False

                # Debug mode stops after first 50 Q frames
                if debug > 0 and skip >= 50:
                    break

                if skip == idx:
                    storeFrame = True
                    #wstdout('    * run_id=' + str(header.run_id) +
                    #        ' sub_run_id=' + str(header.sub_run_id) +
                    #        ' event_id=' + str(header.event_id) + '\n')
            elif (frame.Stop == icetray.I3Frame.Physics) and storeFrame:
                keptFrames.append(frame)

    finally:
        input_i3.close()

    return keptFrames


def flattenFrameData(node, key, datadic=None, prefix='', sep='_'):
    if datadic is None:
        datadic = {}

    if isinstance(key, str):
        datadic[prefix+sep+key] = node.__getattribute__(key)

    elif isinstance(key, list):
        for k in key:
            flattenFrameData(
                node=node, key=k, datadic=datadic, prefix=prefix, sep=sep
            )

    elif isinstance(key, dict):
        for (sk, skk) in key.iteritems():
            newNode = node.__getattribute__(sk)
            flattenFrameData(
                node=newNode, key=skk, datadic=datadic, prefix=prefix+sep+sk,
                sep=sep
            )


def flattenFrame(frame, key_dict, sep='_'):
    datadic = {}
    for (key, subKeyList) in key_dict.iteritems():
        flattenFrameData(node=frame[key], key=subKeyList, datadic=datadic,
                         prefix=key, sep=sep)
        return datadic


# TODO: Check that this actually works!!!
def passAllCuts(frame): #, must_pass_step2=False):
    import PINGUGeneric as PGEN
    header = frame['I3EventHeader']
    if header.run_id in PGEN.runs199_205 + PGEN.runs329_331:
        if frame['NewestBgRejCutsStep1'] and frame['NewestBgRejCutsStep2']:
            return True
    #if header.run_id in PGEN.runs355_357:
    #    if (frame['Cuts_V4_Step1'] and frame['Cuts_V4_Step2']):
    #        return True
    if header.run_id in (PGEN.runs355_357 + PGEN.runs363_365 + PGEN.runs369_371
                         + PGEN.runs388_390):
        #if frame['Cuts_V5_Step1'].value:
        if frame['Cuts_V5_Step1']: # and frame['Cuts_V5_Step2']:
            return True
            #if must_pass_step2:
            #    if frame['Cuts_V5_Step2_upgoing'].value:
            #        return True
            #else:
            #    return True
    return False


def main():
    """main?"""
    import dataio # pylint: disable=import-error

    debug = False

    eventSelIDNum = 0

    eventSelID = 'eventsel_' + format(eventSelIDNum, '02d')

    basenames = ['ind_in_accurate.csv',
                 'ind_in_inaccurate.csv',
                 'ind_out_accurate.csv',
                 'ind_out_inaccurate.csv']

    outputnames = ['insideacc',
                   'insideinac',
                   'outsideacc',
                   'outsideinac']

    eventFilePaths = [os.path.join(expand('~'),
                                   'cowen',
                                   'quality_of_fit',
                                   'reports_archive',
                                   '2014-05-23',
                                   f)
                      for f in basenames]

    outputdir = os.path.join(
        expand('~'),
        'cowen',
        'data',
        'qof',
        eventSelID)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    if debug > 0:
        basenames = [basenames[0]]
        outputnames = [outputnames[0]]

    for (path, outputname) in zip(eventFilePaths, outputnames):
        eventsToProcess = readJustinTSV(path)

        # Organize by source file
        fnames = list(set([event['srcfname'] for event in eventsToProcess]))
        fnames.sort()

        # Construct output I3 file name
        base = os.path.basename(eventsToProcess[0]['srcfname'])
        fname_re = re.compile(r'^([a-zA-Z0-9_]+\.[0-9]+)\.')
        source_i3fname = eventSelID + outputname + '_' + \
                '.'.join(fname_re.findall(base)[0:2]) + '.i3'
        source_i3path = os.path.join(outputdir, source_i3fname)

        # Display status to user
        wstdout('# Processing events listed in\n')
        wstdout('#   ' + path + '\n')
        wstdout('#\n')
        wstdout('# Output to\n')
        wstdout('#   ' + source_i3path + '\n')

        source_i3f = dataio.I3File(source_i3path, 'w')
        try:
            for fname in fnames:
                thisFileEvents = [event for event in eventsToProcess
                                  if event['srcfname'] == fname]
                wstdout(
                    '\n... Extracting events '
                    + '\n'.join(textwrap.wrap(
                        ', '.join([str(e['eventid']) for e in thisFileEvents]),
                        width=120
                    ))
                    + '\n from ' + str(fname) + '\n'
                )
                input_i3f = dataio.I3File(fname, 'r')
                try:
                    pushSelectEvents(input_i3f, source_i3f, thisFileEvents,
                                     debug=debug)
                finally:
                    input_i3f.close()
                if debug > 0:
                    break
        finally:
            source_i3f.close()
        wstdout('\n')


if __name__ == "__main__":
    main()
