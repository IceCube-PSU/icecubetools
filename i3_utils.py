#!/usr/bin/env python


from __future__ import with_statement

import os
import copy
import re
import sys
import textwrap

import numpy as np

import genericUtils as GUTIL

try:
    from justinTSV import readJustinTSV
except:
    pass


SOURCE_I3_RE = re.compile(r'(.*)\.(i3)(\.bz2){0,1}$', re.IGNORECASE)


def pushEventsCriteria(inputI3, outputI3, criteria, debug=False):
    """
    Grabs events meeting a certain criteria from input file and pushes them to
    the output file.

    inputI3 and outputI3 should be already opened (via I3File);

    criteria should contain a list of dictionaries of the following format:
        { (key, (field, operator, value)) }

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
    from icecube import icetray

    if isinstance(criteria, dict):
        criteria = [criteria]

    skip = -1
    storeFrame = False
    while inputI3.more():
        frame = inputI3.pop_frame()
        if frame.Stop == icetray.I3Frame.DAQ:
            skip += 1

            #-- Default to keep this and its associated Physics (p) frame
            storeFrame = True

            ##-- If there are no remaining events to grab, might as well quit!
            #if len(eventIDs_remaining) == 0:
            #    break

            #-- Debug mode stops after first 50 Q frames
            if debug > 0 and skip >= 50:
                break

            #-- Check all criteria for this frame
            for criterion in criteria:
                for (key, (field, op, value)) in criterion.iteritems():
                    framekey = frame[key]
                    storeFrame &= op(framekey.get(field), value)

        #-- Store whatever frame we're on, if the storeFrame flag is set...
        if storeFrame:
            outputI3.push(frame)

    del frame


def pushSelectEvents(inputI3, outputI3, eventsList, debug=False):
    """Grab events meeting a certain criteria from input file and push them to
    the output file.

    inputI3 and outputI3 should be already opened, via
      dataio.I3File(filename, 'r')
      dataio.I3File(filename, 'w')
    respectively

    eventsList should contain dictionaries with either 'skip', 'eventID', or
    both, to uniquely identify an event
    """
    from icecube import icetray

    eventIDs = [event['eventid'] for event in eventsList]
    eventSkips = [event['skip'] for event in eventsList]

    eventIDs_remaining = copy.deepcopy(eventIDs)

    skip = -1
    storeFrame = False
    while inputI3.more():
        frame = inputI3.pop_frame()
        if frame.Stop == icetray.I3Frame.DAQ:
            skip += 1

            #-- Default to not keeping this and its associated Physics
            #   frame (p-frame)
            storeFrame = False

            #-- If there are no remaining events to grab, might as well quit!
            if len(eventIDs_remaining) == 0:
                break

            #-- Debug mode stops after first 50 Q frames
            if (debug > 0) and (skip >= 50):
                break

            header = frame['I3EventHeader']
            if header.event_id in eventIDs:
                idx = eventIDs.index(header.event_id)
                if skip == eventSkips[idx]:
                    storeFrame = True
                    GUTIL.wstdout(
                        '    * run_id=' + str(header.run_id) +
                        ' sub_run_id=' + str(header.sub_run_id) +
                        ' event_id=' + str(header.event_id) + '\n'
                    )
                else:
                    GUTIL.wstderr(
                        'Skip mismatch! Skip specd:'
                        + str(eventSkips[idx]) +
                        ' but loop skip='+str(skip) + '\n'
                    )
                eventIDs_remaining.remove(header.event_id)

        #-- Store whatever frame we're on, if the storeFrame flag is set...
        if storeFrame:
            outputI3.push(frame)

    del frame


def pushEventsByCriteria(inputI3, outputI3, criteriaList, debug=False):
    """Grab events from an input file and push those events (and all their
    associated P, Q, and other types of frames) that meet all specified
    criteria to the output file.

    inputI3 and outputI3 should be already opened, via
      dataio.I3File(filename, 'r')
      dataio.I3File(filename, 'w')
    respectively

    criteriaList should be a list of functions that operate on a frame, each of
    which returns True (to keep) or False (to throw away). *All* criteria must
    be met for *any one frame* in the sequence for the entire sequence to be
    written to the output I3 file.
    """
    from icecube import icetray

    frameSequenceBuffer = []
    storeSequenceFlags = []
    while inputI3.more():
        frame = inputI3.pop_frame()

        if frame.Stop == icetray.I3Frame.DAQ:
            # Store old sequence to file if any of the physics frames in the
            # sequence matched all of the criteria
            if True in storeSequenceFlags:
                for f in frameSequenceBuffer:
                    outputI3.push(f)

            # Reset the flags and clear the buffer, as a DAQ frame indicates the
            # start of a new sequence of frames that will all be associated
            # with this DAQ frame
            storeSequenceFlags = []
            frameSequenceBuffer = []

        # Store the new frame to the buffer, whatever kind of frame it is
        frameSequenceBuffer.append(frame)

        # Apply criteria only to physics frames
        if frame.Stop == icetray.I3Frame.Physics:
            storeFlag = True
            for criteria in criteriaList:
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
    from icecube import dataio, icetray

    inputI3 = dataio.I3File(i3fname, 'r')

    keys = set()
    try:
        while inputI3.more():
            frame = inputI3.pop_frame()
            keys = keys.union(frame.keys())
        del frame
    finally:
        inputI3.close()

    return keys


def countEvents(i3fname, debug=False):
    """Count the number of events (number of Q frames, actually) in an i3
    file

    """
    from icecube import dataio, icetray

    inputI3 = dataio.I3File(i3fname, 'r')

    try:
        frameCount = 0
        while inputI3.more():
            frame = inputI3.pop_frame()
            if frame.Stop == icetray.I3Frame.DAQ:
                frameCount += 1
                #header = frame['I3EventHeader']

            #-- Debug mode stops after first 50 Q frames
            if debug > 0 and frameCount >= 50:
                break
        del frame
    finally:
        inputI3.close()

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

    """
    def __init__(self, infile, n_per_file=1, n_total=None, outdir=None):
        self.infile_path = os.path.expandvars(os.path.expanduser(infile))
        self.n_per_file = n_per_file
        self.n_total = n_total if n_total is not None else np.inf
        self.outdir = outdir
        if self.outdir is None:
            self.outdir = os.path.dirname(self.infile_path)
        GUTIL.mkdir(self.outdir, warn=False)
        self.outfile = None
        self.event_number = -1
        self.frame_queue = []
        self.finished = False
        self.outfilepath = None

        basename, ext = self.infile_path, None
        while '.' in basename:
            basename, ext = os.path.splitext(basename)
            if ext.lower() == '.i3':
                break
        self.basename = basename

    def split(self):
        """Perform the splitting operation"""
        from icecube import dataio, icetray
        if self.finished:
            raise Exception('Already performed operation.')

        infile = dataio.I3File(self.infile_path, 'r')
        try:
            while infile.more():
                frame = infile.pop_frame()
                if frame.Stop == icetray.I3Frame.DAQ:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    self._increment_event()
                    if self.event_number >= self.n_total:
                        break
                    if self.outfile is None:
                        quotient, _ = divmod(self.event_number,
                                             self.n_per_file)
                        self.outfilepath = os.path.join(
                            self.outdir,
                            '%s_split%d.i3.bz2' % (self.basename, quotient)
                        )
                        self.outfile = dataio.I3File(self.outfilepath, 'w')
                    self.frame_queue.append(frame)
                elif frame.Stop == icetray.I3Frame.Physics:
                    self.frame_queue.append(frame)
        except Exception:
            raise
        else:
            self.finished = True
        finally:
            infile.close()
            self._cleanup_outfile()

    def _increment_event(self):
        self.event_number += 1
        if self.event_number % self.n_per_file == 0:
            self._cleanup_outfile()

    def _append_to_file(self):
        if self.event_number < 0 or self.outfile is None:
            return

        for _ in xrange(len(self.frame_queue)):
            frame = self.frame_queue.pop(0)
            self.outfile.push(frame)

    def _cleanup_outfile(self):
        self._append_to_file()
        if self.outfile is not None:
            sys.stdout.write('o')
            sys.stdout.flush()
            self.outfile.close()
            self.outfile = None


def countEventsInAllI3Files(rootdir='.', recurse=False):
    digits = 12
    f_iter = GUTIL.findFiles('.', regex=r'.*i3(\.tar){0,1}(\.bz2){0,1}', recurse=recurse)
    total_count = 0
    f_count = []
    GUTIL.wstdout(('%'+str(digits)+'s   %s\n') % ('Event count', 'File path'))
    GUTIL.wstdout(('%'+str(digits)+'s   %s\n') % ('-'*digits, '-'*(80-digits-3)))
    for f_path, _, _ in f_iter:
        count = countEvents(f_path)
        total_count += count
        f_count.append((f_path, count))
        GUTIL.wstdout(('%'+str(digits)+'d   %s\n') % (count, f_path))
    GUTIL.wstdout(('-'*digits+'\n'))
    GUTIL.wstdout(
        ('%'+str(digits)+'s total DAQ (Q) frames contained in %d files\n')
        % (total_count, len(f_count))
    )


def getEventPhysicsFrames(i3fname, idx=0, debug=False):
    """
    Grabs event's physics frame(s) in a list, by event index (i.e., DAQ frame
    index, starting at 0 for first DAQ fraome in the I3 file)
    """
    from icecube import dataio, icetray

    skip = -1
    keptFrames = []
    storeFrame = False
    inputI3 = dataio.I3File(i3fname, 'r')
    try:
        while inputI3.more():
            frame = inputI3.pop_frame()
            if frame.Stop == icetray.I3Frame.DAQ:
                skip += 1

                #-- If storeFrame is set, then job is already done since
                #   we've reached the next DAQ frame
                if storeFrame:
                    break

                #-- Default to not keeping this and its associated Physics
                #   frame (p-frame)
                storeFrame = False

                #-- Debug mode stops after first 50 Q frames
                if debug > 0 and skip >= 50:
                    break

                if skip == idx:
                    storeFrame = True
                    #GUTIL.wstdout('    * run_id=' + str(header.run_id) +
                    #        ' sub_run_id=' + str(header.sub_run_id) +
                    #        ' event_id=' + str(header.event_id) + '\n')
            elif (frame.Stop == icetray.I3Frame.Physics) and storeFrame:
                keptFrames.append(frame)

    finally:
        inputI3.close()

    return keptFrames


def flattenFrameData(node, key, datadic={}, prefix='', sep='_'):
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
                node=newNode, key=skk, datadic=datadic, prefix=prefix+sep+sk, sep=sep
            )


def flattenFrame(frame, keyDict, sep='_'):
    datadic = {}
    for (key, subKeyList) in keyDict.iteritems():
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
    if header.run_id in PGEN.runs355_357 + PGEN.runs363_365 + PGEN.runs369_371 + PGEN.runs388_390:
        #if frame['Cuts_V5_Step1'].value:
        if frame['Cuts_V5_Step1']: # and frame['Cuts_V5_Step2']:
            return True
            #if must_pass_step2:
            #    if frame['Cuts_V5_Step2_upgoing'].value:
            #        return True
            #else:
            #    return True
    return False


if __name__ == "__main__":
    import dataio

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

    eventFilePaths = [os.path.join(os.path.expanduser('~'),
                                   'cowen',
                                   'quality_of_fit',
                                   'reports_archive',
                                   '2014-05-23',
                                   f)
                      for f in basenames]

    outputdir = os.path.join(
        os.path.expanduser('~'),
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

        #-- Organize by source file
        fnames = list(set([event['srcfname'] for event in eventsToProcess]))
        fnames.sort()

        #-- Construct output I3 file name
        base = os.path.basename(eventsToProcess[0]['srcfname'])
        fname_re = re.compile(r'^([a-zA-Z0-9_]+\.[0-9]+)\.')
        source_i3fname = eventSelID + outputname + '_' + \
                '.'.join(fname_re.findall(base)[0:2]) + '.i3'
        source_i3path = os.path.join(outputdir, source_i3fname)

        #-- Display status to user
        GUTIL.wstdout('# Processing events listed in\n')
        GUTIL.wstdout('#   ' + path + '\n')
        GUTIL.wstdout('#\n')
        GUTIL.wstdout('# Output to\n')
        GUTIL.wstdout('#   ' + source_i3path + '\n')

        source_i3f = dataio.I3File(source_i3path, 'w')
        try:
            for fname in fnames:
                thisFileEvents = [event for event in eventsToProcess
                                  if event['srcfname'] == fname]
                GUTIL.wstdout(
                    '\n... Extracting events '
                    + '\n'.join(textwrap.wrap(
                        ', '.join([str(e['eventid']) for e in thisFileEvents]),
                        width=120
                    ))
                    + '\n from ' + str(fname) + '\n'
                )
                input_i3f = dataio.I3File(fname, 'r')
                try:
                    pushSelectEvents(input_i3f, source_i3f, thisFileEvents, debug=debug)
                finally:
                    input_i3f.close()
                if debug > 0:
                    break
        finally:
            source_i3f.close()
        GUTIL.wstdout('\n')
