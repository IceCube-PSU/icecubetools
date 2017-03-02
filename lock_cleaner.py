#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
#METAPROJECT /storage/home/jll1062/build/pingusoft/trunk
"""
Clean I3 files and lockfiles left behind by reconstruction
"""

# TODO: Write operations performed (files removed, renamed, failed, kept) to
# a timestamped logfile.

# TODO: Parallelize (to a degree... the disk i/o may be the limiting factor,
# but CPU ~100% indicates we might be able to gain by parallelizing). Be
# careful though that the locks are handled well (or better yet, turn
# _allfilepaths into a shared queue, so that no thread steps on another when
# recursing)

# TODO: Try out `signal` module to register handler(s) for various signals,
# esp. ctl-c (SIG_INT), so that cleanup can occur even though IceCube software
# is trying to kill the process instead of raising the exception

from __future__ import division, print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import cpu_count, Manager, Pool
from os import listdir, remove, rename
from os.path import abspath, getsize, isdir, isfile, join
import sys
import time
from traceback import format_exception

# IceCube utils written by us
from i3_utils import merge

# Justin's personal scripts (from ~jll1062/mypy/bin)
from genericUtils import expand, nsort, timediffstamp, wstderr, wstdout
from repeated_reco import (EXTENSION, FIT_FIELD_SUFFIX, LOCK_SUFFIX, RECO_RE,
                           RECOS, cleanup_lock_f, acquire_lock, pathFromRecos,
                           read_lockfile, recosFromPath)


__all__ = ['CleanupRecoFiles', 'parse_args', 'main']


def get_recos(frame):
    """Get reconstruction #'s from an I3 file frame.

    Looks for all recos specified in `RECO`, where the reconstruction's
    name appended with `FIT_FIELD_SUFFIX` (e.g. "_FitParams"; defined in
    repeated_reco.py) is expected to be in the frame for that
    reconstruction to be deemed present.

    Parameters
    ----------
    frame : I3Frame
        Frame to inspect

    Returns
    -------
    recos : list of integers
        Reconstruction numbers found in the frame, where numbers are
        references to the `RECO` constant defined in `repeated_reco.py`.

    """
    from icecube import dataio
    if isinstance(frame, basestring):
        filepath = expand(frame)
        try:
            i3file = dataio.I3File(filepath)
        except:
            exc_str = ''.join(format_exception(*sys.exc_info()))
            wstdout('ERROR! Could not open file "%s"\n%s\n'
                    % (filepath, exc_str))
            return []

        try:
            frame = i3file.pop_physics()
        except:
            exc_str = ''.join(format_exception(*sys.exc_info()))
            wstdout('ERROR! Could not open file "%s"\n%s\n'
                    % (filepath, exc_str))
            return []
        else:
            i3file.close()

    keys = frame.keys()
    recos = []
    for reco_num, reco_info in enumerate(RECOS):
        reco_name = reco_info['name']
        if reco_name + FIT_FIELD_SUFFIX in keys:
            recos.append(reco_num)

    return sorted(recos)


def merge_and_rename(files):
    """Merge and/or rename all I3 files passed.

    Parameters
    ----------
    files : sequence or iterable

    Returns
    -------
    None if failure, or -- if success -- string containing path of generated
    file

    """
    from icecube import dataio, icetray

    if len(files) == 1:
        filepath = files[0]
        recos_in_file = get_recos(filepath)
        new_path = pathFromRecos(orig_path=filepath, recos=recos_in_file)
        if new_path == filepath:
            return [filepath]

        try:
            rename(filepath, new_path)
        except:
            exc_str = ''.join(format_exception(*sys.exc_info()))
            wstdout('ERROR! Could not move file from "%s" to "%s".\n%s\n'
                    % (filepath, new_path, exc_str))
            return None

        return filepath

    frames = merge(files)
    for frame in frames:
        if frame.Stop == icetray.I3Frame.Physics:
            recos = get_recos(frame)

    new_path = pathFromRecos(orig_path=expand(files[0]), recos=recos)
    if isfile(new_path):
        recos_in_existing_file = get_recos(new_path)
        if set(recos_in_existing_file) != set(recos):
            try:
                remove(new_path)
            except:
                exc_str = ''.join(format_exception(*sys.exc_info()))
                wstdout('ERROR! Could not remove file "%s".\n%s\n'
                        % (new_path, exc_str))
                return None

            i3file = None
            try:
                i3file = dataio.I3File(new_path, 'w')
                for frame in frames:
                    i3file.push(frame)
            except:
                exc_str = ''.join(format_exception(*sys.exc_info()))
                wstdout('ERROR! Could not write to file "%s".\n%s\n'
                        % (new_path, exc_str))
                return None
            finally:
                if i3file is not None:
                    i3file.close()

    for filepath in files:
        if expand(filepath) == new_path:
            continue
        try:
            remove(filepath)
        except:
            exc_str = ''.join(format_exception(*sys.exc_info()))
            wstdout('ERROR! Could not remove file "%s".\n%s\n'
                    % (filepath, exc_str))

    return new_path


class DeepCleaner(object):
    """Perform deep cleaning on each file in a list of files. Intended to be
    usable via `multiprocessing.Pool.map` via

    file_list = multiprocessing.Manager().list(['x', 'y', 'z'])
    pool = multiprocessing.Pool()
    pool.map(DeepCleaner(), file_list)

    """
    def __init__(self):
        self.file_list = None
        self.kept = []
        self.removed = []
        self.renamed = []
        self.failed_to_remove = []
        self.failed_to_rename = []

    def __call__(self, file_list):
        self.file_list = file_list
        while len(self.file_list) > 0:
            filepath = self.file_list.pop(0)
            self._rename_or_remove(filepath)
        ret = OrderedDict()
        ret['removed'] = self.removed
        ret['renamed'] = self.renamed
        ret['failed_to_remove'] = self.failed_to_remove
        ret['failed_to_rename'] = self.failed_to_rename
        return ret

    def _rename_or_remove(self, filepath):
        """Rename or remove an I3 file based on reconstructions found within
        the file. If there is a name conflict with an existing file,
        recursively resolve the conflict by checking the conflicting file, and
        so forth.

        Parameters
        ----------
        filepath : string
            Path to file to be checked.

        Returns
        -------
        status : string, one of ('kept', 'removed', 'renamed', 'failed')

        """
        from icecube import dataio, icetray

        if not filepath.endswith(EXTENSION):
            self._done_with(filepath)
            return 'kept'

        lockfilepath = filepath + LOCK_SUFFIX
        lock_f = None

        #
        # Acquire lock file
        #
        lock_info = {'type': 'rename_or_remove'}
        try:
            lock_f = acquire_lock(lockfilepath, lock_info)
        except IOError, err:
            if err.errno == 11:
                lock_info = read_lockfile(lockfilepath)
                wstdout('File is locked, skipping: "%s"\n' % filepath)
                self._done_with(lockfilepath)
                return 'kept'
            exc_str = ''.join(format_exception(*sys.exc_info()))
            wstdout('ERROR! Trying to acquire lock on lockfile "%s"\n%s\n'
                    % (lockfilepath, exc_str))
            return 'kept'

        except:
            exc_str = ''.join(format_exception(*sys.exc_info()))
            wstdout('ERROR! Trying to acquire lock on lockfile "%s"\n%s\n'
                    % (lockfilepath, exc_str))
            return 'kept'

        #
        # With lock file acquired, ...
        #
        try:
            purportedly_run = set(recosFromPath(filepath))
            unique_recos_in_file = []

            #
            # Open I3 file
            #
            try:
                i3file = dataio.I3File(filepath)
            except:
                exc_str = ''.join(format_exception(*sys.exc_info()))
                wstdout('ERROR! Working with file "%s"\n%s\n'
                        % (filepath, exc_str))
                return 'kept'

            #
            # Work with I3 file...
            #
            try:
                frame_num = 0
                while i3file.more():
                    try:
                        frame = i3file.pop_frame()
                    except:
                        exc_str = ''.join(format_exception(*sys.exc_info()))
                        wstdout('ERROR! Could not pop frame %d in "%s"\n%s\n'
                                % (frame_num, filepath, exc_str))
                        return 'kept'
                    else:
                        frame_num += 1

                    if frame.Stop != icetray.I3Frame.Physics:
                        continue
                    frame_recos = get_recos(frame)
                    if frame_recos not in unique_recos_in_file:
                        unique_recos_in_file.append(set(frame_recos))

            finally:
                i3file.close()

        finally:
            if lock_f is not None:
                try:
                    remove(lockfilepath)
                except:
                    exc_str = ''.join(format_exception(*sys.exc_info()))
                    wstdout('ERROR! Could not remove lockfile "%s"\n%s\n'
                            % (lockfilepath, exc_str))

        recos_run_on_all_events = set()
        if len(unique_recos_in_file) > 0:
            recos_run_on_all_events = reduce(set.intersection,
                                             unique_recos_in_file)

        if purportedly_run != recos_run_on_all_events:
            wstderr('        Recos found do not match filename. Renaming...\n')
            return self._rename(filepath,
                                recos=sorted(recos_run_on_all_events))

        wstderr('        File is in order. Moving on.\n')
        self._done_with(filepath)
        return 'kept'

    def _done_with(self, filepath):
        try:
            self.file_list.remove(filepath)
        except ValueError:
            pass

    def _rename(self, filepath, recos):
        new_name = pathFromRecos(filepath, recos)
        wstderr('        Renaming to: "%s"\n' % new_name)
        if isfile(new_name):
            wstderr('        Conflicting file found, recursing...\n')
            action_taken = self._rename_or_remove(new_name)

            # If conflicting file is moved or removed...
            if action_taken in ['removed', 'renamed']:
                return self._rename_inner(filepath, new_name)

            # Otherwise, conflicting file was kept, or failed to be
            # renamed or removed, so this file must be removed
            else:
                wstderr('        Removing original file (conflict with an'
                        ' existing file of same name)\n')
                return self._remove(filepath)

        else:
            return self._rename_inner(filepath, new_name)

    def _rename_inner(self, filepath, new_name):
        try:
            rename(filepath, new_name)
        except (IOError, OSError):
            wstderr('Failed to rename file "%s" -> "%s"\n'
                    % (filepath, new_name))
            self.failed_to_rename.append((filepath, new_name))
            return 'failed'
        else:
            self.renamed.append((filepath, new_name))
            return 'renamed'
        finally:
            self._done_with(filepath)

    def _remove(self, filepath):
        try:
            remove(filepath)
        except (IOError, OSError):
            self.failed_to_remove.append(filepath)
            wstderr('ERROR: Failed to remove file "%s"\n' % filepath)
            return 'failed'
        else:
            self.removed.append(filepath)
            return 'removed'
        finally:
            self._done_with(filepath)


class CleanupRecoFiles(object):
    """Recursive function for renaming or deleting a file depending on if its
    contents match the reconstructions its filename indicates.

    Parameters
    ----------
    dirpath : string or list of strings
        If string, taken to be path of the directory to be cleaned. If list of
        strings, taken to be a list of files to clean out.

    """
    def __init__(self, dirpath):
        self.dirpath = dirpath
        if isinstance(dirpath, basestring):
            self.dirpath = abspath(expand(dirpath))
            assert isdir(self.dirpath)
            self.refresh_listing()
        else:
            raise ValueError('`dirpath` must be a string')

        self.removed = []
        self.renamed = []
        self.kept = []
        self.failed_to_remove = []
        self.failed_to_rename = []

    def refresh_listing(self):
        self._allfilepaths = [join(self.dirpath, p.strip())
                              for p in nsort(listdir(self.dirpath))]
        self.original_filepaths = deepcopy(self._allfilepaths)

        # Ignore subdirectories
        for filepath in deepcopy(self._allfilepaths):
            if (isdir(filepath)
                    or not (filepath.endswith(EXTENSION)
                            or filepath.endswith(LOCK_SUFFIX))):
                print(filepath)
                self._done_with(filepath)

        if len(self._allfilepaths) == 0:
            raise ValueError('No files to wokrk on in dir. "%s"'
                             % self.dirpath)

    def remove_empty_files(self):
        """Remove all empty (0-length) files in the directory, regardless of
        the type of file (but skip directories).
        """
        start_time = time.time()
        wstdout('> Removing empty files from dir "%s"\n' % self.dirpath)
        for filepath in deepcopy(self._allfilepaths):
            try:
                if getsize(filepath) == 0:
                    self._remove(filepath)
            except (IOError, OSError):
                self._done_with(filepath)

        self.report()
        wstdout('>     Time to remove empty files: %s\n'
                % timediffstamp(time.time() - start_time))

    def cleanup_lockfiles(self):
        """Remove all expired and obsolete-format lockfiles from the
        directory.
        """
        start_time = time.time()
        wstdout('> Cleaning up stale and old-format lock files...\n')
        for lockfilepath in deepcopy(self._allfilepaths):
            if not lockfilepath.endswith(LOCK_SUFFIX):
                continue

            # 1. if source file doesn't exist, remove the lock file
            sourcefilepath = lockfilepath[:-len(LOCK_SUFFIX)]
            if sourcefilepath not in self._allfilepaths:
                self._remove(lockfilepath)
                continue

            # NOTE: temporarily removed this section since for a time the lock
            # files were being overwritten, so they're 0-length... but still
            # valid. Once the set of runs done on 2017-02-22 completes, I think
            # this issue is fixed and so this sectoun should be re-introduced.


            ## 2. If lock file has outdated format, remove it
            ##    * Remove locked file if...?
            #if 'type' not in lock_info.keys():
            #    self._remove(lockfilepath)
            #    continue

            ## 3. If lock is expired, remove the lock file
            #if lock_info['expires_at'] > time.time():
            #    # Remove locked file if lock type is outfile_lock, as this
            #    # means the output file was not fully written before the
            #    # process died
            #    if lock_info['type'] == 'outfile_lock':
            #        self._remove(lock_info['outfile'])
            #    self._remove(lockfilepath)

            # TODO: the following might require some ssh magic, and/or qstat
            # magic... whatever it is, it'll be a lot of magic to get it to
            # work. So do this another day, since there's no magic today.

            # TODO: SSH via subprocess fails when run under Parrot on ACI!
            # "PRIV_END: seteuid: Operation not permitted"

            # 4. If lock is not expired but process is dead, remove lock
            #    * Remove locked file if lock type is outfile_lock

            # 5. If lock can be acquired on the file, then the lock has expired
            #    and therefore should be deleted.
            try:
                lock_f = acquire_lock(lockfilepath)
            except IOError, err:
                # Could not acquire lock on file, since it is locked: OK
                if err.errno == 11:
                    continue
                continue
            except:
                continue
            else:
                # TODO: remove this clause once the buggy runs where lock files
                # were being overwritten has been cleaned up
                lock_info = read_lockfile(lockfilepath)
                if 'type' in lock_info and lock_info['type'] == 'outfile_lock':
                    self._remove(lock_info['outfile'])
                lock_f.close()
                self._remove(lockfilepath)

        self.report()
        wstdout('>     Time to clean up lock files: %s\n'
                % timediffstamp(time.time() - start_time))

    # TODO
    def tabulate_unique_recos(self):
        pass

    # TODO
    def identify_duplicated_base_events(self):
        pass

    def group_by_event(self):
        """Group I3 files by common event (i.e., exclude whatever processing
        steps have been performed)

        Returns
        -------
        groups : OrderedDict
            Each key is the event base string (excluding reco string and
            extension) and the corresponding value is a list of paths to each
            file in the group.

        """
        groups = OrderedDict()
        for filepath in self._allfilepaths:
            if not filepath.endswith(EXTENSION):
                continue
            base = RECO_RE.sub('', filepath[:-len(EXTENSION)])
            if not groups.has_key(base):
                groups[base] = []
            groups[base].append(filepath)
        return groups

    def deep_clean(self, n_procs=8*cpu_count()):
        """Run the cleaning process

        Parameters
        ----------
        ignore_locks : bool
            Whether to ignore (valid) locks. Note that invalid locks are
            ignored regardless of the value of `ignore_locks`.

        """
        start_time = time.time()
        wstdout('> Deep-cleaning the I3 files in the directory...\n')

        # Create a manager for objects synchronized across workers
        mgr = Manager()
        groups = [mgr.list(g) for g in self.group_by_event().values()]

        pool = Pool(processes=n_procs)
        ret = pool.map(DeepCleaner(), groups)
        removed = []
        renamed = []
        failed_to_remove = []
        failed_to_rename = []
        for d in ret:
            removed.extend(d['removed'])
            renamed.extend(d['renamed'])
            failed_to_remove.extend(d['failed_to_remove'])
            failed_to_rename.extend(d['failed_to_rename'])

        wstderr(' '.join([str(len(g)) for g in groups]) + '\n')

        # TODO: remove files that -- while not redundant -- have fewer recos
        # (or, stretch goal: merge the non-redundant info together)

        self.report()
        wstdout('>     Time to run deep cleaning: %s\n'
                % timediffstamp(time.time() - start_time))

    def merge_and_rename(self, n_procs=cpu_count()):
        """Merge and/or rename"""
        start_time = time.time()
        wstdout('> Merging and/or renaming I3 files in the directory...\n')
        # Create a manager for objects synchronized across workers
        mgr = Manager()
        groups = [mgr.list(g) for g in self.group_by_event().values()]
        pool = Pool(processes=n_procs)
        ret = pool.map(merge_and_rename, groups)
        successes = []
        failures = 0
        for r in ret:
            if r is None:
                failures += 1
            else:
                successes.append(r)
        wstdout('> Failures: %5d; successes: %5d\n'
                % (failures, len(successes)))
        wstdout('>     Time to run merge_and_rename: %s\n'
                % timediffstamp(time.time() - start_time, hms_always=True))
        return failures, successes

    def report(self, kept=None, removed=None, renamed=None,
               failed_to_remove=None, failed_to_rename=None):
        """Report what was done to stdout"""
        if removed is None:
            removed = self.removed
        if renamed is None:
            renamed = self.renamed
        if failed_to_remove is None:
            failed_to_remove = self.failed_to_remove
        if failed_to_rename is None:
            failed_to_rename = self.failed_to_rename
        if kept is None:
            kept = self.kept

        wstdout('>     Total files removed             : %5d\n'
                % len(removed))
        wstdout('>     Total files failed to be removed: %5d\n'
                % len(failed_to_remove))
        wstdout('>     Total files renamed             : %5d\n'
                % len(renamed))
        wstdout('>     Total files failed to be renamed: %5d\n'
                % len(failed_to_rename))

        # TODO: figure out "intentionally kept" files through all three
        # cleaning methods

        #wstdout('>     Total files intentionally kept  : %5d\n'
        #        % len(kept))

    def _done_with(self, filepath):
        try:
            self._allfilepaths.remove(filepath)
        except ValueError:
            pass

    def _remove(self, filepath):
        try:
            remove(filepath)
        except (IOError, OSError):
            self.failed_to_remove.append(filepath)
            wstderr('ERROR: Failed to remove file "%s"\n' % filepath)
            return 'failed'
        else:
            self.removed.append(filepath)
            return 'removed'
        finally:
            self._done_with(filepath)


def parse_args(descr=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(
        description=descr,
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-d', '--dir',
        help='''Directory in which to find the files for cleaning.'''
    )
    parser.add_argument(
        '--remove-empty', action='store_true',
        help='''Remove empty (0-length) files prior to cleaning up locks.'''
    )
    parser.add_argument(
        '--deep-clean', action='store_true',
        help='''Look inside I3 files and remove or rename if the recos are
        missing or incomplete as compared to those indicated by the
        filename. Also, merge files that come from the same event.'''
    )

    args = parser.parse_args()

    args.dir = abspath(expand(args.dir))
    assert isdir(args.dir)

    return args


def main():
    """Main"""
    start_time_sec = time.time()
    args = parse_args()

    cleaner = CleanupRecoFiles(dirpath=args.dir)
    if args.remove_empty:
        cleaner.remove_empty_files()

    cleaner.cleanup_lockfiles()

    if args.deep_clean:
        cleaner.merge_and_rename()

    #lock_count = 0
    #i3_count = 0
    #other_count = 0
    #wstdout('  -> %3d I3 files removed\n' % i3_count)
    #wstdout('  -> %3d lock files removed\n' % lock_count)
    #wstdout('  -> %3d other files removed\n' % other_count)

    wstdout('Script run time: %s\n'
            % timediffstamp(time.time() - start_time_sec))

main.__doc__ = __doc__


if __name__ == '__main__':
    main()
