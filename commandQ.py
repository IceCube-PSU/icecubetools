#!/usr/bin/env python
"""
Simple command queue.
"""

from __future__ import division

from argparse import ArgumentParser
from collections import Iterable, Sequence
import datetime
import fcntl
import os
import random
import subprocess
import sys
import time
import traceback

__author__ = 'J.L. Lanfranchi'
__date__ = '2016-11-07'


QUEUE_FPATH = os.path.expandvars(os.path.expanduser('~/.command_queue'))
DFLT_TIMEOUT = 10 # seconds


def mkdir(d):
    try:
        os.makedirs(d, mode=0750)
    except OSError as err:
        # Ignore "dir already exists" error; raise otherwise
        if err[0] != 17:
            raise err


def get_command(timeout):
    t0 = time.time()
    timeout_time = t0 + timeout
    while time.time() < timeout_time:
        lockf = open(QUEUE_FPATH + '.lock', 'w')
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError, err:
            if err.errno == 11:
                time.sleep(random.random()*timeout/100)
                continue
            else:
                raise

        try:
            with open(QUEUE_FPATH, 'r') as queue:
                commands = queue.readlines()
                this_command = commands.pop(0).strip()
            if len(commands) == 0:
                break
            else:
                with open(QUEUE_FPATH, 'w') as queue:
                    queue.writelines(commands)
            return this_command

        except IOError, err:
            if err.errno == 2:
                return None

        finally:
            lockf.close()

    # If you get here, then failed to get command
    raise Exception('Timeout: Failed to get a command in %s sec.' %timeout)


def insert(index, object, timeout=DFLT_TIMEOUT):
    dirname = os.path.dirname(QUEUE_FPATH)
    if dirname != '':
        mkdir(dirname)

    t0 = time.time()
    timeout_time = t0 + timeout
    while time.time() < timeout_time:
        lockf = open(QUEUE_FPATH + '.lock', 'w')
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError, err:
            if err.errno == 11:
                time.sleep(random.random()*timeout/100)
                continue
            else:
                raise

        try:
            try:
                with open(QUEUE_FPATH, 'r') as queue:
                    commands = queue.readlines()
            except IOError, err:
                # Only allow "No such file or directory" error
                if err.errno != 2:
                    raise
                commands = []
            if not command.endswith('\n'):
                command += '\n'
            commands.insert(index, command)
            with open(QUEUE_FPATH, 'w') as queue:
                queue.writelines(commands)

            return

        finally:
            lockf.close()

    # If you get here, then failed to get command
    raise Exception('Timeout: Failed to get a command in %s sec.' %timeout)


def append(command, timeout=DFLT_TIMEOUT):
    dirname = os.path.dirname(QUEUE_FPATH)
    if dirname != '':
        mkdir(dirname)

    t0 = time.time()
    timeout_time = t0 + timeout
    while time.time() < timeout_time:
        lockf = open(QUEUE_FPATH + '.lock', 'w')
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError, err:
            if err.errno == 11:
                time.sleep(random.random()/10.0)
                continue
            else:
                raise

        try:
            try:
                with open(QUEUE_FPATH, 'r') as queue:
                    commands = queue.readlines()
            except IOError, err:
                # Only allow "No such file or directory" error
                if err.errno != 2:
                    raise
                commands = []
            if not command.endswith('\n'):
                command += '\n'
            commands.append(command)
            with open(QUEUE_FPATH, 'w') as queue:
                queue.writelines(commands)

            return

        finally:
            lockf.close()

    # If you get here, then failed to get command
    raise Exception('Timeout: Failed to get a command in %s sec.' %timeout)


def extend(commands, timeout=DFLT_TIMEOUT):
    assert isinstance(commands, Iterable)

    dirname = os.path.dirname(QUEUE_FPATH)
    if dirname != '':
        mkdir(dirname)

    t0 = time.time()
    timeout_time = t0 + timeout
    while time.time() < timeout_time:
        lockf = open(QUEUE_FPATH + '.lock', 'w')
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError, err:
            if err.errno == 11:
                time.sleep(random.random()/10.0)
                continue
            else:
                raise

        try:
            try:
                with open(QUEUE_FPATH, 'r') as queue:
                    current_commands = queue.readlines()
            except IOError, err:
                # Only allow "No such file or directory" error
                if err.errno != 2:
                    raise
                current_commands = []

            new_commands = []
            for command in commands:
                if not command.endswith('\n'):
                    command += '\n'
                new_commands.append(command)

            current_commands.extend(new_commands)
            with open(QUEUE_FPATH, 'w') as queue:
                queue.writelines(commands)

            return

        finally:
            lockf.close()

    # If you get here, then failed to get command
    raise Exception('Timeout: Failed to get a command in %s sec.' %timeout)


def push(command, timeout):
    insert(0, command, timeout=timeout)


def run_command(command, cpu_threads=1, cpu_cores=None, gpus=None, retry_failed=True):
    env = os.environ.copy()

    if cpu_cores is not None:
        raise NotImplementedError('`cpu_cores`')

    if gpus is not None:
        if isinstance(gpus, int):
            gpus = str(gpus)
        elif isinstance(gpus, Sequence):
            gpus = ','.join([str(n) for n in gpus])
        if not isinstance(gpus, basestring):
            raise ValueError('Unhandled `gpus`: "%s" of type %s'
                             %(gpus, type(gpus)))
        env['CUDA_VISIBLE_DEVICES'] = gpus

    env['MKL_NUM_THREADS'] = format(cpu_threads, 'd')
    env['OMP_NUM_THREADS'] = format(cpu_threads, 'd')

    is_exc = False
    kill = False
    try:
        ret = subprocess.call(command, env=env, shell=True,
                              stderr=subprocess.STDOUT)
    except (KeyboardInterrupt, SystemExit):
        is_exc = True
        kill = True
    except:
        is_exc = True
        kill = False

    if is_exc:
        exc_str = traceback.format_exc()
        print exc_str

        if retry_failed:
            shell_safe_exc = [('# %s\n' % l) for l in tb.split('\n')]

            # Put the command to the end of the queue (in case it will always fail)
            append('# The following command failed or was cancelled; error message:')
            extend(shell_safe_exc)
            append(command)

        if kill:
            raise

    return ret


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-g', '--gpu', type=str, required=True,
        help='''Which gpu(s) to employ for all commands launched from this
        process'''
    )
    parser.add_argument(
        '--timeout', type=int, default=DFLT_TIMEOUT,
        help='''Default timeout (seconds)'''
    )
    args = parser.parse_args()
    return args


def main():
    queue_start_time = time.time()
    args = parse_args()
    print ('Queue worker process started to launch jobs with'
           ' CUDA_VISIBLE_DEVICES=%s' %args.gpu)

    previous_command_none = False
    while True:
        command = get_command(args.timeout)

        if command is None:
            previous_command_none = True
            wstdout('.')
            time.sleep(30 + random.random())
            continue

        if previous_command_none:
            previous_command_none = False
            wstdout('\n')

        # Don't do anything for trivial commands
        stripped = command.strip()
        if len(stripped) == 0 or stripped.startswith('#'):
            continue

        print '='*80
        print 'Started: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        print 'Command to be run:\n%s' %command
        print ''
        print '>'*80
        t0 = time.time()
        ret = run_command(command=command, gpus=args.gpu)
        t1 = time.time()
        print '<'*80
        print ''
        print 'Finished: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        dt = t1 - t0
        print 'Command run time = %s (%s s)\n\n' %(hms(dt), dt)

    run_time = time.time() - queue_start_time
    print '\nQueue exhausted (file %s no longer exists).' %QUEUE_FPATH
    print 'Total run time = %s (%s sec)' %(hms(run_time), run_time)
    return


def hms(s):
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return '%02d:%02d:%s' %(h, m, s)


def wstdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def wstderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


if __name__ == '__main__':
    main()
