#!/usr/bin/env python


from argparse import ArgumentParser
from collections import Sequence
import fcntl
import os
import random
import subprocess
import time


QUEUE_FPATH = os.path.expandvars(os.path.expanduser('~/command.queue'))
LOCK_FPATH = os.path.expandvars(os.path.expanduser('~/command.lock'))


def get_command(timeout):
    t0 = time.time()
    timeout_time = t0 + timeout
    while time.time() < timeout_time:
        lockf = open(LOCK_FPATH, 'w')
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError, err:
            if err.errno == 11:
                time.sleep(random.random()/10.0)
                continue
            else:
                raise
        try:
            with open(QUEUE_FPATH, 'r') as queue:
                commands = queue.readlines()
                this_command = commands.pop().strip()
            if len(commands) == 0:
                os.remove(QUEUE_FPATH)
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
    raise TimeoutError('Failed to get a command in %s sec.' %timeout)


def run_command(command, cpu_threads=1, cpu_cores=None, gpus=None):
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
        # export CUDA_VISIBLE_DEVICES=gpus
        env['CUDA_VISIBLE_DEVICES'] = gpus

    env['MKL_NUM_THREADS'] = format(cpu_threads, 'd')
    env['OMP_NUM_THREADS'] = format(cpu_threads, 'd')

    ret = subprocess.call(command, env=env, shell=True,
                          stderr=subprocess.STDOUT)

    return ret


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-g', '--gpu', type=str, required=True,
        help='''Which gpu(s) to employ for all commands launched from this
        process'''
    )
    parser.add_argument(
        '--timeout', type=int, default=10,
        help='''Default timeout (seconds)'''
    )
    args = parser.parse_args()
    return args


def main():
    queue_start_time = time.time()
    args = parse_args()

    while True:
        print '='*80
        print 'Getting a new command...'
        command = get_command(args.timeout)
        if command is None:
            break
        print 'Command to be run:\n    ' + command
        print '>'*80
        t0 = time.time()
        ret = run_command(command=command, gpus=args.gpu)
        t1 = time.time()
        print '<'*80
        print 'Command run time = %s sec' %(t1-t0)
        print '\n\n'

    run_time = time.time() - queue_start_time
    print '\nQueue exhausted (file %s no longer exists).' %QUEUE_FPATH
    print 'Total run time = %s sec' %run_time
    return


if __name__ == '__main__':
    main()
