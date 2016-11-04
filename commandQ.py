#!/usr/bin/env python

import fcntl
import os
import popen
import random


QUEUE_FPATH = os.path.expandvars(os.path.expanduser('~/commandQ.queue'))
LOCK_FPATH = os.path.expandvars(os.path.expanduser('~/commandQ.lock'))


def get_command(self, timeout=10):
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
                os.path.unlink(QUEUE_FPATH)
            else:
                with open(QUEUE_FPATH, 'w') as queue:
                    queue.writelines(commands)
            return this_command

        finally:
            lockf.close()

    raise TimeoutError('Failed to get a command in %s sec.' %timeout)


#class Worker(object):
#    def __init__(self):
#        pass

        
