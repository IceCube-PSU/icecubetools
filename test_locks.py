#!/usr/bin/env python

import sys
import time
from repeated_reco import acquire_lock, cleanup_lock_f

hold_time = 0.1
wait_time = 0.01
held = 0.0
not_held = 0.0

try:
    while True:
        try:
            lock_f = acquire_lock('/storage/group/dfc13_collab/lock.delme')
        except IOError, err:
            if err.errno != 11:
                raise
            not_held += 1
            lock_f = None
            time.sleep(wait_time)
        else:
            held += 1
            time.sleep(hold_time)
        finally:
            if lock_f is not None:
                cleanup_lock_f(lock_f)
finally:
    sys.stdout.write(
        '\n\nheld for %.4f pct of time\n\n'
        % (100 * held*hold_time / (held*hold_time + not_held*wait_time))
    )

