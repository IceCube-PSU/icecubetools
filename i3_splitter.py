#!/usr/bin/env python

"""
Command-line interface to i3_utils.Split, used to split an I3 file into smaller
pieces.
"""

from __future__ import absolute_import, division, print_function, with_statement

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time

import numpy as np # pylint: disable=unused-import

from i3_utils import Split


def parse_args():
    """parse command-line arguments"""
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--infile', required=True,
        help='Input I3 filename'
    )
    parser.add_argument(
        '--outdir', default=None,
        help='Output directory'
    )
    #parser.add_argument(
    #    '--n-per-file', type=int, required=True,
    #    help='Maximum number of events per output file'
    #)
    parser.add_argument(
        '--n-total', type=int, default=None,
        help='''Total events to split out into sub-files; if not specified,
        take all (that pass --keep-criteria)'''
    )
    parser.add_argument(
        '--keep-criteria', type=str, default=None,
        help='''Criteria for choosing the event for splitting out; events that
        fail to meet the criteria will not count towards n-total or n-per-file.
        This will be evaluated where the `frame` variable is available to
        retrieve info from.'''
    )
    return parser.parse_args()


def main():
    """main"""
    args = parse_args()
    args_dict = vars(args)
    split = Split(**args_dict)
    t0 = time.time()
    split.split()
    t1 = time.time()
    dt = t1 - t0
    print('')
    print('Total number of frames read: %d' % split.all_frame_number)
    print('Total number of events read: %d' % split.all_event_number)
    print('Total number of events written: %d' % split.events_written)
    print('Total time: %0.3f s; time per event (whether kept or discarded):'
          ' %0.3f ms' % (dt, dt/(split.all_event_number + 1)*1000))


if __name__ == '__main__':
    main()
