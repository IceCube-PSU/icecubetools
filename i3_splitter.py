#!/usr/bin/env python


from __future__ import division, print_statement, with_statement

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from i3Utils import Split

import os
import copy
import operator
import re
import sys
import tables
import textwrap
import time

import numpy as np


def parse_args():
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
    parser.add_argument(
        '--n-per-file', type=int, required=True,
        help='Maximum number of events per output file'
    )
    parser.add_argument(
        '--n-total', type=int, default=None,
        help='Total events to split out into sub-files'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args_dict = vars(args)
    split = Split(**args_dict)
    t0 = time.time()
    split.split()
    t1 = time.time()
    dt = t1 - t0
    print('\nTotal time: %0.3f s; time per event: %0.3f ms'
          % (dt, dt/(split.event_number+1)*1000))


if __name__ == '__main__':
    main()
