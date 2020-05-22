#!/usr/bin/env python

# Copyright (c) 2014-2015
# Justin L. Lanfranchi <jll1062@phys.psu.edu>
# and the IceCube Collaboration <http://www.icecube.wisc.edu>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
# OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
#
# $Id$
#
# @file 2014-12-16_step2.2_i3llhp_summary_to_binfiles.py
# @version $Revision$
# @date $Date$
# @author J.L. Lanfranchi

"""
Extract from I3 file the log likelihood and corresponding MultiNest parameter
values (LLHP) from HybridReco run(s) contained in the file.
"""


import argparse
import re
import time

import numpy as np

from genericUtils import timediffstamp


EXTRACT_LLHP_HSUMM = False
"""Extract horizontal summary of the LLH vs. Param value data?"""

LLH_MINSTEPSIZE = 0.1
"""LLH stepsize traversing down "horizontal (LLH-vs-parameter) summary"""

EXTRACT_LLHP_ALL = False
"""Extract ALL LLHP data from the I3 files?"""

# Extract only LLHP values with LLH values this far from max-LLH?
# If specify >= 0, no "top" LLHP data is extracted.
# NOTE: Keep 99.9% of top in linear-likelihood space
#       => value of np.log(0.001) = -6.9077552789821368
#EXTRACT_LLHP_TOP_THRESH = np.log(0.001)
#===============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--infile',
        type=str,
        required=True,
        #nargs='+',
        help='Input file(s) to process'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        required=False,
        default=None,
        help='''Directory into which to write the binary file(s) (defaults is
        same directory as --infile)'''
    )
    parser.add_argument(
        '--llhp-top-thresh',
        type=float,
        required=False,
        default=np.log(1e-5),
        help='''Keep top log likelihoods down to this threshold value (should be
        negative to set a threshold; a value greater than or equal to 0 keeps
        *all* LLHP). E.g., log(0.001) = -6.9 keeps 99.9% of LLHP's'''
    )
    parser.add_argument(
        '--llhp-top-min-llhp',
        type=int,
        required=True,
        #default=50,
        help='''Minimum number of LLHP values to keep (regardless of how far down
        in LLH we must go)'''
    )
    args = parser.parse_args()
    return args


## "MAIN" ##
t0 = time.time()
args = parse_args()

# Import these things here since they can take a long time to import (or not
# import at all) but if e.g. we just want to display help message, no need to
# wait for silly icecube software bs to run first.
from I3Tray import *
from icecube import icetray, dataclasses, dataio, ExtractLLHP, multinest_icetray

import i3Utils



# TODO: 'Test' still sends files to same directory as non-'Test' run!

if TEST:
    COUNT_PER_RUN = (10,)

for (RUN_NUM, CONTINUE_FROM_FILENUM, PROC_N) in zip(RUNS, CONTINUE_FROM_FILENUMS, PROC_N_FILES):
    tr0 = time.time()
    print ''

    # Compile a regex to extract the file number from the file name
    fnum_rex = re.compile(r'.*([0-9]{6})\.i3(\.bz2){0,1}$')

    # Find i3 files to process, specify target directory for binary data files
    if PGEN_PRESENT:
        fileLoc = PGEN.FileLocations(run=RUN_NUM)
        i3_src_dir = fileLoc.source_i3_dir
        llhpbin_tgt_dir = fileLoc.llhp_binfiles_dir
        if TEST:
            fileLoc.llhp_binfiles_dir += '_test'
        fileLoc.createDirs(which='llhp_bin')
    else:
        i3_src_dir = I3_SRC_DIRS[RUN_NUM]
        llhpbin_tgt_dir = LLHPBIN_TGT_DIRS[RUN_NUM]

    print 'root dir:', i3_src_dir
    print 'llhpbin_tgt_dir:', llhpbin_tgt_dir

    tr1 = time.time()
    print 'tr1-tr0', timediffstamp(tr1-tr0)

    count = 0
    for filenum, (ffpath, basename, match) in enumerate(fiter):
        tl0 = time.time()
        #filenum = int(fnum_rex.findall(ffpath)[0][0])

        # Print info particular to this file's LLHP extraction
        print 'ffpath ; fnum_rex finds...:', ffpath, ';', fnum_rex.findall(ffpath)
        print 'filenum:', filenum
        print 'extract_llhp_hsumm', EXTRACT_LLHP_HSUMM
        if EXTRACT_LLHP_HSUMM:
            print 'llh_minstepsize:', LLH_MINSTEPSIZE
        print 'extract_llhp_all:', EXTRACT_LLHP_ALL
        print 'llhp_top_thresh:', args.llhp_top_thresh
        print 'llhp_top_min_llhp:', args.llhp_top_min_llhp
        print 'must pass cuts?:', PASS_CUTS_PRESENT

        # Put the I3 tray together with all modules and above-specified
        # parameters
        tray = I3Tray()
        tray.AddModule('I3Reader', 'reader', Filename=ffpath)
        #tray.AddModule(passAllCuts, 'passAllCuts')
        tray.AddModule(
            'ExtractLLHP', 'ExtractLLHP',
            outdir=llhpbin_tgt_dir,
            file_number=filenum,
            llh_minstepsize=LLH_MINSTEPSIZE,
            extract_llhp_hsumm=EXTRACT_LLHP_HSUMM,
            extract_llhp_all=EXTRACT_LLHP_ALL,
            EXTRACT_LLHP_TOP_THRESH=args.llhp_top_thresh,
            EXTRACT_LLHP_TOP_MIN_LLHP=args.llhp_top_min_llhp
        )
        tray.AddModule('TrashCan', 'Done')
        tl1 = time.time()
        print 'tl1-tl0', timediffstamp(tl1-tl0)
        if TEST:
            tray.Execute(100)
        else:
            tray.Execute()
        tl2 = time.time()
        print 'tl2-tl0', timediffstamp(tl2-tl0)
        tray.Finish()
        tl3 = time.time()
        print 'tl3-tl0', timediffstamp(tl3-tl0)
    tr2 = time.time()
    print 'tr2-tr0', timediffstamp(tr2-tr0)
t1 = time.time()
print 't1-t0', timediffstamp(t1-t0)
