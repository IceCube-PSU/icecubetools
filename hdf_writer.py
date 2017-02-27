#!/usr/bin/env python
# authors: Created by lschulte on 19/12/12
#          Updated by J.L. Lanfranchi, 2016 (jll1062@phys.psu.edu)
"""
Take an IceCube I3 data file (or files) and a txt file containing a list of
frame object names and write a single HDF5 file containing all specified
objects.

"""


from argparse import ArgumentParser
import os

from numpy import loadtxt

from I3Tray import *
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService


def mkdir(d, mode=0750, warn=True):
    d = os.path.expandvars(os.path.expanduser(d))
    if warn and os.path.isdir(d):
        print 'Directory %s already exists.' % d

    try:
        os.makedirs(os.path.expandvars(os.path.expanduser(d)), mode=mode)
    except OSError as err:
        if err[0] != 17:
            raise err
    else:
        print 'Created directory: ' + d


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '-o', '--outfile', metavar='OUTFILE', type=str, required=True,
        help='HDF5 file path that will be generated'
    )
    parser.add_argument(
        '-k', '--keys', type=str, required=False, default=None,
        help='''Key file path (contains the I3 frame object names) you want to
        write to the HDF5 file'''
    )
    parser.add_argument(
        'infiles', metavar='INFILES', nargs='+',
        help='List of I3 file path(s) to be converted'
    )
    args = parser.parse_args()
    args.outfile = os.path.expandvars(os.path.expanduser(args.outfile))
    return args


def main():
    # Make sure args are valid, and provide help, before importing all the
    # IceCube dreck
    args = parse_args()
    assert (args.outfile.lower().endswith('.hdf5')
            or args.outfile.lower().endswith('.hdf')
            or args.outfile.lower().endswith('.h5')
            or args.outfile.lower().endswith('.hd5'))

    from icecube import icetray

    # One or more of the following imports is actually necessary, but many may
    # not be necessary to convert files. Rather than try to figure out which is
    # which, all available imports (that don't fail) are included here.

    from icecube import multinest_icetray

    #from icecube import AtmCscdEnergyReco
    from icecube import gulliver_modules
    #from icecube import production_histograms
    from icecube import BadDomList
    from icecube import hdfwriter
    from icecube import pybdtmodule
    from icecube import CascadeVariables
    from icecube import icepick
    #from icecube import recclasses
    from icecube import CoincSuite
    from icecube import rootwriter
    from icecube import DeepCore_Filter
    #from icecube import core_removal
    from icecube import improvedLinefit
    from icecube import shield
    from icecube import DomTools
    from icecube import cramer_rao
    from icecube import interfaces
    #from icecube import shovelart
    from icecube import HiveSplitter
    from icecube import credo
    from icecube import ipdf
    #from icecube import shovelio
    from icecube import IceHive
    from icecube import cscd_llh
    #from icecube import level3_filter_cascade
    from icecube import simclasses
    from icecube import KalmanFilter
    from icecube import daq_decode
    #from icecube import level3_filter_lowen
    from icecube import spline_reco
    from icecube import NoiseEngine
    from icecube import dataclasses
    #from icecube import level3_filter_muon
    from icecube import static_twc
    from icecube import SLOPtools
    from icecube import dataio
    from icecube import lilliput
    from icecube import steamshovel
    from icecube import STTools
    from icecube import dipolefit
    from icecube import linefit
    from icecube import tableio
    #from icecube import SeededRTCleaning
    from icecube import double_muon
    from icecube import load_pybindings
    from icecube import tensor_of_inertia
    from icecube import TopologicalSplitter
    from icecube import dst
    from icecube import millipede
    #from icecube import test_unregistered
    from icecube import VHESelfVeto
    from icecube import fill_ratio
    #from icecube import mue
    #from icecube import topeventcleaning
    from icecube import WaveCalibrator
    from icecube import filter_tools
    #from icecube import ophelia
    #from icecube import toprec
    #from icecube import astro
    #from icecube import filterscripts
    from icecube import paraboloid
    from icecube import tpx
    #from icecube import bayesian_priors
    from icecube import finiteReco
    from icecube import payload_parsing
    from icecube import trigger_sim
    from icecube import clast
    #from icecube import frame_object_diff
    from icecube import photonics_service
    from icecube import trigger_splitter
    from icecube import coinc_twc
    #from icecube import full_event_followup
    from icecube import photospline
    from icecube import wavedeform
    from icecube import common_variables
    from icecube import gulliver
    from icecube import phys_services
    from icecube import wavereform
    from icecube import common_variables__direct_hits
    #from icecube import gulliver_bootstrap
    #from icecube import portia

    if args.keys:
        keys = list(loadtxt(args.keys, dtype=str))
        book_everything = False
        num_keys = len(keys)
    else:
        keys = [] #['I3EventHeader']
        book_everything = True
        num_keys = 'all'
        print 'You specified no key list. Writing everything!'
        #print 'You specified no key list. Writing only the I3EventHeaders.'

    print ''
    print '='*79
    print 'Will read %d i3 file(s) and write %s key(s) to %s' \
            %(len(args.infiles), num_keys, args.outfile)
    print '='*79
    print ''

    outdir = os.path.dirname(args.outfile)
    if outdir not in ['', '.', './']:
        mkdir(outdir, warn=False)

    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', filenamelist=args.infiles)
    hdf_service = I3HDFTableService(args.outfile)
    tray.AddModule(
        I3TableWriter, 'writer',
        tableservice=hdf_service,
        keys=keys,
        BookEverything=book_everything,
        SubEventStreams=['fullevent', 'SLOPSplit', 'InIceSplit', 'in_ice',
                         'nullsplitter']
    )
    tray.AddModule('TrashCan', 'byebye')
    tray.Execute()
    tray.Finish()


if __name__ == '__main__':
    main()
