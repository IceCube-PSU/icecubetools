#!/usr/bin/env python

"""
Take a bunch of i3 files and a txt file containing a list of frame objects and
write a HDF file

created by lschulte on 19/12/12
"""

from optparse import OptionParser

from numpy import loadtxt

from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from I3Tray import *


if __name__ == '__main__':
    parser = OptionParser(
        description='''Take a bunch of i3 files and a txt file containing a
        list of frame objects and write a HDF file'''
    )
    parser.add_option(
        '-o', '--outfile', metavar='OUTFILE', type=str, dest='outfile',
        help='The output HDF5 file'
    )
    parser.add_option(
        '-k', '--keys', dest='keys', action='store', type=str,
        help='The file containing the keys you want to write'
    )
    (options, args) = parser.parse_args()

    if options.keys:
        keys = list(loadtxt(options.keys, dtype=str))
    else:
        keys = ['I3EventHeader']
        print 'You specified no key list. Writing only the I3EventHeaders.'

    print 'I will read %d i3 files and write %d keys to %s' \
            %(len(args), len(options.keys), options.outfile)

    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', filenamelist=args)
    hdf_service = I3HDFTableService(options.outfile)
    tray.AddModule(
        I3TableWriter, 'writer',
        tableservice=hdf_service,
        keys=keys,
        #BookEverything=True,
        SubEventStreams=['fullevent', 'SLOPSplit', 'InIceSplit', 'in_ice', 'nullsplitter']
    )
    tray.AddModule('TrashCan', 'byebye')
    tray.Execute()
    tray.Finish()
