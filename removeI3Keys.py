#!/usr/bin/env python

##
## Take a bunch of i3 files and a txt file containing a list of frame objects and write a HDF file
##
## created by lschulte on 19/12/12
##

#############
## imports ##
#############


## system stuff
import sys
#from argparse import ArgumentParser
from optparse import OptionParser

## icecube stuff
from icecube import dataio, dataclasses, icetray, multinest_icetray
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from I3Tray import *

## icecube modules for converters
from icecube import clast, common_variables, fill_ratio, linefit, cscd_llh#, millipede, credo

## read the txt file easily
from numpy import loadtxt


###########
## Input ##
###########

#parser = ArgumentParser(description='Take a bunch of i3 files and a txt file containing a list of frame objects and write a HDF file')
parser = OptionParser(description='Take a bunch of i3 files and a txt file containing a list of frame objects and write a HDF file')
#parser.add_argument('outfile', metavar='OUTFILE', type=str, nargs=1,
#		    help='The output HDF5 file')
parser.add_option('-o', '--outfile', metavar='OUTFILE', type=str, dest='outfile',
		    help='The output I3 file name')
#parser.add_argument('infiles', metavar='INFILES', type=str, nargs='+', 
#		    help='The files you want to scan')
#parser.add_argument('-k', '--keys', dest='keys', action='store', type=str, nargs=1,
#		    help='The file containing the keys you want to write')
parser.add_option('-k', '--keys', dest='keys', action='store', type=str,
		    help='The file containing the keys you want to remove')

(options, args) = parser.parse_args()

## get the keys
if options.keys:
    keys = list(loadtxt(options.keys, dtype=str))
else:
    keys = []
    print 'You specified no key list. Writing only the I3EventHeaders.'

print 'I will read %d i3 files and write %d keys to %s' %(len(args), len(keys), options.outfile)


##############
## The Tray ##
##############

tray = I3Tray() 

tray.AddModule("I3Reader", "reader", filenamelist=args)
tray.AddModule("Delete",   "del",    keys=keys)
tray.AddModule("I3Writer", "writer", filename=options.outfile)
tray.AddModule("TrashCan", "byebye")

tray.Execute()
tray.Finish()
