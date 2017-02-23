#!/usr/bin/env python


from __future__ import with_statement

import argparse
import os
import re
import socket
from subprocess import Popen, PIPE


__all__ = ['CACHEFILE_NAME', 'HOST_TO_GROUP_MAPPING', 'GROUP', 'CLUSTERNAME',
           'getclustername_whois', 'getclustername_dig',
           'getclustername_tracepath', 'getclustername', 'getclustergroup',
           'get_spline_tables', 'get_folder_name', 'get_gcd_folder']


CACHEFILE_NAME = os.path.expandvars("${HOME}/.cluster.name")

HOST_TO_GROUP_MAPPING = {
    "mcgill.ca": "Guillimin",
    "user.msu.edu": "USER-JP",
    "msu.edu": "MSU",
    "wisc.edu": "Madison",
    "westgrid.ca": "Jasper",
    "scinet.utoronto.ca": "GPC",
    "aci.ics.psu.edu": "PSU_ACI",
    "rcc.psu.edu": "PSU_RCC",
    "ifh.de": "Desy",
    "desy.de": "Desy"
}


def getclustername_whois():
    """Get cluster name (hostname) via whois.

    Unfortunately whois is generally not available, so even if this is a good
    method it is not preferred
    """
    public_ip_child = Popen(
        "curl -s icanhazip.com",
        shell=True,
        stdout=PIPE,
        universal_newlines=True
    )
    public_ip = str(public_ip_child.stdout.readline()).strip()

    cmd = 'whois {:s} |grep Organization |head -n1|sed "s/^Organization:   //"'
    host_child = Popen(
        cmd.format(public_ip), shell=True, stdout=PIPE, universal_newlines=True
    )

    host = str(host_child.stdout.readline()).strip()
    if len(host) == 0:
        raise ValueError('Could not get hostname via whois.')

    return host


def getclustername_dig():
    """Get cluster name (hostname) via dig."""
    public_ip_child = Popen(
        "curl -s icanhazip.com",
        shell=True,
        stdout=PIPE,
        universal_newlines=True
    )
    public_ip = str(public_ip_child.stdout.readline()).strip()

    host_child = Popen(
        "dig -x {:s} +short @8.8.8.8".format(public_ip),
        shell=True,
        stdout=PIPE,
        universal_newlines=True
    )

    host = str(host_child.stdout.readline()).strip()
    if len(host) == 0:
        raise ValueError('Could not get hostname via dig.')

    return host


def getclustername_tracepath():
    """Get cluster name (hostname) via tracepath. This is very slow if it
    works, but faster if it fails.
    """
    child = Popen('tracepath google.com', shell=True, stdout=PIPE)
    re1 = re.compile(":[ ]*([^ ]*) ")
    re2 = re.compile("[a-zA-Z]")

    while True:
        line = str(child.stdout.readline()).strip()
        if len(line) == 0:
            break
        if 'LOCALHOST' in line:
            continue
        match = re1.search(line)
        host = match.group(1)
        if re2.match(host) is None:
            continue
        break
    child.terminate()

    if host is None or len(host) == 0:
        raise ValueError('Could not get hostname via tracepath.')

    return host


def _write_host_to_cache(host):
    try:
        with file(CACHEFILE_NAME, 'w') as cachefile:
            cachefile.write(host + '\n')
    except (IOError, OSError):
        pass


def getclustername():
    """Get cluster name, trying various methods."""

    # Try fast and simple way first...

    # 1. socket
    host = socket.gethostname()
    host_lower = host.lower()
    if any([(s.lower() in host_lower) for s in HOST_TO_GROUP_MAPPING.keys()]):
        return host

    # Move on to slower methods next...

    # 2. non-trivial string in the cache file
    if os.path.isfile(CACHEFILE_NAME):
        with file(CACHEFILE_NAME, 'r') as cachefile:
            host = str(cachefile.readline()).strip()
        if len(host) > 0:
            return host

    # 3. dig
    try:
        host = getclustername_dig()
    except:
        pass
    else:
        _write_host_to_cache(host)
        return host

    # 4. tracepath
    try:
        host = getclustername_tracepath()
    except:
        pass
    else:
        _write_host_to_cache(host)
        return host

    # 5. whois
    try:
        host = getclustername_whois()
    except:
        pass
    else:
        _write_host_to_cache(host)
        return host

    return "unknown"


CLUSTERNAME = getclustername()


def getclustergroup():
    host = CLUSTERNAME.lower()
    group = None
    for str_fragment, grp in HOST_TO_GROUP_MAPPING.items():
        if str_fragment in host:
            group = grp

    if group is None:
        print("Unknown group of host {:s}".format(host))
        return None

    return group


GROUP = getclustergroup()


def get_spline_tables():
    if GROUP in ["GPC"]:
        cascade_spline_dir = "/scratch/project/d/dgrant/jpandre/spline_tables/cascades/SPICEMie"
        track_spline_dir   = "/scratch/project/d/dgrant/jpandre/spline_tables/muon_zerolength/SPICEMie"
        photonics_dir      = "/scratch/project/d/dgrant/jpandre/photon-tables/SPICEMie/"

    elif GROUP in ["Guillimin"]:
        cascade_spline_dir = "/gs/project/ngw-282-aa/tables/spline_tables/cascades/SPICEMie"
        track_spline_dir   = "/gs/project/ngw-282-aa/tables/spline_tables/muon_zerolength/SPICEMie"
        photonics_dir      = "/gs/project/ngw-282-aa/tables/photon-tables/SPICEMie/"

    elif GROUP in ["Jasper"]:
        cascade_spline_dir = "/home/jpa14/scratch/spline_tables/cascades/SPICEMie"
        track_spline_dir   = "/home/jpa14/scratch/spline_tables/muon_zerolength/SPICEMie"
        photonics_dir      = "/home/jpa14/scratch/photon-tables/SPICEMie/"

    elif GROUP in ["MSU"]:
        cascade_spline_dir = "/mnt/research/IceCube/tables/spline_tables/cascades/SPICEMie"
        track_spline_dir   = "/mnt/research/IceCube/tables/spline_tables/muon_zerolength/SPICEMie"
        photonics_dir      = "/mnt/research/IceCube/tables/photon-tables/SPICEMie/"

    elif GROUP in ["USER-JP"]:
        cascade_spline_dir = "/home/jp/stock/I3Files/spline_tables/cascades/SPICEMie"
        track_spline_dir   = "/home/jp/stock/I3Files/spline_tables/muon_zerolength/SPICEMie"
        photonics_dir      = "/home/jp/stock/I3Files/photon-tables/SPICEMie/"

    elif GROUP in ["PSU_ACI"]:
        cascade_spline_dir = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines"
        track_spline_dir   = "/storage/group/dfc13_collab/spline_tables/muon_zero"
        photonics_dir      = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/"

    elif GROUP in ["PSU_RCC"]:
        raise NotImplementedError('Please enter appropriate values in the script for RCC!')
        #cascade_spline_dir =
        #track_spline_dir   =
        #photonics_dir      =

    else:
        # Use spline tables from CVMFS
        cascade_spline_dir = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines"
        track_spline_dir   = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/splines"
        photonics_dir      = "/cvmfs/icecube.opensciencegrid.org/data/photon-tables/SPICEMie/"

    PhotoSplineAmplitudeTableCscd  = cascade_spline_dir + '/ems_mie_z20_a10.abs.fits'
    PhotoSplineTimingTableCscd     = cascade_spline_dir + '/ems_mie_z20_a10.prob.fits'
    PhotoSplineAmplitudeTableMuon  = track_spline_dir + '/ZeroLengthMieMuons_150_z20_a10.abs.fits'
    PhotoSplineTimingTableMuon     = track_spline_dir + '/ZeroLengthMieMuons_150_z20_a10.prob.fits'

    spline_tables = dict()
    spline_tables["cscd_amplitude"] = PhotoSplineAmplitudeTableCscd
    spline_tables["cscd_timing"]    = PhotoSplineTimingTableCscd
    spline_tables["trk_amplitude"]  = PhotoSplineAmplitudeTableMuon
    spline_tables["trk_timing"]     = PhotoSplineTimingTableMuon
    spline_tables["photonics_dir"]  = photonics_dir

    return spline_tables


def get_folder_name(step):
    if GROUP in ["GPC", "MSU", "Guillimin"]:
        folder = os.path.expandvars("$SCRATCH")
    elif GROUP in ["Jasper"]:
        folder = os.path.expandvars("$HOME/scratch")
    elif GROUP in ["Madison"]:
        folder = os.path.expandvars("/data/user/$USER/scratch")
    elif GROUP in ["USER-JP"]:
        folder = os.path.expandvars("$HOME/stock/I3Files/pingu")
    else:
        folder = os.path.expandvars("$HOME/pingu")
    folder  = folder + ("/pingu-std-proc_Step%s" % step)

    return folder


def get_gcd_folder():
    gcd_folder = ""
    if GROUP in ["GPC"]:
        gcd_folder = "/scratch/project/d/dgrant/jpandre/gcd_folder"
    elif GROUP in ["Guillimin"]:
        gcd_folder = "/gs/project/ngw-282-aa/gcd_folder"
    elif GROUP in ["Jasper"]:
        gcd_folder = "/lustre/home/jpa14/scratch/pingu/gcd_creation"
    elif GROUP in ["MSU"]:
        gcd_folder = "/mnt/research/IceCube/gcd_file"
    elif GROUP in ["USER-JP"]:
        gcd_folder = "/home/jp/stock/I3Files/gcd"
    else: #if GROUP in ["Madison", "Desy", "none"]:
        # Use GCDs from CVMFS
        gcd_folder = "/cvmfs/icecube.opensciencegrid.org/data/GCD"
    return gcd_folder


if __name__ == '__main__':
    gl_parser = argparse.ArgumentParser(description="Cluster discovery tool.")
    parser = gl_parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--whereami" ,
        action="store_true",
        help = "Simplified host/group printing"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help = "Print all information that can be discovered with python calls"
    )
    parser.add_argument(
        "--gcd",
        action="store_true",
        help = "return path for GCD files"
    )
    parser.add_argument(
        "--step",
        default="",
        help='return standard folder name for step "STEP"'
    )
    args = gl_parser.parse_args()

    if args.gcd:
        print(get_gcd_folder())
    elif args.step != "":
        print(get_folder_name(args.step))
    elif args.whereami:
        print("Identifier hostname='%s' placed in group='%s'"
              % (CLUSTERNAME, GROUP))
    else:
        args.debug = True

    if args.debug:
        print("Identifier hostname='%s' placed in group='%s'"
              % (CLUSTERNAME, GROUP))
        print("=====================")
        print("Step0 folder: %s" % get_folder_name(0))
        print("=====================")
        print("GCD   folder: %s" % get_gcd_folder())
        print("=====================")
        print("Spline tables:")
        tables = get_spline_tables()
        for key in tables.keys():
            print(" %15s => %s" % (key, tables[key]))
