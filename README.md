icecubetools
============

Generic tools for working with IceCube software


Converting I3 to HDF5
---------------------
Use `convert_all_i3_hdf5.sh` to convert a batch of I3 files to HDF5 (the "icetray" format, _not_ the PISA format HDF5 file; this file can then be converted to PISA HDF5 using utils within the PISA project).

First arg is source dir, second arg is dest dir, and third arg is the keyfile containing keys to grab.
E.g.:
```bash
export PSUI3TOOLS="$HOME/icecubetools"

$PSUI3TOOLS/icecubeconvert_all_i3_hdf5.sh \
    /path/to/source/i3/files \
    /path/to/dest/hdf5/dir \
    $PSUI3TOOLS/interesting_keys/run1003_proc_v5.1.nopid.txt
```
