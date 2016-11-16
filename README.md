icecubetools
============

Generic tools for working with IceCube software


Converting I3 to HDF5
---------------------
Use `convert_all_i3_hdf5.sh` to convert a batch of I3 files to HDF5 (the "icetray" format, _not_ the PISA format HDF5 file; this file can then be converted to PISA HDF5 using utils within the PISA project).

First arg is source dir, second arg is dest dir, and third arg is the keyfile containing keys to grab. Optionally you can specify fourth and fifth args: Fourth arg is the count modulus (number of processes you'll run in parallel) and the fifth arg is the "offset". If the count is 4, then the offset can be 0, 1, 2, or 3.

E.g.:
```bash
# Load an appropriate IceTray software environment
icerec_V05-00-05

# Define location of the icecubetools dir
export PSUI3TOOLS="$HOME/icecubetools"

# Perform the conversion
$PSUI3TOOLS/icecubeconvert_all_i3_hdf5.sh \
    /path/to/source/i3/files \
    /path/to/dest/hdf5/dir \
    $PSUI3TOOLS/interesting_keys/run1003_proc_v5.1.nopid.txt
```

To convert files in parallel, replace the last line with the following and supply the 4th and 5th args (32 processes are run in parallel in the below example)
```bash
t0=`date`
for i in {0..32}
do
    $PSUI3TOOLS/convert_all_i3_hdf5.sh \
        /path/to/source/i3/files \
        /path/to/dest/hdf5/dir \
        $PSUI3TOOLS/interesting_keys/run1003_proc_v5.1.nopid.txt \
        32 \
        $i \
        &
done
wait
t1=`date`
echo "Script was started at $t0"
echo "Script completed at $t1"
```
