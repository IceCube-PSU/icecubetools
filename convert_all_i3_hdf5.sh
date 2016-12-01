#!/bin/bash

START=`date`
echo $START

HDFWRITER="${PSUI3TOOLS}/hdf_writer.py"

[[ ($# -eq 3) || ($# -eq 5) ]] || { echo "Supply 3 or 5 arguments"; exit 1; }

SOURCE_DIR=$1
TARGET_DIR=$2
KEYFILE=$3

(( $# == 3 )) && { MODULO=1; OFFSET=0; }
(( $# == 5 )) && { MODULO=$4; OFFSET=$5; }

echo "HDFWRITER  = "${HDFWRITER}
echo "SOURCE DIR = "${SOURCE_DIR}
echo "TARGET DIR = "${TARGET_DIR}
echo "KEYFILE    = "${KEYFILE}
echo "MODULO     = "${MODULO}
echo "OFFSET     = "${OFFSET}

COUNT=-1
while IFS= read -r -d '' FILE
do
	(( COUNT++ ))
	(( ($COUNT - $OFFSET) % $MODULO != 0 )) && continue;
	NOEXTFILE=$(echo "$FILE" | sed --regexp-extended 's/\.i3(\.bz2){0,1}$//I')
	echo $NOEXTFILE
	BASENAME=$(basename ${NOEXTFILE})
	echo $BASENAME
	echo ""
	echo "> Input file  : "${FILE}
	echo "> Output file : "${TARGET_DIR}/${BASENAME}.hdf5
	[[ -e ${TARGET_DIR}/${BASENAME}.hdf5 ]] && { echo "> ... skipping, output file already exists"; continue; }
	${HDFWRITER} ${FILE} -k ${KEYFILE} -o ${TARGET_DIR}/${BASENAME}.hdf5 && { echo "> ... converted successfully."; } || { echo "> ... error in conversion!"; }
done < <(find -L ${SOURCE_DIR} -maxdepth 1 -regextype posix-egrep -iregex '.*.i3(.bz2){0,1}$' -type f -print0)

echo "Started:  "$START
echo "Finished: "`date`
