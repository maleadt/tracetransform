#!/bin/bash

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
CALLPATH=$PWD

# Manage parameters
if [ $# -ne 2 ]; then
    echo "Please provide input filename and rotation angle"
    exit 1
fi

# Call MATLAB
cd $SCRIPTPATH
matlab -nodisplay -r "rottest $CALLPATH/$1 $2; quit" | tail -n+11
cd $CALLPATH
