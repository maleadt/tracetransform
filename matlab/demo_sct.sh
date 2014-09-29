#!/bin/bash

$(dirname $(readlink -f $0))/demo.pl -singleCompThread "$@"
