#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
octave-cli --silent "$DIR/demo.m" "$@"

