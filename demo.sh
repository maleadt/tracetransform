#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

scilab-cli -nb -e "exec('$DIR/demo.sce'); exit" -args "$@" \
	| egrep --line-buffered -v "Truecolor Image" &
trap "kill -15 $!; exit 1" SIGINT
wait

