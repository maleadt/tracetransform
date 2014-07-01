#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
scilab-cli -nb -e "exec('$DIR/demo.sce'); exit" -args "$@"
