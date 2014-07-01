#!/bin/bash
scilab-cli -nb -e 'exec("demo.sce"); exit' -args "$@" | tail -n +5

