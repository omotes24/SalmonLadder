#!/bin/bash
# usage: after.sh <file> <pattern> <command ...>   -- waits until <pattern> appears in <file>, then runs the command
F=$1; PAT=$2; shift 2
until grep -q "$PAT" "$F" 2>/dev/null; do sleep 15; done
exec "$@"
