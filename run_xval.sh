#!/usr/bin/env bash
# CPU job (~4 min): how harsh is the PKS-like simulation? (xval.py)
set -uo pipefail
cd "$(dirname "$0")"
python xval.py
