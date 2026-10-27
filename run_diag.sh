#!/usr/bin/env bash
# CPU job (~1 min): PKS vs GBT diagnostics -> run.log and ../diag.txt
set -uo pipefail
cd "$(dirname "$0")"
python diag.py | tee ../diag.txt
