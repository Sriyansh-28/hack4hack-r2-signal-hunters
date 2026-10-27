#!/usr/bin/env bash
# (~1 min) which PKS cadences carry the strong ON broadband sweeps + EM estimate of the test label mix -> ../diag3.txt
set -uo pipefail
cd "$(dirname "$0")"
python diag3.py | tee ../diag3.txt
