#!/usr/bin/env bash
# (~1 min) what is in the PKS ON panels? (diag2.py) -> run.log and ../diag2.txt
set -uo pipefail
cd "$(dirname "$0")"
python diag2.py | tee ../diag2.txt
