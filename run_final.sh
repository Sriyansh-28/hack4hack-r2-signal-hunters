#!/usr/bin/env bash
# Final GPU job (one MIG slice, ~45 min): CNN v2 + pseudo-labels (~25 min), CNN v2 without features (~17 min),
# then the final stackers. Each model saves its own predictions, so a later failure does not lose earlier work.
set -uo pipefail
cd "$(dirname "$0")"
bash run_v2pl.sh     || echo "!!! run_v2pl.sh failed"
bash run_v2notab.sh  || echo "!!! run_v2notab.sh failed"
bash run_stack_pl.sh || echo "!!! run_stack_pl.sh failed"
