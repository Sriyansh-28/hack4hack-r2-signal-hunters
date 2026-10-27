#!/usr/bin/env bash
# Submit a Round 2 step script to PBS.  Usage:  bash ~/Hack4hack/R2/code/submit2.sh <script.sh> [walltime] [cpu|after:JOBID]
#   cpu = CPU-only job in the cpuq queue (never waits for a GPU slice)
#   after:JOBID = start only after that job ends (keeps ONE GPU job at a time - cluster rule)
S=${1:?usage: submit2.sh script.sh [walltime] [cpu]}
WT=${2:-02:00:00}
cd "$HOME/Hack4hack/R2/code" || exit 1
sed -i 's/\r$//' ./*.sh ./*.pbs ./*.py 2>/dev/null          # Windows line endings -> Unix
if [ "${3:-}" = cpu ]; then
  id=$(/opt/pbs/bin/qsub -q cpuq -l select=1:ncpus=8:mem=24gb -l walltime="$WT" -v STEPS="$S",GPU=0 job2.pbs)
elif [ "${3:-}" != "${3#cpu-after:}" ]; then
  id=$(/opt/pbs/bin/qsub -q cpuq -l select=1:ncpus=8:mem=24gb -l walltime="$WT" -W depend=afterany:"${3#cpu-after:}" -v STEPS="$S",GPU=0 job2.pbs)
elif [ "${3:-}" != "${3#after:}" ]; then
  id=$(/opt/pbs/bin/qsub -W depend=afterany:"${3#after:}" -l walltime="$WT" -v STEPS="$S" job2.pbs)
else
  id=$(/opt/pbs/bin/qsub -l walltime="$WT" -v STEPS="$S" job2.pbs)
fi
echo "submitted $S as $id  (status: /opt/pbs/bin/qstat -u \$USER ; log: tail -f ~/Hack4hack/R2/run.log)"
