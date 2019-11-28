#!/bin/bash
#
#SBATCH  --mail-type=ALL                      # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=/itet-stor/matthmey/net_scratch/logs/log%j.log      # where to store the output ( %j is the JOBID )
#SBATCH  --cpus-per-task=1                    # Use 16 CPUS
#SBATCH  --gres=gpu:1                         # Use 1 GPUS
#SBATCH  --mem=32G                            # use 32GB
#SBATCH  --account=tik                        # we are TIK!

#eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"
source activate permafrost
#
echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
cd /home/matthmey/data/projects/stuett/frontends/permafrostanalytics/
python -u ideas/machine_learning/classification.py -p /home/perma/permasense_vault/datasets/permafrost_hackathon/ -l --classifier seismic
echo finished at: `date`
exit 0;

