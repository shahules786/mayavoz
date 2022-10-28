#!/bin/bash
set -e

echo '----------------------------------------------------'
echo ' SLURM_CLUSTER_NAME = '$SLURM_CLUSTER_NAME
echo '    SLURMD_NODENAME = '$SLURMD_NODENAME
echo '        SLURM_JOBID = '$SLURM_JOBID
echo '     SLURM_JOB_USER = '$SLURM_JOB_USER
echo '    SLURM_PARTITION = '$SLURM_JOB_PARTITION
echo '  SLURM_JOB_ACCOUNT = '$SLURM_JOB_ACCOUNT
echo '----------------------------------------------------'

#TeamCity Output
cat << EOF
##teamcity[buildNumber '$SLURM_JOBID']
EOF

echo "Load HPC modules"
module load anaconda

echo "Activate Environment"
source activate enhancer
export TRANSFORMERS_OFFLINE=True
export PYTHONPATH=${PYTHONPATH}:/scratch/c.sistc3/enhancer
export HYDRA_FULL_ERROR=1

echo $PYTHONPATH

source ~/mlflow_settings.sh

echo "Making temp dir"
mkdir temp
pwd

# echo "files"
# rm -rf  /scratch/c.sistc3/MS-SNSD/DNS30/CleanSpeech_training
# rm -rf /scratch/c.sistc3/MS-SNSD/DNS30/NoisySpeech_training
# rm -rf /scratch/c.sistc3/MS-SNSD/DNS30/NoisySpeech_testing
# rm -rf /scratch/c.sistc3/MS-SNSD/DNS30/CleanSpeech_testing

# mkdir /scratch/c.sistc3/MS-SNSD/DNS30
python noisyspeech_synthesizer.py

mv ./CleanSpeech_testing/ /scratch/c.sistc3/MS-SNSD/DNS30
mv ./NoisySpeech_testing/ /scratch/c.sistc3/MS-SNSD/DNS30
ls /scratch/c.sistc3/MS-SNSD/DNS30
#python enhancer/cli/train.py
