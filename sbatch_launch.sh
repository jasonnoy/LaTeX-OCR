#!/bin/bash
#SBATCH --job-name=build_math
#SBATCH --output=./logs/build_math%j.out
#SBATCH --error=./logs/build_math%j.err
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --exclude=g0001
#SBATCH --partition=dev
#SBATCH --export=ALL

srun build_webdataset.sh

echo "Done with job $SLURM_JOB_ID"
