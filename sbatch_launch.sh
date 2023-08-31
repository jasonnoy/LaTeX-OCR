#!/bin/bash
#SBATCH --job-name=extract_math
#SBATCH --output=./logs/extract_math%j.out
#SBATCH --error=./logs/extract_math%j.err
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=3
#SBATCH --exclude=g0001
#SBATCH --partition=dev
#SBATCH --export=ALL

srun extract_tex.sh

echo "Done with job $SLURM_JOB_ID"
