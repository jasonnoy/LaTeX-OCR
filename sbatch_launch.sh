#!/bin/bash
#SBATCH --job-name=math_ocr
#SBATCH --output=./logs/math_ocr%j.out
#SBATCH --error=./logs/math_ocr%j.err
#SBATCH --gres=gpu:0
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=55
#SBATCH --exclude=g0001
#SBATCH --partition=dev
#SBATCH --export=ALL

srun parallel_latex.sh

echo "Done with job $SLURM_JOB_ID"
