#!/bin/bash
#SBATCH --job-name=math_ocr
#SBATCH --output=./logs/math_ocr%j.out
#SBATCH --error=./logs/math_ocr%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=dev
#SBATCH --export=ALL

srun build_webdataset.sh

echo "Done with job $SLURM_JOB_ID"
