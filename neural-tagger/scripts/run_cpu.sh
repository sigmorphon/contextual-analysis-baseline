#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4400MB
#SBATCH -t 2-00:00:00 

set -x  # echo commands to stdout
set -e  # exit on error

module load cuda
