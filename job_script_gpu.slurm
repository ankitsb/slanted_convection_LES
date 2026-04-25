#!/bin/bash
#SBATCH --job-name=SC_v3
## SBATCH --nodes=1
#SBATCH --partition=gpu-a100-preempt ##-short  ##
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --account=punim1661
## Use an account that has GPGPU access

module purge
module load CUDA/12.5.1

export JULIA_DEPOT_PATH=/data/gpfs/projects/punim1661/.julia

julia code.jl

##Job monitor command to list the resource usage
my-job-stats -a -n -s

