#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=03:00:00
#SBATCH --output=/pscratch/sd/m/mlisondo/logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/m/mlisondo/logs/slurm-%j.out
#SBATCH --mail-user=Mateo.lisondo@upr.edu
#SBATCH --mail-type=ALL

module load conda/Miniforge3-24.11.3-0
cd /pscratch/sd/m/mlisondo/projects/spanet-scm/SPANet
conda activate ./environment

python -m spanet.test_scm -g -k 3 -o spanet_output_scm/dual_head/version_0 --log_directory spanet_output_scm/dual_head/version_0 -tf data/full_hadronic_ttbar/testing.h5

