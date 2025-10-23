#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=32
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/m/mlisondo/logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/m/mlisondo/logs/slurm-%j.out
#SBATCH --mail-user=Mateo.lisondo@upr.edu
#SBATCH --mail-type=ALL

module load conda/Miniforge3-24.11.3-0
cd /pscratch/sd/m/mlisondo/projects/spanet-scm/SPANet
conda activate ./environment

python -m spanet.train_scm \
  -of options_files/full_hadronic_ttbar/dual_head.json \
  --gpus 1 \
  --log_dir ./spanet_output_scm \
  --name dual_head \
  --spanet_log_directory ./spanet_output/version_4

