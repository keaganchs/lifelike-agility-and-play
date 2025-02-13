#!/bin/bash

#SBATCH --job-name=lifelike_agility_and_play
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200
#SBATCH --time=23:59:59

#SBATCH --gres=gpu:1

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate lifelike

export CONTROL_FREQ=50
export PORT_OFFSET=5
export USE_TORQUE_ACTIONS=False
export GAMMA=0.95

# Launch all tasks concurrently using multi-program mode.
srun --multi-prog pmc.conf
