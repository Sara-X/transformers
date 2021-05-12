#!/usr/bin/bash
#
#SBATCH --job-name=myTrainTask2GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=log_train_%A_%a.out
#SBATCH --error=log_train_%A_%a.err
#SBATCH --time=24:00:00
source $HOME/.bashrc

python3 impression_score.py
