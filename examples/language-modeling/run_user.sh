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

python3 run_clm_ner.py \
  --model_name_or_path gpt2 \
  --train_file /scratch/jx880/capstone/transformers/examples/language-modeling/data/train_user_entities.txt \
  --validation_file /scratch/jx880/capstone/transformers/examples/language-modeling/data/dev_user_entities.txt \
  --output_dir /scratch/jx880/capstone/transformers/examples/language-modeling/results_large_768 \
  --do_train \
  --do_eval \
  --num_train_epochs 6 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4
