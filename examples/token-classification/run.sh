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

python3 run_ner.py \
  --model_name_or_path bert-base-cased \
  --train_file /scratch/jx880/capstone/transformers/examples/token-classification/data/large_train.txt \
  --validation_file /scratch/jx880/capstone/transformers/examples/token-classification/data/large_dev.txt \
  --test_file /scratch/jx880/capstone/transformers/examples/token-classification/data/large_test.txt \
  --output_dir /scratch/jx880/capstone/transformers/examples/token-classification/ner_results_large_pred3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --do_train \
  --do_eval \
  --do_predict
