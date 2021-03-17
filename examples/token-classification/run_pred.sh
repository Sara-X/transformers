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
  --model_name_or_path bert-base-uncased \
  --train_file /scratch/jx880/capstone/transformers/examples/token-classification/data/large_train.txt \
  --validation_file /scratch/jx880/capstone/transformers/examples/token-classification/data/large_dev.txt \
  --output_dir /scratch/jx880/capstone/transformers/examples/token-classification/ner_results_large \
  --do_predict
