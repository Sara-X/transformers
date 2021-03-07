# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
  --train_file /scratch/jx880/capstone/transformers/examples/token-classification/data/small_train.txt \
  --validation_file /scratch/jx880/capstone/transformers/examples/token-classification/data/small_dev.txt \
  --output_dir /tmp/test-ner \
  --do_train \
  --do_eval
