import torch
import numpy as np
import transformers
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import json
import random

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)

class TokenClassifier:
   def __init__(self, model_path, use_fast=True, from_tf=False):
       config = AutoConfig.from_pretrained(model_path)
      #  self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
       self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config, from_tf=from_tf)

   def predict(self, input_ids):
       # Evaluation is done using CPU here
       # add a dummy batch dimension
       input_ids = torch.tensor([input_ids])
       #print(input_ids.shape)

       with torch.no_grad():
           result = self.model(input_ids=input_ids)
           logits = result["logits"].detach().cpu().numpy()
           # reduce the last axis with class with maximum score
          #  token_predict = np.argmax(logits, axis=-1)
           return logits

def normalize(arr):
  return sklearn.preprocessing.normalize(np.array(arr).reshape(1,-1), norm='l2').tolist()[0]

def auc(y, pred):
# x, y: np array
  fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
  auc = metrics.auc(fpr, tpr)
  return auc, thresholds

def get_preds_labels(scr_name, impression_scores):
  preds = []
  labels = []
  for usertime in impression_scores:
    user_scores = impression_scores[usertime]['normalized_loss']
    for idx in range(len(user_scores)):
      preds.append(user_scores[idx])
      labels.append(impression_scores[usertime]['labels'][idx])
  return preds, labels

def get_auc(scr_name, impression_scores):
  y, pred = get_preds_labels(scr_name, impression_scores)
  return auc(y,pred)


def main():
  with open("/scratch/jx880/capstone/transformers/examples/language-modeling/score_impr/news_entities_mapped.json", "r") as f:
    news_entities = json.load(f)
  
  with open("/scratch/jx880/capstone/transformers/examples/language-modeling/score_impr/small_user_history_news.json", "r") as f:
    user_news_history = json.load(f)
   
  headers = ['Impression ID', 'User ID','Time','History','Impression']
  df_behav_train = pd.read_csv("/content/drive/MyDrive/NewsRecommendation/MINDsmall_train/behaviors.tsv",sep='\t', names = headers)
  
  headers = ['Impression ID', 'User ID','Time','History','Impression']
  df_behav_dev = pd.read_csv("/content/drive/MyDrive/NewsRecommendation/MINDsmall_dev/behaviors.tsv",sep='\t', names = headers)


  f = TokenClassifier("/scratch/jx880/capstone/transformers/examples/language-modeling/results_small4")

  impression_scores = {}
  for idx, row in df_behav_train.iterrows():
    print(idx)
    user = row['User ID']
    time = row['Time']
    user_time = user + " " + time
    if user_time in impression_scores:
      continue
    # print(user)
    if user not in user_news_history:
      continue
    history = user_news_history[user]
    if len(history) < 50:
      continue
    impression = row['Impression'].split(" ")

    history_size = 50
    hist_ents = [news_entities[news] for news in history[len(history) - history_size:]]
    hist = " 0 ".join([" ".join(news) for news in hist_ents])
    # print(user, hist_ents)

    scores = []   # entities avg
    avg_in = []
    avg_out = []
    max_in = []
    max_out = []
    min_in = []
    min_out = []
    labels = []
    for news in impression:
      # print(len(ents))
      newsid = news.split("-")[0]
      news_label = news.split("-")[1]
      labels.append(news_label)
      news_ents = news_entities[newsid]
      # if len(news_ents) == 0:
      #   scores.append(20)
      #   # print(news,"here")
      #   continue
      # ents.append(news_entities[newsid])

      # print(news_ents)

      ent_scores = []  # entity avg
      in_ent_scores = []
      out_ent_scores = []

      for ent in news_ents:
        hist_list = hist.split(" ")
        lm_labels = torch.tensor([-100 for i in range(len(hist_list))] + [int(ent)])
        input_ids = [int(id) for id in hist_list] + [0]
        # print("lm_labels",len(lm_labels),lm_labels)
        # print("input_ids",len(input_ids),input_ids)

        logits = torch.tensor(f.predict(input_ids)[0])
        loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        output = float(loss(logits, lm_labels))

        ent_scores.append(output)

        if ent in hist_list:
          in_ent_scores.append(output)
        else:
          out_ent_scores.append(output)

      if len(ent_scores) == 0:
        scores.append(20)
      else:
        scores.append(sum(ent_scores)/len(ent_scores))
      if len(in_ent_scores) == 0:
        # print(user,time,"here")
        avg_in.append(20)
        max_in.append(20)
        min_in.append(20)
      else:
        avg_in.append(sum(in_ent_scores)/len(in_ent_scores))
        max_in.append(max(in_ent_scores))
        min_in.append(min(in_ent_scores))
      if len(out_ent_scores) == 0:
        avg_out.append(20)
        max_out.append(20)
        min_out.append(20)
      else:
        avg_out.append(sum(out_ent_scores)/len(out_ent_scores))
        max_out.append(max(out_ent_scores))
        min_out.append(min(out_ent_scores))
      # print(news,score)

    impression_scores[user_time] = {}
    impression_scores[user_time]['loss'] = scores

    impression_scores[user_time]['normalized_loss'] = normalize(scores)
    # print(user_time,avg_in)
    impression_scores[user_time]['avg_in'] = normalize(avg_in)
    impression_scores[user_time]['avg_out'] = normalize(avg_out)
    impression_scores[user_time]['max_out'] = normalize(max_out)
    impression_scores[user_time]['max_in'] = normalize(max_in)
    impression_scores[user_time]['min_out'] = normalize(min_out)
    impression_scores[user_time]['min_in'] = normalize(min_in)
    impression_scores[user_time]['label'] = labels
  with open("/scratch/jx880/capstone/transformers/examples/language-modeling/score_impr/small_train_impression_scores.json", "w") as f:
    json.dump(impression_scores, f)
      
if __name__ == "__main__":
    main()
      
