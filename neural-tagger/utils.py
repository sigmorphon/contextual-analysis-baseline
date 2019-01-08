from __future__ import division, print_function
from conllu import parse
from tags import Tags, Tag, Label

import os
import re
import math
import numpy as np
import itertools

import torch
from torch.autograd import Variable
import torch.nn.functional as F
np.set_printoptions(threshold=np.nan)


FROZEN_TAG = "__frozen__"

def freeze_dict(obj):
    if isinstance(obj, dict):
        dict_items = list(obj.items())
        dict_items.append((FROZEN_TAG, True))
        return tuple([(k, freeze_dict(v)) for k, v in dict_items])
    return obj

def unfreeze_dict(obj):
    if isinstance(obj, tuple):
        if (FROZEN_TAG, True) in obj:
            out = dict((k, unfreeze_dict(v)) for k, v in obj)
            del out[FROZEN_TAG]
            return out
    return obj


def read_unimorph(treebank_path, langs, train_or_dev, tgt_size=None, tgt_frac=1.0, test=False):

  """
   Reads conll formatted file

   langs: list of languages
   train: read training data
   returns: dict with data for each language
   as list of tuples of sentences and morph-tags
  """

  annot_sents = {}
  unique = []
  for lang in langs:

    dev_or_test = "train" if not test else train_or_dev

    if not test:
      for file in os.listdir(treebank_path):
        if file.endswith("train.conllu"):
          filepath = os.path.join(treebank_path, file)
          break
    else:
      for file in os.listdir(treebank_path):
        if file.endswith(dev_or_test + ".conllu") :
          filepath = os.path.join(treebank_path, file)
          break

    with open(filepath) as f:
      data = f.readlines()[:-1]
      data = [line for line in data if line[0]!='#']
      split_data = " ".join(data).split("\n \n")
      ud = [parse(sent)[0] for sent in split_data]

      all_text = []
      all_tags = []
     
      ud = ud[:(int)(len(ud)*tgt_frac)]
      if langs[-1]==lang and tgt_size:
        tgt_size = min(tgt_size, len(ud))
        ud = ud[:tgt_size]
      for sent in ud:
        sent_text = []
        sent_tags = []
        for word in sent:
          word_tags = []
          if word['feats']:
            word_tags = word['feats']
          if word['upostag']:
            if word_tags:
              word_tags += ";" + word['upostag']
            else:
              word_tags = word['upostag']

          if word_tags:
            #word_tags = freeze_dict(word_tags)
            if word_tags not in unique:
              unique.append(word_tags)

          sent_text.append(word['form'])
          sent_tags.append(word_tags)

        all_text.append(sent_text)
        all_tags.append(sent_tags)

      annot_sents[lang] = [(w, m) for w, m in zip(all_text, all_tags)]

  return annot_sents, unique


def jackknife(train_dataset, folds=10):
    dev_size = (int)(len(train_dataset)/folds)
    dev_datasets = []
    new_train_datasets = []
    for k in range(folds):
        start_idx = k * dev_size
        if k == folds-1:
             dev_datasets.append(train_dataset[start_idx:])
        else:
             dev_datasets.append(train_dataset[start_idx: start_idx + dev_size])
        new_train_datasets.append(train_dataset[0:start_idx] + train_dataset[start_idx + dev_size:])
    return new_train_datasets, dev_datasets
    

def write_unimorph(treebank_path, hyps, logProbs, sentCount, k, dev_or_test, frac=None, dev_size=None):
    filepath = None
    suffix = "train.conllu" if k!=-1 else dev_or_test+".conllu"
    for file in os.listdir(treebank_path):
      if file.endswith(suffix):
        filepath = os.path.join(treebank_path, file)
        break

    if filepath==None:
        print("No test set provided!")

    write_sents = []
    sent = 0
    flattened_hyps = [item for sublist in hyps for item in sublist]
    folds = 10
    
    with open(filepath) as f:
        all_data = f.readlines()
        if k!=-1:
            data = all_data[k*dev_size: k*dev_size + dev_size]
        if k == folds-1:
            data = all_data[k*dev_size:]
        i = 0
        for line in all_data:
            #if sent==sentCount:
            #   break
            if line.strip()=="":
               sent += 1
            if k!=-1 and sent < k*dev_size:
               continue
            if k!=-1 and k!=folds-1 and sent >= k*dev_size + dev_size:
               break
            
            if line[0]=='#' or line.strip()=="":
               write_sents.append(line)
            else:
               line = line.split("\t")
               hypString = flattened_hyps[i].split(";")
               line[5] = ";".join(hypString[:-1]) if hypString!="" else "_"
               #line[3] = hypString[-1]  # POS
               #line.append(",".join([str(logProb) for logProb in logProbs[i]])+"\n")
               write_sents.append("\t".join(line))
               i += 1


    directory = "jackknife_predictions" if k!=-1 else "baseline_predictions"

    if not os.path.exists(directory):
      os.makedirs(directory)

    if k!=-1:
      filename = filepath.split("/")[-1] + str(k) + ".baseline.pred"
    else:
      filename = filepath.split("/")[-1] + ".baseline.pred"

    with open(directory + "/" + filename,'w') as f:
        f.writelines(write_sents)

def addNullLabels(annot_sents, langs, unique_tags):

  seen_tagsets = []

  for lang in langs:
    i = 0
    for w, m in annot_sents[lang]:
      new_tags = []
      for tags in m:
        tag_dict = unfreeze_dict(tags)
        for tag in unique_tags:
          if tag.name not in tag_dict:
            tag_dict[tag.name] = "NULL"
        tag_dict = freeze_dict(tag_dict)
        new_tags.append(tag_dict)
        if tag_dict not in seen_tagsets:
            seen_tagsets.append(tag_dict)

      annot_sents[lang][i] = (w, new_tags)
      i += 1

  return annot_sents, seen_tagsets


def removeNullLabels(tagset):

  newDict = {}
  for t, v in unfreeze_dict(tagset).items():
      if v!='NULL':
          newDict[t] = v

  return freeze_dict(newDict)

def sortbylength(data, lang_ids, maxlen=500):
  """
  :param data: List of tuples of source sentences and morph tags
  :param lang_ids: List of lang IDs for each sentence
  :param maxlen: Maximum sentence length permitted
  :return: Sorted data and sorted langIDs
  """
  src = [elem[0] for elem in data]
  tgt = [elem[1] for elem in data]
  indexed_src = [(i,src[i]) for i in range(len(src))]
  sorted_indexed_src = sorted(indexed_src, key=lambda x: -len(x[1]))
  sorted_src = [item[1] for item in sorted_indexed_src if len(item[1])<maxlen]
  sort_order = [item[0] for item in sorted_indexed_src if len(item[1])<maxlen]
  sorted_tgt = [tgt[i] for i in sort_order]
  sorted_lang_ids = [lang_ids[i] for i in sort_order]
  sorted_data = [(src, tgt) for src, tgt in zip(sorted_src, sorted_tgt)]

  return sorted_data, sorted_lang_ids


def get_train_order(training_data, batch_size, startIdx=0):
  """
  :param data: List of tuples of source sentences and morph tags
  :return: start idxs of batches
  """

  lengths = [len(elem[0]) for elem in training_data]
  start_idxs = []
  end_idxs = []
  prev_length=-1
  batch_counter = 0

  for i, length in enumerate(lengths, start=startIdx):

    if length!=prev_length or batch_counter>batch_size:
      start_idxs.append(i)
      if prev_length!=-1:
        end_idxs.append(i-1)
      batch_counter = 1

    batch_counter += 1
    prev_length = length

  end_idxs.append(startIdx + len(lengths)-1)

  return [(s,e) for s,e in zip(start_idxs, end_idxs)]

def find_unique_tags(train_data_tags, null_label=False):

  unique_tags = Tags()

  for tags in train_data_tags:
    for tag, label in unfreeze_dict(tags).items():
      if not unique_tags.tagExists(tag):
        unique_tags.addTag(tag)

      curTag = unique_tags.getTagbyName(tag)

      if not curTag.labelExists(label):
        curTag.addLabel(label)

  # Add null labels to unseen tags in each tag set
  if null_label:
    for tag in unique_tags:
      tag.addLabel("NULL")

  return unique_tags

def get_var(x,  gpu=False, volatile=False):
  x = Variable(x, volatile=volatile)
  if gpu:
    x = x.cuda()
  return x

def prepare_sequence(seq, to_ix, gpu=False):
  if isinstance(to_ix, dict):
    idxs = [to_ix[w] if w in to_ix else to_ix["UNK"] for w in seq]
  elif isinstance(to_ix, list):
    idxs = [to_ix.index(w) if w in to_ix else to_ix.index("UNK") for w in seq]
  tensor = torch.LongTensor(idxs)
  return get_var(tensor, gpu)

def to_scalar(var):
  # returns a python float
  return var.view(-1).data.tolist()[0]

def argmax(vec):
  # return the argmax as a python int
  _, idx = torch.max(vec, 1)
  return to_scalar(idx)

def logSumExp(a, b):
  maxi = np.maximum(a, b)
  aexp = a - maxi
  bexp = b - maxi
  sumOfExp = np.exp(aexp) + np.exp(bexp)
  return maxi + np.log(sumOfExp)

def logSumExpTensor(vec):
  # vec -> 16, tag_size
  batch_size = vec.size()[0]
  vec = vec.view(batch_size, -1)
  max_score = torch.max(vec, 1)[0]
  max_score_broadcast = max_score.view(-1, 1).expand(-1, vec.size()[1])
  return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1))

def logSumExpTensors(a, b):

  maxi = torch.max(a, b)
  aexp = a - maxi
  bexp = b - maxi
  sumOfExp = torch.exp(aexp) + torch.exp(bexp)
  return maxi + torch.log(sumOfExp)

def logDot(a, b, redAxis=None):

  if redAxis==1:
    b = b.transpose()

  max_a = np.amax(a)
  max_b = np.amax(b)

  C = np.dot(np.exp(a - max_a), np.exp(b - max_b))
  np.log(C, out=C)
  # else:
  #   np.log(C + 1e-300, out=C)

  C += max_a + max_b

  return C.transpose() if redAxis==1 else C


def logMax(a, b, redAxis=None):

  if redAxis==1:
    b = b.transpose()

  max_a = np.amax(a)
  max_b = np.amax(b)

  C = np.max(np.exp(a[:, :, None]-max_a) * np.exp(b[None, :, :]-max_b), axis=1)

  # if np.isfinite(C).all():
  np.log(C, out=C)
  # else:
  #   np.log(C + 1e-300, out=C)

  C += max_a + max_b

  return C.transpose() if redAxis==1 else C

def logNormalize(a):

  denom = np.logaddexp.reduce(a, 1)
  return (a.transpose()- denom).transpose()

def logNormalizeTensor(a):

  denom = logSumExpTensor(a)
  if len(a.size())==2:
    denom = denom.view(-1, 1).expand(-1, a.size()[1])
  elif len(a.size())==3:
    denom = denom.view(a.size()[0], 1, 1).expand(-1, a.size()[1], a.size()[2])

  return (a-denom)

def getCorrectCount(golds, hyps):

  correct = 0

  for i, word_tags in enumerate(golds, start=0):
    allCorrect = True
    for k, v in word_tags.items():
      if k in hyps[i]:
        if v!=hyps[i][k]:
          allCorrect = False
          break

    if allCorrect==True:
      correct += 1

  return correct
