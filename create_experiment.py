# coding=utf8

import _init_paths
import argparse, os, json
import numpy as np
import math
from parsers.factory import get_parser
import string

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

def word_preprocess(phrase):
  """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
  replacements = {
    u'½': u'half',
    u'—' : u'-',
    u'™': u'',
    u'¢': u'cent',
    u'ç': u'c',
    u'û': u'u',
    u'é': u'e',
    u'°': u' degree',
    u'è': u'e',
    u'…': u'',
  }
  for k, v in replacements.iteritems():
    phrase = phrase.replace(k, v)
  return str(phrase).lower().translate(None, string.punctuation)


def build_vocab(dataDict):
   vocab = {}
   for k,v in dataDict.iteritems():
       if k == 'song_id_to_name':
          continue
       for k2,v2 in v.iteritems():
          if k2 != 'song_id':
             tokens = []
             for v in v2["labels"]:
                tokens.append(word_preprocess(v))
             v2["tokens"] = tokens
             for t in tokens:
               if not t in vocab:
                 vocab[t] = 1
               else:
                 vocab[t] = vocab[t] + 1 

   return vocab   

def encodeGroundTruth(dataDict,vocab,minFreq):

   gt = {}
   token_to_idx = {}
   idx_to_token = {}
   idd = 0
   num_samples = 0
   for k,v in dataDict.iteritems():
       if k == 'song_id_to_name':
          continue

       for k2,v2 in v.iteritems():
          if k2 != 'song_id':
             tags = []
             for t in v2["tokens"]:
                if vocab[t] >= minFreq:
                  if not t in token_to_idx:
                     token_to_idx[t] = idd
                     idx_to_token[idd] = t
                     idd = idd + 1
                  tags.append(token_to_idx[t]) 
             if tags:
                gt[k2] = tags
                num_samples = num_samples + 1
   return gt,num_samples,token_to_idx,idx_to_token

def random_split(dataDict,exp):
   
   vocab = build_vocab(dataDict)
   exp["gt"],num_clips,exp["token_to_idx"],exp["idx_to_token"] = encodeGroundTruth(dataDict,vocab,exp["label_min_freq"])

   print("Vocab in use :")
   print(exp["idx_to_token"])
   
   train = int(math.ceil(exp["train"]*num_clips))
   val = int(math.floor(exp["val"]*num_clips))
   test = int(math.floor(exp["test"]*num_clips))

   diff = train + val + test - num_clips
   test = test - diff
   
   print "Total clips :" + str(num_clips) 
   print OKGREEN + "Split : " + str(train) + "/" + str(val) + "/" + str(test) + ENDC


   it = 0
   j=0

   for k,v in exp["gt"].iteritems():
      if it == 0:
         exp["train_idxs"].append(int(k))
      if it == 1:
         exp["val_idxs"].append(int(k))
      if it == 2:
         exp["test_idxs"].append(int(k))
      j=j+1
      if j == train:
         it = 1
      if j == train+val:
         it = 2
                       

def main(args):

   json_file = "/home/as641651/user/Thesis/CNNs-for-automatic-tagging-of-music-tracks/databases/dd_new_additional.json"
  
   with open(json_file) as f:
      dataDict = json.load(f)

   experiment = {}
   experiment["gt"] = {}
   experiment["group"] = False
   experiment["max_group"] = 5
   experiment["label_min_freq"] = 2
   experiment["train"] = 0.7
   experiment["val"] = 0.3
   experiment["test"] = 0.
   experiment["train_idxs"] = []
   experiment["val_idxs"] = []
   experiment["test_idxs"] = []
   experiment["token_to_idx"] = {}
   experiment["idx_to_token"] = {}

   random_split(dataDict,experiment)

   with open(args.json_file, 'w') as f:
      json.dump(experiment, f)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # OUTPUT settings
  parser.add_argument('--json_file',
      default='experiment.json',
      help='Path to output HDF5 file')

  # OPTIONS
  args = parser.parse_args()
  

  main(args)
