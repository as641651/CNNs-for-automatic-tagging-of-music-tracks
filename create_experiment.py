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
   info_vocab = {}
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

             info_tokens = []
             for v in v2["labels"]: ##TODO: info labels
                info_tokens.append(word_preprocess(v))
                v2["info_tokens"] = info_tokens
             for t in info_tokens:
               if not t in info_vocab:
                 info_vocab[t] = 1
               else:
                 info_vocab[t] = info_vocab[t] + 1 

   return vocab,info_vocab   

def encodeGroundTruth(dataDict,vocab,info_vocab,minFreq,info_vocab_minFreq):

   gt = {}
   info_tags = {}
   song_clips = {}
   token_to_idx = {}
   idx_to_token = {}
   idd = 0
   info_token_to_idx = {}
   info_idx_to_token = {}
   info_idd = 0
   num_clips = 0
   num_songs = 0
   for k,v in dataDict.iteritems():
       if k == 'song_id_to_name':
          continue

       clips = []
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
                clips.append(int(k2))
                num_clips = num_clips + 1

                info_tags_list = []
                for t in v2["info_tokens"]:
                  if info_vocab[t] >= info_vocab_minFreq:
                    if not t in info_token_to_idx:
                       info_token_to_idx[t] = info_idd
                       info_idx_to_token[info_idd] = t
                       info_idd = info_idd + 1
                    info_tags_list.append(info_token_to_idx[t]) 
                  info_tags[k2] = info_tags_list
               
       if clips:
          song_clips[dataDict[k]['song_id']] = clips
          num_songs = num_songs + 1

   return gt,info_tags,num_clips,token_to_idx,idx_to_token,info_token_to_idx,info_idx_to_token,song_clips,num_songs

def random_split(dataDict,exp):
   
   vocab,info_vocab = build_vocab(dataDict)
   exp["gt"],exp["info_tags"],num_clips,exp["token_to_idx"],exp["idx_to_token"],exp["info_token_to_idx"],exp["info_idx_to_token"],exp["song_clips"],num_songs = encodeGroundTruth(dataDict,vocab,info_vocab,exp["label_min_freq"],exp["info_tag_min_freq"])

   exp["vocab_size"] = len(exp["idx_to_token"])
   exp["info_vocab_size"] = len(exp["info_idx_to_token"])

   print("Vocab in use : Total(%d)"%len(exp["idx_to_token"]))
   print(exp["idx_to_token"])

   print("info Vocab in use : Total(%d)"%len(exp["info_idx_to_token"]))
   print(exp["info_idx_to_token"])
   
   num_samples = num_clips
   sample_idx = exp["gt"]
   if exp["group"]:
      num_samples = num_songs
      sample_idx = exp["song_clips"]
   
   train = int(math.ceil(exp["train"]*num_samples))
   val = int(math.floor(exp["val"]*num_samples))
   test = int(math.floor(exp["test"]*num_samples))

   diff = train + val + test - num_samples
   test = test - diff
   
   print "Total Songs :" + str(num_songs) 
   print "Total clips :" + str(num_clips) 
   print "Group by songs :" + str(exp["group"]) 
   print OKGREEN + "Split : " + str(train) + "/" + str(val) + "/" + str(test) + ENDC


   it = 0
   j=0

   for k,v in sample_idx.iteritems():
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

   with open(args.config_file, "r") as f:
      split_config = json.load(f)
 
   with open(split_config["data_json"], "r") as f:
      dataDict = json.load(f)

   experiment = {}
   experiment["gt"] = {}
   experiment["song_clips"] = {}
   experiment["group"] = bool(split_config["group"])
   experiment["label_min_freq"] = int(split_config["min_label_freq"])
   experiment["info_tag_min_freq"] = int(split_config["min_info_tag_freq"])
   experiment["train"] = float(split_config["train_percent"])
   experiment["val"] = float(split_config["val_percent"])
   experiment["test"] = float(split_config["test_percent"])
   experiment["train_idxs"] = []
   experiment["val_idxs"] = []
   experiment["test_idxs"] = []
   experiment["token_to_idx"] = {}
   experiment["idx_to_token"] = {}
   experiment["vocab_size"] = 0
   experiment["info_vocab_size"] = 0

   random_split(dataDict,experiment)
 
   with open(str(split_config["split_info_path"]), 'w') as f:
      json.dump(experiment, f)

   print WARNING + "Wrote output : " + ENDC + split_config["split_info_path"]


def print_help():
   print FAIL + "Requires a json file with config data: " + ENDC
   print "{"
   print "  \"data_json\": //THE DATABASE JSON FILE , "
   print "  \"split_info_path\": //PATH OF THE OUTPUT SPLIT_INFO JSON FILE ,"
   print "  \"train_percent\": //PERCENT SAMPLES FOR TRAIN SET ,"
   print "  \"val_percent\": //PERCENT SAMPLES FOR VAL SET ,"
   print "  \"test_percent\": //PERCENT SAMPLES FOR TEST SET ,"
   print "  \"min_label_freq\": //THE GT LABELS HAVE TO APPEAR MORE THAN THIS VALUE TO BE CONSIDERED FOR TRAINING ,"
   print "  \"min_info_tag_freq\": //THE INFO LABELS HAVE TO APPEAR MORE THAN THIS VALUE TO BE CONSIDERED FOR TRAINING ,"
   print "  \"group\": //True/False  GROUPS CLIPS FROM SAME SONG "
   print "}"

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # OUTPUT settings
  parser.add_argument('-c','--config_file',
      default='',
      help='Path to config file')

  # OPTIONS
  args = parser.parse_args()

  if not args.config_file: 
     print_help()
     exit(-1)  

  main(args)
