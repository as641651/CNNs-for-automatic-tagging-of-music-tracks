# coding=utf8

#import _init_paths
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


def build_vocab(dataDict,bYear,bArtist,bOthers):
   vocab = {}
   info_vocab = {}
   for k,v in dataDict.iteritems():
       if k == 'song_id_to_name':
          continue
       v["info_tokens"] = []
        
   for k,v in dataDict.iteritems():
       if k == 'song_id_to_name':
          continue
       for k2,v2 in v.iteritems():
          if k2 != 'song_id' and k2 != 'info_tags' and k2 != 'info_tokens':
             tokens = []
             for vo in v2["labels"]:
                tokens.append(word_preprocess(vo))
             v2["tokens"] = tokens
             for t in tokens:
               if not t in vocab:
                 vocab[t] = 1
               else:
                 vocab[t] = vocab[t] + 1 
       
          elif k2 == 'info_tags':
             info_tokens = []
             if bYear:
               if v["info_tags"]["year"] != -1:
                  info_tokens.append(word_preprocess(v["info_tags"]["year"]))
             if bArtist:
               for a in v["info_tags"]["artist"]:
                  info_tokens.append(word_preprocess(a))
             if bOthers:
               for o in v["info_tags"]["additional_tags"]:
                  info_tokens.append(word_preprocess(o))
        
             v["info_tokens"] = info_tokens
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
   clips_song = {}
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
          if k2 != 'song_id' and k2 != 'info_tags' and k2 != 'info_tokens':
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

          elif k2 == 'info_tags':
             info_tags_list = []
             for t in v["info_tokens"]:
                if info_vocab[t] >= info_vocab_minFreq:
                  if not t in info_token_to_idx:
                     info_token_to_idx[t] = info_idd
                     info_idx_to_token[info_idd] = t
                     info_idd = info_idd + 1
                  info_tags_list.append(info_token_to_idx[t]) 
               
       if clips:
          song_clips[v['song_id']] = clips
          info_tags[v['song_id']] = info_tags_list
          num_songs = num_songs + 1
          for c in clips:
             clips_song[c] = v['song_id']

   assert(num_songs == len(song_clips)) #just a correctness assurance check :P
   return gt,info_tags,num_clips,token_to_idx,idx_to_token,info_token_to_idx,info_idx_to_token,song_clips,num_songs,clips_song

def random_split(dataDict,exp):
   
   vocab,info_vocab = build_vocab(dataDict,exp["use_year"],exp["use_artist"],exp["use_other_tags"])
   exp["gt"],exp["info_tags"],num_clips,exp["token_to_idx"],exp["idx_to_token"],exp["info_token_to_idx"],exp["info_idx_to_token"],exp["song_clips"],num_songs,exp["clips_song"] = encodeGroundTruth(dataDict,vocab,info_vocab,exp["label_min_freq"],exp["info_tag_min_freq"])

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

                       
def write_comm(split_path,db_path):
   comm = {}
   comm["split_info_path"] = os.path.abspath(split_path)
   comm["h5_file"] = os.path.abspath(db_path)
   with open('../cache/tmp.json', 'w') as f:
      json.dump(comm, f)

def main(args):

   with open(args.config_file, "r") as f:
      split_config = json.load(f)
   
   json_path = os.path.join(split_config["db_dir"],split_config["data_json"])
   if not os.path.exists(json_path):
      print FAIL + "Path does not exist :" + ENDC + json_path
      exit(-1)

   h5_path = os.path.join(split_config["db_dir"],split_config["data_h5"])
   if not os.path.exists(h5_path):
      print FAIL + "Path does not exist :" + ENDC + h5_path
      exit(-1)

   split_info_file = split_config["data_h5"]+str(split_config["train_percent"])+str(split_config["val_percent"])+str(split_config["test_percent"])+str(split_config["min_label_freq"])+str(split_config["min_info_tag_freq"])+str(split_config["group"])+str(split_config["use_year"])+str(split_config["use_artist"])+str(split_config["use_other_tags"])

   split_info_file = split_info_file + ".json"
   split_info_path = os.path.join("../cache",split_info_file)

   if(os.path.exists(split_info_path)):
      print "Loading " + split_info_path + " from cache..."
      with open(split_info_path, "r") as f:
         split_info = json.load(f)
         print "DB Name: " + split_config["data_h5"]
         print "Vocab in use .."
         print split_info["idx_to_token"]
         print "TOTAL : " + str(len(split_info["idx_to_token"]))
         print "info_vocab in use .. "
         print split_info["info_idx_to_token"]
         write_comm(split_info_path, h5_path)
      return

 
   with open(json_path, "r") as f:
      dataDict = json.load(f)

   experiment = {}
   experiment["gt"] = {}
   experiment["song_clips"] = {}
   experiment["clips_song"] = {}
   experiment["group"] = bool(split_config["group"])
   experiment["use_year"] = bool(split_config["use_year"])
   experiment["use_artist"] = bool(split_config["use_artist"])
   experiment["use_other_tags"] = bool(split_config["use_other_tags"])
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
   
   with open(split_info_path, 'w') as f:
      json.dump(experiment, f)

   write_comm(split_info_path, h5_path)
  # print json.dumps(blob,sort_keys=True,indent=4)
   print WARNING + "Wrote output : " + ENDC + split_info_path


def print_help():
   print FAIL + "Requires a json file with config data: " + ENDC
   print "{"
   print "  \"data_json\": //THE DATABASE JSON FILE , "
   print "  \"data_h5\": //NAME OF HDF5 FILE ,"
   print "  \"db_dir\": //NAME OF Dir ,"
   print "  \"train_percent\": //PERCENT SAMPLES FOR TRAIN SET ,"
   print "  \"val_percent\": //PERCENT SAMPLES FOR VAL SET ,"
   print "  \"test_percent\": //PERCENT SAMPLES FOR TEST SET ,"
   print "  \"min_label_freq\": //THE GT LABELS HAVE TO APPEAR MORE THAN THIS VALUE TO BE CONSIDERED FOR TRAINING ,"
   print "  \"min_info_tag_freq\": //THE INFO LABELS HAVE TO APPEAR MORE THAN THIS VALUE TO BE CONSIDERED FOR TRAINING ,"
   print "  \"group\": //True/False  GROUPS CLIPS FROM SAME SONG "
   print "  \"use_year\": //True/False  INFO_TAGS  "
   print "  \"use_artist\": //True/False  INFO_TAGS  "
   print "  \"use_other_tags\": //True/False  INFO_TAGS like album or something else  "
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
