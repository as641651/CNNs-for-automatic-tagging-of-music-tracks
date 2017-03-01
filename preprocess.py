# coding=utf8
import _init_paths
import argparse, os, json
import numpy as np
import h5py

from parsers.factory import get_parser
from parsers.utils import addData
from audio_processor.factory import get_audio_processor

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'


def main(settings):

  # read in the data
  reader = get_parser(settings["parser"])
  dataDict = reader.parse_input(settings)
  num_clips = dataDict["Total_tracks"]

  if os.path.exists(settings["h5_output"]):
     os.remove(settings["h5_output"])
     print WARNING + "Removed old " + settings["h5_output"] + ENDC
  
  fh5 = h5py.File(settings["h5_output"], 'w')
 
  json_file = settings["h5_output"].split(".")[0] + "_processed.json"
  shape = (dataDict["Total_Songs"]+1,80,1)

  audio_processor = get_audio_processor(settings["audio_processor"])
   
  json_struct = {}
  song_id_to_name = {}
  clips_processed = 0
  skipped = 0
  songs_processed = 0
  for k,v in dataDict.iteritems():
     if k == 'Total_Songs' or k == 'Total_tracks' or k == 'song_id_to_name':
        continue
     j=0
     for k2,v2 in v.iteritems():
         if k2 == 'song_id' or k2 == 'info_tags':
            continue
         else:
            input_path = os.path.join(settings["input_dir"],v2['path'])
            if not os.path.exists(input_path):
               print WARNING + "WARNING: Skipping track as path does not exist : %s"%input_path
               skipped = skipped + 1
               continue
            try:
               input_data = audio_processor.process_input(input_path,settings)
            except EOFError:
               print WARNING + "EOF error in " + input_path + ENDC
               skipped = skipped + 1
               continue

            L = input_data.shape[2]
            fh5.create_dataset(str(k2),data=input_data)
            j = j+1
            clips_processed = clips_processed + 1 
            ## we reconstruct the dict created by parser to exclude the samples that fail audio processing
            addData(json_struct,k,v["song_id"],k2,v2["path"],v2["labels"],v["info_tags"]["year"],v["info_tags"]["artist"],v["info_tags"]["additional_tags"])
            print OKGREEN + "[(%d + %d )/%d]"%(clips_processed,skipped,num_clips) + ENDC + " Processed:\n" + WARNING + "clip_id: " + ENDC +"%d"%int(k2) + WARNING + "\npath: " + ENDC + "%s"%(v2['path'])

     
     if "max_samples" in settings:
        if settings["max_samples"] > 0 and clips_processed > settings["max_samples"]:
           break

     songs_processed = songs_processed + 1
     print "Songs processed :" ,  songs_processed
     if(songs_processed%20 == 0):
        print OKGREEN + "dumping output to " + json_file + ENDC
        json_struct["song_id_to_name"] = song_id_to_name
        with open(json_file, 'w') as f:
           json.dump(json_struct, f)
     
  fh5.close()
    
  with open(json_file, 'w') as f:
    json.dump(json_struct, f)


def print_help():
   print FAIL + "Requires a json file with preprocess setting: " + ENDC
   print "{"
   print "  \"data\": //FILES REQUIRED BY THE PARSER, "
   print "  \"input_dir\": //ROOT FOLDER CONTAINING THE INPUT FILES,"
   print "  \"parser\": //CURRENTLY AVAILABLE : [magna_parser],"
   print "  \"audio_processor\": //CURRENTLY AVAILABLE : [melgram],"
   print "  \"h5_output\": //PATH OF OUTPUT DATABASE"
   print "  \"max_samples\": //MAX NUMBER OF FILES TO BE PROCESSED"
   print "}"


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--cfg',
      default='',
      help='Path to settings json file')

  # OPTIONS
  args = parser.parse_args()

  if not args.cfg: 
     print_help()
     exit(-1)  


  with open(args.cfg, "r") as f:
     settings = json.load(f)
  
  #SANITY CHECKS
  if not "parser" in settings:
     print FAIL + "Please specifiy a parser : [ magna_parser ]" + ENDC
     exit(-1)
  if not "data" in settings:
     print FAIL + "Please specifiy datafile" + ENDC
     exit(-1)
  if not "input_dir" in settings:
     print FAIL + "Please specifiy the directory containing input files" + ENDC
     exit(-1)
  if not "audio_processor" in settings:
     print FAIL + "Please specifiy the audio processor" + ENDC
     exit(-1)
  if not "h5_output" in settings:
     print FAIL + "Please specifiy the path to output file" + ENDC
     exit(-1)

  settings["data"] = str(settings["data"])
  settings["input_dir"] = str(settings["input_dir"])
  settings["audio_processor"] = str(settings["audio_processor"])
  settings["parser"] = str(settings["parser"])
  settings["h5_output"] = str(settings["h5_output"])

  print WARNING + "ARGUMENTS: " + ENDC
  print WARNING + "data :" + ENDC + settings["data"]
  print WARNING + "input_dir : " + ENDC + settings["input_dir"]
  print WARNING + "audio_processor: " + ENDC + settings["audio_processor"]
  print WARNING + "parser :" + ENDC + settings["parser"]
  print WARNING + "h5_output :" + ENDC + settings["h5_output"] + " , " + settings["h5_output"].split(".")[0] + "_processed.json"

  main(settings)

