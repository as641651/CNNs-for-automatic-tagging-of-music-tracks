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


def main(args):

  # read in the data
  reader = get_parser(args.parser)
  dataDict = reader.parse_input(args)
  num_clips = dataDict["Total_tracks"]  
  fh5 = h5py.File(args.h5_output, 'w')
 
  json_file = args.h5_output.split(".")[0] + "_processed.json"
  shape = (dataDict["Total_Songs"]+1,80,1)

  audio_processor = get_audio_processor(args.ap)
   
  json_struct = {}
  song_id_to_name = {}
  clips_processed = 0
  skipped = 0
  songs_processed = 0
  for k,v in dataDict.iteritems():
     if k == 'Total_Songs' or k == 'Total_tracks':
        continue
     j=0
     for k2,v2 in v.iteritems():
         if k2 == 'song_id':
            song_id_to_name[v["song_id"]] = k
         else:
            input_path = os.path.join(args.input_dir,v2['path'])
            if not os.path.exists(input_path):
               print WARNING + "WARNING: Skipping track as path does not exist : %s"%input_path
               skipped = skipped + 1
               continue
            try:
               input_data = audio_processor.process_input(input_path,args)
            except EOFError:
               print WARNING + "EOF error in " + input_path + ENDC
               skipped = skipped + 1
               continue

            L = input_data.shape[2]
            fh5.create_dataset(k2,data=input_data)
            j = j+1
            clips_processed = clips_processed + 1
            addData(json_struct,k,v["song_id"],k2,v2["path"],v2["labels"])
            print OKGREEN + "[(%d + %d )/%d]"%(clips_processed,skipped,num_clips) + ENDC + " Processed:\n" + WARNING + "clip_id: " + ENDC +"%d"%int(k2) + WARNING + "\npath: " + ENDC + "%s"%(v2['path'])

     
     if( args.debug and clips_processed > 20):
        break

     songs_processed = songs_processed + 1
     print "Songs processed :" ,  songs_processed
     if(songs_processed%5 == 0):
        print OKGREEN + "dumping output to " + json_file + ENDC
        json_struct["song_id_to_name"] = song_id_to_name
        with open(json_file, 'w') as f:
           json.dump(json_struct, f)
     
  fh5.close()
    
  json_struct["song_id_to_name"] = song_id_to_name
  with open(json_file, 'w') as f:
    json.dump(json_struct, f)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # INPUT settings
  parser.add_argument('--data',
      default='',
      help='Input CSV file')
  parser.add_argument('--parser',
      default='',
      help='parsers available : [ magna_parser]')
  parser.add_argument('--input_dir',
      default='',
      help='Directory containing all tracks')
  parser.add_argument('--ap',
      default='melgram',
      help='audio processers available: [melgram]')
  parser.add_argument('--debug',
      action = "store_true",
      default=False,
      help='debug set')

  # OUTPUT settings
  parser.add_argument('--h5_output',
      default='database.h5',
      help='Path to output HDF5 file')

  # OPTIONS
  args = parser.parse_args()
  
  #SANITY CHECKS
  if not args.parser:
     print FAIL + "Please specifiy a parser : [ magna_parser ]" + ENDC
  if not args.data:
     print FAIL + "Please specifiy datafile" + ENDC
  if not args.input_dir:
     print FAIL + "Please specifiy the directory containing input files" + ENDC

  print WARNING + "ARGUMENTS: " + ENDC
  print WARNING + "--data " + ENDC + args.data
  print WARNING + "--songs_dir " + ENDC + args.input_dir
  print WARNING + "--ap " + ENDC + args.ap
  print WARNING + "--parser " + ENDC + args.parser
  print WARNING + "--h5_output " + ENDC + args.h5_output + " , " + args.h5_output.split(".")[0] + "_additional.json"

  print args
  main(args)

