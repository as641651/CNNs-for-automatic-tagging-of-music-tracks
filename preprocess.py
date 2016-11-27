import _init_paths
import argparse, os, json
import numpy as np
import h5py

from parsers.factory import get_parser
from audio_processor.factory import get_audio_processor

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

def main(args):

  # read in the data
  reader = get_parser(args.parser)
  dataDict = reader.parse_input(args.data)
  num_clips = dataDict["Total_tracks"]+1
  
  fh5 = h5py.File(args.h5_output, 'w')
 
  shape = (dataDict["Total_Songs"]+1,80,1)
  song_clip = np.empty(shape,dtype=np.int32)
  song_clip.fill(-1)

  audio_processor = get_audio_processor(args.ap)
  shape_input = (num_clips,audio_processor.num_channels, audio_processor.feature_length, int(audio_processor.num_samples))
  print shape_input
  clip_input = fh5.create_dataset('inputs',shape_input,dtype=np.int32)
   
  json_struct = {}
  clip_idx_map = {}
  clips_processed = 0
  for k,v in dataDict.iteritems():
     song_struct = {}
     if k == 'Total_Songs' or k == 'Total_tracks':
        continue
     j=0
     for k2,v2 in v.iteritems():
         if k2 != 'song_id':
            input_path = os.path.join(args.input_dir,v2['path'])
            if not os.path.exists(input_path):
               print WARNING + "WARNING: Skipping track as path does not exist : %s"%input_path
               continue
            clip_idx_map[k2] = clips_processed
            song_struct[k2] = v2['path']
            input_data = audio_processor.process_input(input_path,args)
            L = input_data.shape[2]
            clip_input[clips_processed,:,:,:L] = input_data
            song_clip[int(v['song_id'])][j] = int(k2)
            j = j+1
            clips_processed = clips_processed + 1
            print OKGREEN + "[%d/%d]"%(clips_processed,num_clips) + ENDC + " Processed:\n" + WARNING + "clip_id: " + ENDC +"%d"%int(k2) + WARNING + "\npath: " + ENDC + "%s"%(v2['path'])

     song_struct['name'] = k           
     json_struct[v['song_id']] = song_struct
     
     if(clips_processed > 10):
        break
     
  json_struct["clip_idx"] = clip_idx_map
  
  fh5.create_dataset('song_clip',data=song_clip)
  fh5.close()
        
  json_file = args.h5_output.split(".")[0] + "_additional.json"
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

