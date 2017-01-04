import csv
import json
from parsers.parser_base import parser_base
import os
from parsers.utils import addData



class magna_parser(parser_base):
    def __init__(self):
        parser_base.__init__(self)
        self._name = "magna_parser"

    def parse_input(self,args):
       datafile = args["data"]
       csvfile,annofile = datafile.split(',')
       labels = {}
       with open(annofile,"rb") as af:
          tags = csv.DictReader(af,delimiter='\t')
          for t in tags:
             for k in t.keys():
               if t[k] == str(1) and k != "clip_id":
                  try:
                    labels[t["clip_id"]].append(k)   
                  except KeyError:
                    labels[t["clip_id"]] = []
                    labels[t["clip_id"]].append(k)   

       blob = {}
       song_id = 0
       tracks = 0
       with open(csvfile,"rb") as f:
          data = csv.DictReader(f,delimiter='\t')
          for r in data:
             if not r["clip_id"] in labels:
                continue
             path = os.path.join(args["input_dir"],r["mp3_path"])
             if not os.path.exists(path):
                continue
             artist = r["artist"].split()
             additional_tags = r["album"].split()
             if not r["title"] in blob:
                song_id = song_id + 1
                addData(blob,r["title"],song_id,r["clip_id"],r["mp3_path"],labels[r["clip_id"]],-1,artist,additional_tags)
             else:
                addData(blob,r["title"],blob[r["title"]]["song_id"],r["clip_id"],r["mp3_path"],labels[r["clip_id"]],artist,additional_tags)
               
             tracks = tracks + 1
       blob["Total_Songs"] = song_id
       blob["Total_tracks"] = tracks
       return blob        



if __name__ == '__main__':

   parser = magna_parser()

   datafile = "../../../music_data/magna/clip_info_final.csv"
   annofile = "../../../music_data/magna/annotations_final.csv"
   args = {}
   args["data"] = datafile + "," + annofile
   args["input_dir"] = "../../../music_data/magna/"

   blob = parser.parse_input(args)
   print json.dumps(blob,sort_keys=True,indent=4)



