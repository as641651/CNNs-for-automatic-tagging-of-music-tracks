import csv
import json
from parsers.parser_base import parser_base



class magna_parser(parser_base):
    def __init__(self):
        parser_base.__init__(self)
        self._name = "magna_parser"

    def parse_input(self,datafile):
       csvfile,annofile = datafile.split(',')
       labels = {}
       with open(annofile,"rb") as af:
          tags = csv.DictReader(af,delimiter='\t')
          for t in tags:
             labels[t["clip_id"]] = []
             for k in t.keys():
               if t[k] == str(1) and k != "clip_id":
                  labels[t["clip_id"]].append(k)                        
       blob = {}
       song_id = 0
       tracks = 0
       with open(csvfile,"rb") as f:
          data = csv.DictReader(f,delimiter='\t')
          for r in data:
             if not r["clip_id"] in labels:
                continue
             if not r["title"] in blob:
                blob[r["title"]] = {}
                song_id = song_id + 1
                blob[r["title"]]["song_id"] = song_id
             clip = {}
             clip["path"] = r["mp3_path"]
             clip["labels"] = labels[r["clip_id"]]
             blob[r["title"]][r["clip_id"]] = clip
             tracks = tracks + 1
       blob["Total_Songs"] = song_id
       blob["Total_tracks"] = tracks
       return blob        

"""
datafile = "music_data/magna/clip_info_final.csv"
annofile = "music_data/magna/annotations_final.csv"
blob = Parse_Magna_csv(datafile,annofile)
print json.dumps(blob,sort_keys=True,indent=4)"""
