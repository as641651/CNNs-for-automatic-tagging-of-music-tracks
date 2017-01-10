import csv
import json
from parser_base import parser_base
import os
from utils import addData



class paolo_parser(parser_base):
    def __init__(self):
        parser_base.__init__(self)
        self._name = "paolo_parser"

    def parse_input(self,args):
       datafile = args["data"]
       lines = [line.rstrip() for line in open(datafile)]
       start = False
       wordDict = {}
       num_words = 0
       blob = {}
       song_id = 0
       for l in range(len(lines)):
           words = lines[l].split()
           slc = 0
           found_genre = False
           labels = []
           year = -1
           title = ""
           path = ""
           for w in words:
              if w[:6] == "|----+" and not start:
                start = True
                break
              if not start:
                continue
              if w[:4] == "====" and start:
                 start = False
                 break
              #print words
              if words[1] != '|' and not found_genre:
                 labels.append(words[1])
                 found_genre = True
              
              if w[-1] == ",": ##remove the comma that sticks at the end of the word
                 w = w[:-1]

              if slc == 3 and w != '|':
                 labels.append(w)
              if w == '|':
                 slc = slc + 1

           if start and labels:
             for la in labels:
                try:
                  wordDict[la] = wordDict[la]+1
                except KeyError:
                  wordDict[la] = 1
                  num_words = num_words+1
                  
             year = int(lines[l].split("/")[0].split("|")[-1])
             path = str(year) + "/" + lines[l].split("/")[1].split("|")[0].rstrip()
             path = os.path.join(args["input_dir"],path)
             title = lines[l].split("/")[1].split("|")[0].rstrip()[:-4]
             if True: # os.path.exists(path):
               song_id = song_id + 1
               additional_tags = title.split("-")[0].split()
               addData(blob,title,song_id,song_id,path,labels,year,"",additional_tags)

       blob["Total_Songs"] = song_id
       #print(wordDict)
       #print(num_words)
       #print(song_id)
       return blob



if __name__ == '__main__':

   parser = paolo_parser()

   datafile = "../../classify-Dec16.org"
   args = {}
   args["data"] = datafile
   args["input_dir"] = "../../../music_data/waves/"

   blob = parser.parse_input(args)
   print json.dumps(blob,sort_keys=True,indent=4)



