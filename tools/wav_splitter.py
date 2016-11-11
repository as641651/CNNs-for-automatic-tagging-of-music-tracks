#! /usr/bin/env python

import utils
import wave
import json
import os
from optparse import OptionParser

"""Creates a separate directory for each song and stores the chunks in .wav format.
   The segmentation details of the tracks are in "database.json". Please make sure that 
   this file is in the same directory as the python script"""

OKGREEN = '\033[92m'
ENDC = '\033[0m'

optionParser = OptionParser()
optionParser.add_option("--dir",dest="dirPath", default="/home/sankaran/Thesis/data/Waves", help="Main directory containing the songs :(eg /home/Waves)")
optionParser.add_option("--db",dest="db", help="database file")
optionParser.add_option("-y",dest="year", default="2005", help="Sub folder name (default: 2005)")
optionParser.add_option("-o",dest="chunk_dir", default="", help="Path for the chunk dir (Default: Chunks are created in a separate folder inside the actual data path)")
(options,args) = optionParser.parse_args()

dirPath = options.dirPath

if not options.db:
   print "Please specify a database file (eg --db ../databases/database_Paolo.json)"
   quit()

with open(options.db) as f:
     data = json.load(f)
     
chunkDir = options.chunk_dir

for e in data:
   if (data[e]["year"] != str(options.year)):
       continue
   path = os.path.join(dirPath,data[e]["year"])
   if not os.path.exists(path):
      print path, "does not exist"
      continue
   filePath = os.path.join(path,data[e]["name"])
   if options.chunk_dir:
       path = os.path.join(options.chunk_dir,data[e]["year"]+"_chunks")
   chunkDir = os.path.join(path,data[e]["name"].split(".wav")[0] + "_chunks")
   if not os.path.exists(chunkDir):
      os.makedirs(chunkDir)
   segments = data[e]["segments"]
   start = 0
   for s in range(1,len(segments)):
      end = int(segments[s])
      if end - start >= 30:
         outFile = "chunk_%d_%d.wav"%(start,end)
         outFile = os.path.join(chunkDir,outFile)
         utils.make_chunk(filePath,outFile,start,end)
         start = end
   print OKGREEN + "processed " + ENDC + filePath
