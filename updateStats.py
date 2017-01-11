#! /usr/bin/env python
"""Aravind Sankaran, RWTH"""

from optparse import OptionParser
import json
import os

"""Should be executed whenever the database is updated.
   
   parses the file from line number 'startline'
   inserts words after the 3rd "|" in each line

   Inputs:
	datafile (default : stats/classify-Sep16.org)

   outputs:
        words.json: list of all words used with its number of occurances
        years.json: list of years with number of tracks in each year"""

def CreateDict(datafile):
    lines = [line.rstrip() for line in open(datafile)]
    wordDict = {}
    years = {}
    artists = {}
    start = False
    num_songs = 0
    num_words = 0
    for l in range(len(lines)):
       words = lines[l].split()
       slc = 0
       found_year = False
       labeled = False
       for w in words:
          if w[:6] == "|----+" and not start:
             print w
             start = True
          if not start:
             continue
          if w[:4] == "====" and start:
             start = False
             print w
             break
          if w[-1] == ",": ##remove the comma that sticks at the end of the word
             w = w[:-1]
          if slc > 1 and not found_year:       
            y = w.split("/")[0]
            if y in years:
               years[y] = years[y]+1
            else:
               years[y] = 1
            found_year = True
          if slc == 3 and w != '|':
            labeled = True
            if w in wordDict:
               wordDict[w] = wordDict[w]+1
            else:
               wordDict[w] = 1
               num_words = num_words + 1
          if w == "|":
            slc = slc + 1
       if labeled:
         num_songs = num_songs+1
    return wordDict,years,num_songs,num_words

optionParser = OptionParser()
optionParser.add_option("-f",dest="datafile", default="classify-Dec16.org", help="Default: classify-Dec16.org")
(options,args) = optionParser.parse_args()

print "Data File: ", options.datafile
wordlist,years,num_songs,num_words = CreateDict(options.datafile)
print json.dumps(wordlist,sort_keys=True,indent=4)
print json.dumps(years,sort_keys=True,indent=4)
print "Total songs tagged : " + str(num_songs)
print "Total Tags used : " + str(num_words)
if os.path.exists("stats/"):
   with open("stats/words.json","w") as wo:
      wo.write(json.dumps(wordlist,wo,sort_keys=True,indent=4))
   with open("stats/years.json","w") as yo:
      yo.write(json.dumps(years,yo,sort_keys=True,indent=4))
