#! /usr/bin/env python
"""Aravind Sankaran, RWTH"""

from optparse import OptionParser
import json

"""Should be executed whenever the database is updated.
   
   parses the file from line number 'startline'
   inserts words after the 3rd "|" in each line

   Inputs:
	datafile (default : stats/classify-Sep16.org)
        startline (defulat : 45)

   outputs:
        words.json: list of all words used with its number of occurances
        years.json: list of years with number of tracks in each year"""

def CreateDict(datafile,startline):
    lines = [line.rstrip() for line in open(datafile)]
    wordDict = {}
    years = {}
    artists = {}
    for l in range(startline,len(lines)):
       words = lines[l].split()
       slc = 0
       found_year = False
       for w in words:
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
            if w in wordDict:
               wordDict[w] = wordDict[w]+1
            else:
               wordDict[w] = 1
          if w == "|":
            slc = slc + 1
    return wordDict,years

optionParser = OptionParser()
optionParser.add_option("-f",dest="datafile", default="classify-Sep16.org", help="Default: classify-Sep16.org")
optionParser.add_option("-s",dest="startline", default="45", help="line number from which words should be searched. Default: 45")
(options,args) = optionParser.parse_args()

print "Data File: ", options.datafile
wordlist,years = CreateDict(options.datafile,int(options.startline))
print json.dumps(wordlist,sort_keys=True,indent=4)
print json.dumps(years,sort_keys=True,indent=4)
with open("stats/words.json","w") as wo:
   wo.write(json.dumps(wordlist,wo,sort_keys=True,indent=4))
with open("stats/years.json","w") as yo:
   yo.write(json.dumps(years,yo,sort_keys=True,indent=4))
