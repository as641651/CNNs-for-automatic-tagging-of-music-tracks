# coding=utf8
import argparse, os, json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-s',
      default='',
      help='song to add for testing')
  parser.add_argument('-c',
      default='',
      help='json data file')
  parser.add_argument('-v',
      default=False,
      action="store_true",
      help='show labels')
  parser.add_argument('-r',
      default=False,
      action="store_true",
      help='remove entries')

  # OPTIONS
  args = parser.parse_args()
  song = args.s

  cache_file = "test_cache.json"
  if args.r:
    if os.path.exists(cache_file):
      os.remove(cache_file)

  if os.path.exists(args.c):
     with open(args.c,"r") as f:
       dataDict = json.load(f)
  else:
     print "json path does not exist " + args.c
     exit(-1)

      
  try:
    with open(cache_file,"r") as f:
      try:
        cacheD = json.load(f)
      except ValueError:
        cacheD = {}
        cacheD["songs"] = []
        cacheD["lt"] = {}
  except IOError:
    cacheD = {}
    cacheD["songs"] = []
    cacheD["lt"] = {}

  try:
    v = dataDict[song]
  except KeyError:
    for s in cacheD["songs"]:
      print s
    exit(-1)

  if not song in cacheD["songs"]:
    cacheD["songs"].append(song)
    print song
    for k2,v2 in v.iteritems():
      if k2 != 'song_id' and k2 != 'info_tags' and k2 != 'info_tokens':
        for w in v2["labels"]:
          try:
            cacheD["lt"][w] = cacheD["lt"][w] + 1
          except KeyError:
            cacheD["lt"][w] = 1

  with open(cache_file,"w") as fw:
     json.dump(cacheD,fw)

  #for s in cacheD["songs"]:
  #  print s
  if args.v:
    print(json.dumps(cacheD["lt"],indent=4,sort_keys=True))
    print "num songs : " + str(len(cacheD["songs"]))
    print "num labels in test : " + str(len(cacheD["lt"]))


  
