# coding=utf8
import argparse, os, json

def searchTags(dataDict,words):
  
  songs = {}
  for k,v in dataDict.iteritems():
       if k == 'song_id_to_name':
          continue

       for k2,v2 in v.iteritems():
          if k2 != 'song_id' and k2 != 'info_tags' and k2 != 'info_tokens':
             for w in v2["labels"]:
                if w in words:
                  songs[k] = v2["labels"]
                  break

  return songs
                  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-w',
      default='',
      help='words separated by comma')
  parser.add_argument('-c',
      default='',
      help='json data file')

  # OPTIONS
  args = parser.parse_args()
  words = args.w.split(",")
  if os.path.exists(args.c):
     with open(args.c,"r") as f:
       dataDict = json.load(f)
  else:
     print "json path does not exist " + args.c
     exit(-1)

  print "searching for tags : "
  print words

  if words:
    songs = searchTags(dataDict,words)
    for k,v in songs.iteritems():
      word_str = ""
      for w in v:
        word_str = word_str + ", " + w
       
      try:
        print k + " :   " + word_str
      except UnicodeEncodeError:
        pass
    #print(json.dumps(songs,indent=4,sort_keys=True))

  
