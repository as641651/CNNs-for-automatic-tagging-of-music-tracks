# _*_ coding:utf-8 _*_
import argparse, os, json

def check_sub_genre(labels):
  subgenre = [
    "B",
    "D",
    "E",
    "F",
    "H",
    "S",
    "T",
    "hH",
    "Tr",
    "TH",
  ]

  slabels = []
  for s in subgenre:
    if s in labels:
      slabels.append(s)

  return slabels


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  # OUTPUT settings
  parser.add_argument('-c',
      default='',
      help='datafile')
  parser.add_argument('-w',
      default='',
      help='out datafile')

  # OPTIONS
  args = parser.parse_args()

  with open(args.c,"r") as f:
     dataDict = json.load(f)

  newDict = dataDict.copy()
  for k,v in dataDict.iteritems(): 
       if k == 'song_id_to_name':
          continue
       for k2,v2 in v.iteritems():
          if k2 != 'song_id' and k2 != 'info_tags' and k2 != 'info_tokens':
             nl = check_sub_genre(v2["labels"])
             if nl:    
               newDict[k][k2]["labels"] = nl
             else:
               del newDict[k]

  with open(args.w, "w") as f:
     json.dump(newDict,f)
