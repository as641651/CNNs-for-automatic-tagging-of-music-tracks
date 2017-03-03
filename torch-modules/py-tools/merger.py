# coding=utf8
import argparse, os, json

def replace(phrase,replacements):
  for k, v in replacements.iteritems():
    phrase = phrase.replace(k, v)

  return phrase


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  # OUTPUT settings
  parser.add_argument('-c',
      default='',
      help='datafile')
  parser.add_argument('-w',
      default='',
      help='out datafile')
  parser.add_argument('-l',
      default='',
      help='synon json')

  # OPTIONS
  args = parser.parse_args()

  with open(args.c,"r") as f:
     dataDict = json.load(f)

  with open(args.l,"r") as f:
     replacements = json.load(f)

  for k,v in dataDict.iteritems(): 
       if k == 'song_id_to_name':
          continue
       for k2,v2 in v.iteritems():
          if k2 != 'song_id' and k2 != 'info_tags' and k2 != 'info_tokens':
             nl = []
             for l in v2["labels"]:
                nl.append(replace(l,replacements))
             dataDict[k][k2]["labels"] = nl

  with open(args.w, "w") as f:
     json.dump(dataDict,f)
