# _*_ coding:utf-8 _*_
import argparse, os, json

def replace(phrase):
  replacements = {
    "moving": "drama",
    "struggente": "drama",
    "passion": "drama",
    "tribal": "afro",
    "samba": "afro",
    "spanish": "afro",
    "latin": "afro",
    "ethnic": "afro",
    "cattiva": "aggressive",
    "pads": "strings",
    "silly": "cheesy",
    "jazzy": "sax",
  }
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

  # OPTIONS
  args = parser.parse_args()

  with open(args.c,"r") as f:
     dataDict = json.load(f)

  for k,v in dataDict.iteritems(): 
       if k == 'song_id_to_name':
          continue
       for k2,v2 in v.iteritems():
          if k2 != 'song_id' and k2 != 'info_tags' and k2 != 'info_tokens':
             nl = []
             for l in v2["labels"]:
                nl.append(replace(l))
             dataDict[k][k2]["labels"] = nl

  with open(args.w, "w") as f:
     json.dump(dataDict,f)
