# coding=utf8
import argparse, os, json
import copy

def filter_labels(labels,filterl):

  slabels = []
  for s in filterl:
    if s in labels:
      slabels.append(s)

  return slabels

def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v)) 
        for k, v in dictionary.items())

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
      help='label json file')

  # OPTIONS
  args = parser.parse_args()

  lines = [line.rstrip() for line in open(args.l)]
  print(lines)
  print(len(lines))

  with open(args.c,"r") as f:
     dataDict = json.load(f)


  newDict = dataDict.copy()
  newDict = convert_keys_to_string(newDict) 
  filtered = {}
  for k,v in dataDict.iteritems(): 
       if k == 'song_id_to_name':
          continue
       for k2,v2 in v.iteritems():
          if k2 != 'song_id' and k2 != 'info_tags' and k2 != 'info_tokens':
             nl = filter_labels(v2["labels"],lines)
             if nl:    
               newDict[k][k2]["labels"] = nl
               for n in nl:
                 filtered[n] = 1
             else:
               del newDict[k][k2]

  with open(args.w, "w") as f:
     json.dump(newDict,f)

