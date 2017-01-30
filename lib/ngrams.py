# coding=utf8

import argparse, os, json

def ngrams(gt,vocab_size):

   u = {}
   b = {}
   t = {}
   for l in range(vocab_size):
      u[l] = 0.0
      b[l] = {}
      t[l] = {}
      for l1 in range(vocab_size):
          b[l][l1] = 0.0
          t[l][l1] = {}
          for l2 in range(vocab_size):
              t[l][l1][l2] = 0.0

   for k,v in gt.iteritems():
      for l in v:
        u[l] += 1.0
        for l1 in v[(v.index(l)):]:
          if l != l1:
            b[l][l1] += 1.0
            b[l1][l] += 1.0
            for l2 in v[v.index(l1):]:
              if l2 != l1 and l2!=l:
                t[l][l1][l2] += 1.0
                t[l][l2][l1] += 1.0
                t[l1][l][l2] += 1.0
                t[l1][l2][l] += 1.0
                t[l2][l][l1] += 1.0
                t[l2][l1][l] += 1.0
   """  
   for i in u:
     for j in b[i]:
       for k in t[i][j]:
         t[i][j][k] = t[i][j][k]/max(b[i][j],1)
         t[i][k][j] = t[i][k][j]/max(b[i][j],1)
         t[j][i][k] = t[j][i][k]/max(b[i][j],1)
         t[j][k][i] = t[j][k][i]/max(b[i][j],1)
         t[k][i][j] = t[k][i][j]/max(b[i][j],1)
         t[k][j][i] = t[k][j][i]/max(b[i][j],1)
       b[i][j] = b[i][j]/max(u[i],1)
       b[j][i] = b[j][i]/max(u[i],1)
     u[i] = u[i]/float(vocab_size) 
     """

   return u,b,t



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # OUTPUT settings
  parser.add_argument('-i','--input_file',
      default='',
      help='Path to input json file')

  # OPTIONS
  args = parser.parse_args()

  if not args.input_file: 
     print("No iput file")
     exit(-1)  

  with open(args.input_file, "r") as f:
    exp_info = json.load(f)

  vocab_size = len(exp_info["idx_to_token"])
  gt = exp_info["gt"]
  u,b,t = ngrams(gt,vocab_size)
  #print(json.dumps(t,indent=4))
  #print(json.dumps(u,indent=4))
