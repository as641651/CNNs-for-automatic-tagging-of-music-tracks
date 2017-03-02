import argparse, os, json

def find_perf(log,p):
   perf = []
   with open(log) as f: 
     for line in f:
       if not "W.MEAN" in line: 
         continue
       if not p in line:
         continue
       perf.append(float(line.split(',')[1][:-2]))
   
   return perf

def append_exp(args,exp):
   idx = str(args.j) 
   record = {}
   record['s'] = args.s 
   record['d'] = args.d 
   record['auc'] = find_perf(args.l, "AUC") 
   record['ap'] = find_perf(args.l, "AP") 
   record['f1'] = find_perf(args.l, "F1") 

   exp[idx] = record
   
def print_records(k,v):
     print k
     if os.path.exists(k):
        with open(k,"r") as f:
          cfg = json.load(f)
          print json.dumps(cfg,sort_keys=True,indent=4)
     print v['s'] + '\t' + v['d']
     print "AUC" + '\t' + "AP" + '\t' + "F1"
     for i in range(len(v['auc'])):
        print "%.3f\t%.3f\t%.3f"%(v['auc'][i], v['ap'][i], v['f1'][i]) 
     print '\n'

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-s',default='',help='semantic name')
  parser.add_argument('-j',default='',help='json file')
  parser.add_argument('-l',default='',help='log')
  parser.add_argument('-d',default='',help='description')
  parser.add_argument('-rm',default='',help='remove record')

  # OPTIONS
  args = parser.parse_args()

  exp_file = "records.json"
  try:
    with open(exp_file,"r") as f:
       try:
         exp = json.load(f)
       except ValueError:
         exp = {}
  except IOError:
     exp = {}

  if args.rm != '':
    if args.rm in exp: del exp[args.rm]
  elif args.l != '':
    append_exp(args,exp)

  if args.l == '' and args.j != '':
    if str(args.j) in exp:
      print_records(str(args.j),exp[str(args.j)])
  elif args.l == '' and args.j == '' and args.s != '':
    for k,v in exp.iteritems():
      if v['s'] == args.s:
        print_records(k,v)    
  else:
    for k,v in exp.iteritems():
      print_records(k,v)

  with open(exp_file,"w") as fw:
     json.dump(exp,fw)
