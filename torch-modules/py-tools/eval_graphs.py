import numpy as np
import argparse, os, json
from generate_report import write_tex_report
from eval_report import eval_report
from eval_report import categorize

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()

  # OUTPUT settings
  parser.add_argument('-l',
      default='',
      help='Path to log file')

  parser.add_argument('-i',
      default='',
      help='Path to image dir')
  # OPTIONS
  args = parser.parse_args()

  if not os.path.exists(args.l): 
     print(args.l + " does not exist")
     exit(-1)

  if not os.path.exists(args.i): 
     print(args.i + " does not exist")
     exit(-1)  

  with open(args.l, "r") as f:
    log = json.load(f)
  with open("groups.json", "r") as f:
    groups = json.load(f)

  r_auc = categorize(log,groups,"auc")
  r_ap = categorize(log,groups,"ap")
  ap_imgs = []
  auc_imgs = []
  #print(json.dumps(r,indent=4,sort_keys=True))
  for k,v in r_auc.iteritems():
    if not v["0"]:
      continue
    gmean_auc = []
    gmean_ap = []
    cts = {}
    wts = []
    for k1,v1 in v[str(len(v)-1)].iteritems():
      cts[k1] = log[str(len(v)-1)]["count"][k1]
      wts.append(log[str(len(v)-1)]["count"][k1])
      gmean_auc.append(v1)
      gmean_ap.append(r_ap[k][str(len(v)-1)][k1])
    wts = np.array(wts)
    wts = wts/float(sum(wts))
    er_auc = eval_report(cts)
    er_ap = eval_report(cts)

    for i in range(len(v)):
      er_auc.addResult(v[str(i)],str(i))
    im_path = os.path.join(args.i,"auc_" + k + ".png")
    auc_imgs.append("auc_" + k)
    er_auc.plot("AUC Improvement scores -\n " + k, "AUC-0.5", [-0.5,0.5],sum(gmean_auc*wts),log[str(len(v)-1)]["m_auc"],im_path)

    for i in range(len(r_ap[k])):
      er_ap.addResult(r_ap[k][str(i)],str(i))
    im_path = os.path.join(args.i,"ap_" + k + ".png")
    ap_imgs.append("ap_" + k)
    er_ap.plot("AP scores -\n " + k, "AP", [0.,1.],sum(gmean_ap*wts),log[str(len(v)-1)]["m_ap"],im_path)

  write_tex_report(args.i,auc_imgs,ap_imgs)  
