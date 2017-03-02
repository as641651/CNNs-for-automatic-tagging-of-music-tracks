import numpy as np
import argparse, os, json
from generate_report import write_tex_report
from eval_report import eval_report
from eval_report import categorize

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()

  # OUTPUT settings
  parser.add_argument('--l1',
      default='',
      help='Path to log file 1')
  
  parser.add_argument('--l2',
      default='',
      help='Path to log file 2')

  parser.add_argument('-i',
      default='',
      help='Path to image dir')
  # OPTIONS
  args = parser.parse_args()

  if not os.path.exists(args.l1): 
     print(args.l1 + " does not exist")
     exit(-1)

  if not os.path.exists(args.l2): 
     print(args.l2 + " does not exist")
     exit(-1)

  if not os.path.exists(args.i): 
     print(args.i + " does not exist")
     exit(-1)  

  with open(args.l1, "r") as f:
    log1 = json.load(f)

  with open(args.l2, "r") as f:
    log2 = json.load(f)

  with open("groups.json", "r") as f:
    groups = json.load(f)

  r_auc_l1 = categorize(log1,groups,"auc")
  r_ap_l1 = categorize(log1,groups,"ap")
  r_auc_l2 = categorize(log2,groups,"auc")
  r_ap_l2 = categorize(log2,groups,"ap")

  ap_imgs = []
  auc_imgs = []
  #print(json.dumps(r_auc_l1,indent=4,sort_keys=True))
  #print(json.dumps(r_auc_l2,indent=4,sort_keys=True))
  for k,v in r_auc_l1.iteritems():
    if not v["0"]:
      continue
    cts = {}
    wts = []
    mean1_auc = []
    mean2_auc = []
    mean1_ap = []
    mean2_ap = []

    n_l1 = str(len(v)-1)
    n_l2 = str(len(r_auc_l2[k]) - 1)
    for k1,v1 in v[str(n_l1)].iteritems():
      cts[k1] = log1[str(n_l1)]["count"][k1]
      wts.append(log1[n_l1]["count"][k1])
      mean1_auc.append(v1)
      mean1_ap.append(r_ap_l1[k][n_l1][k1])

    for k1,v1 in r_auc_l2[k][n_l2].iteritems():
      mean2_auc.append(v1)
      mean2_ap.append(r_ap_l2[k][n_l2][k1])
      
    wts = np.array(wts)
    wts = wts/float(sum(wts))

    er_auc = eval_report(cts)
    er_ap = eval_report(cts)

    er_auc.addResult(r_auc_l1[k][n_l1],args.l1)
    er_auc.addResult(r_auc_l2[k][n_l2],args.l2)
    im_path = os.path.join(args.i,"auc_" + k + ".png")
    auc_imgs.append("auc_" + k)
    er_auc.plot("AUC Improvement scores -\n " + k, "AUC-0.5", [-0.5,0.5],sum(mean1_auc*wts),sum(mean2_auc*wts),im_path,True)

    er_ap.addResult(r_ap_l1[k][n_l1],args.l1)
    er_ap.addResult(r_ap_l2[k][n_l2],args.l2)
    im_path = os.path.join(args.i,"ap_" + k + ".png")
    ap_imgs.append("ap_" + k)
    er_ap.plot("AP scores -\n " + k, "AP", [0.,1.],sum(mean1_ap*wts),sum(mean2_ap*wts),im_path,True)

  write_tex_report(args.i,auc_imgs,ap_imgs)  
