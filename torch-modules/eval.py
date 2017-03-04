import _init_paths
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from DrawMatrix import DrawMatrix 
import os
#from sklearn.metrics import precision_recall_curve as prc
#from sklearn.metrics import average_precision_score as aps 


def evaluate(pred,scores):
   prec = {}
   rec = {}
   acc = {}
   f1 = {}
   ap = 0
   min_th = [0.2,0.4,0.6,0.8]
   for th in min_th:
     tp = 0
     fp = 0
     tn = 0
     fn = 0
     for i in range(len(scores)):
       if scores[i] >= th:
          if pred[i] == 1: 
             tp = tp+1
          else:
             fp = fp+1
       else:
          if pred[i] == 1: 
             fn = fn + 1
          else:
             tn = tn+1  
     
     prec[th] = tp/float(max(tp+fp,1))
     rec[th] = tp/float(tp+fn)
     acc[th] = (tp+tn)/float(tp+fp+tn+fn)
     f1[th] = 2*tp/float(2*tp+fp+fn)
   
   
   ap = ap + prec[0.2]*(1.0-rec[0.2])           
   ap = ap + prec[0.4]*(rec[0.2] - rec[0.4])           
   ap = ap + prec[0.6]*(rec[0.4] - rec[0.6])           
   ap = ap + prec[0.8]*(rec[0.6] - rec[0.8]) 
   

   results = {}
   results["acc"] = acc           
   results["prec"] = prec
   results["rec"] = rec
   results["f1"] = f1
   results["ap"] = ap
   #print(pred,scores)
   results["auc"] = roc_auc_score(pred,scores)

   return results
              
def getPatternMatrix(records,th,idx_to_token):
  n = max(records["labels_in_test"])
  pm = {}
  for cls in records["labels_in_test"]:
    gt = {}
    gt[cls] = 0
    for v in records[str(cls)]:
       if v["confidence"] >= th:
          if v["target"] == 1:
            gt[cls] = gt[cls] + 1
          else:
            for g in v["gt"]:
              #nv = float(len(v["gt"]))
              if not g in gt:
                 gt[g] = 1
              else:
                 gt[g] = gt[g] + 1
    if len(gt) > 0:
       #p = []
       #l = []
       p = {}
       for k,v in gt.iteritems():
         # p.append(v)
         # l.append(idx_to_token[str(k)])
         p[str(idx_to_token[str(k-1)])] = v
       pm[str(idx_to_token[str(cls-1)])] = p #{}
       #pm[str(idx_to_token[str(cls)])]["p"] = p
       #pm[str(idx_to_token[str(cls)])]["l"] = l

  return pm
  
with open("tmp.json","r") as f:
  records = json.load(f)

with open(records["info_json"],"r") as f:
  idx_to_token = json.load(f)["idx_to_token"]

ch_id = 0
results = {}
if records["log_path"] != '':
  if os.path.exists(records["log_path"]):
     with open(records["log_path"],"r") as f:
       results = json.load(f)
       ch_id = len(results)

pm = getPatternMatrix(records,0.2,idx_to_token)
print json.dumps(pm,sort_keys=True,indent=4)
#DrawMatrix(pm)

cls = 0
cls_auc = []
cls_ap = []
cls_f1 = []
cls_acc = []
weights = []
log = {}

log["ap"] = {}
log["auc"] = {}
log["prec"] = {}
log["rec"] = {}
log["f1"] = {}
log["acc"] = {}
log["count"] = {}

for cls in records["labels_in_test"]:
   count = 0
   scores = []
   pred = []
   
   for v in records[str(cls)]:
      scores.append(v["confidence"])
      pred.append(v["target"])
      if(v["target"] == 1):    
        count = count+1
   res = evaluate(pred,scores)
   auc = res["auc"]
   ap = res["ap"]
   prec = res["prec"][0.2]
   rec = res["rec"][0.2]
   f1 = res["f1"][0.2]
   acc = res["acc"][0.2]
   print str(idx_to_token[str(cls-1)]) + " :\n AUC: %.2f, AP: %.2f, PREC@0.2: %.2f, RECALL@0.2: %.2f, F1@0.2: %.2f, ACC@0.2: %.2f COUNT: %d "%(auc,ap,prec,rec,f1,acc,count)
   cls_auc.append(auc)
   cls_ap.append(ap)
   cls_f1.append(f1)
   cls_acc.append(acc)
   weights.append(count)
   log["ap"][str(idx_to_token[str(cls-1)])] = ap
   log["auc"][str(idx_to_token[str(cls-1)])] = auc -0.5
   log["rec"][str(idx_to_token[str(cls-1)])] = rec
   log["prec"][str(idx_to_token[str(cls-1)])] = prec
   log["f1"][str(idx_to_token[str(cls-1)])] = f1
   log["acc"][str(idx_to_token[str(cls-1)])] = acc
   log["count"][str(idx_to_token[str(cls-1)])] = count

weights = np.array(weights)
cls_auc = np.array(cls_auc)
cls_ap = np.array(cls_ap)
cls_f1 = np.array(cls_f1)
cls_acc = np.array(cls_acc)

weights = weights/float(sum(weights))
m_auc = sum(cls_auc*weights)#/float(len(cls_auc))
m_ap = sum(cls_ap*weights)#/float(len(cls_ap))
m_f1 = sum(cls_f1*weights)
m_acc = sum(cls_acc*weights)
log["m_auc"] = m_auc - 0.5
log["m_ap"] = m_ap
log["m_f1"] = m_f1
log["m_acc"] = m_acc

print("W.MEAN AUC : ", m_auc)
print("W.MEAN AP : ", m_ap)
print("W.MEAN F1@0.2 : ", m_f1)
print("W.MEAN ACC@0.2 : ", m_acc)

if records["log_path"] != '':
  results[ch_id] = log
  with open(records["log_path"],"w") as f:
     json.dump(results,f)
   
