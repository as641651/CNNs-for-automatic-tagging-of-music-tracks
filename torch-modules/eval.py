import _init_paths
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from DrawMatrix import DrawMatrix 
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
   
   
   ap = ap + prec[0.2]*(rec[0.2])           
   ap = ap + prec[0.4]*(abs(rec[0.4] - rec[0.2]))           
   ap = ap + prec[0.6]*(abs(rec[0.6] - rec[0.4]))           
   ap = ap + prec[0.8]*(abs(rec[0.8] - rec[0.6]))

   results = {}
   results["acc"] = acc           
   results["prec"] = prec
   results["rec"] = rec
   results["f1"] = f1
   results["ap"] = ap
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
         p[str(idx_to_token[str(k)])] = v
       pm[str(idx_to_token[str(cls)])] = p #{}
       #pm[str(idx_to_token[str(cls)])]["p"] = p
       #pm[str(idx_to_token[str(cls)])]["l"] = l

  return pm
  
with open("tmp.json","r") as f:
  records = json.load(f)

with open(records["info_json"],"r") as f:
  idx_to_token = json.load(f)["idx_to_token"]

pm = getPatternMatrix(records,0.2,idx_to_token)
print json.dumps(pm,sort_keys=True,indent=4)
#DrawMatrix(pm)

cls = 0
cls_auc = []
cls_ap = []
cls_f1 = []
weights = []
for cls in records["labels_in_test"]:
   count = 0
   scores = []
   pred = []
   
   for v in records[str(cls)]:
      scores.append(v["confidence"])
      pred.append(v["target"])
      if(v["target"] == 1):    
        count = count+1
    #  auc = roc_auc_score(pred,scores)
    #  prec, recall, th = prc(pred,scores)
    #  ap = average_precision(prec,recall)
   res = evaluate(pred,scores)
   auc = res["auc"]
   ap = res["ap"]
   prec = res["prec"][0.2]
   rec = res["rec"][0.2]
   f1 = res["f1"][0.2]
   print str(idx_to_token[str(cls)]) + " :\n AUC: %.2f, AP: %.2f, PREC@0.2: %.2f, RECALL@0.2: %.2f, F1@0.2: %.2f, COUNT: %d "%(auc,ap,prec,rec,f1,count)
   cls_auc.append(auc)
   cls_ap.append(ap)
   cls_f1.append(f1)
   weights.append(count)

weights = np.array(weights)
cls_auc = np.array(cls_auc)
cls_ap = np.array(cls_ap)
cls_f1 = np.array(cls_f1)

weights = weights/float(sum(weights))
m_auc = sum(cls_auc*weights)#/float(len(cls_auc))
m_ap = sum(cls_ap*weights)#/float(len(cls_ap))
m_f1 = sum(cls_f1*weights)
print("W.MEAN AUC : ", m_auc)
print("W.MEAN AP : ", m_ap)
print("W.MEAN F1@0.2 : ", m_f1)


