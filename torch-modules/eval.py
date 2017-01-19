import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score as aps

with open("tmp.json","r") as f:
  records = json.load(f)

print(records["labels_in_test"])
cls = 0
cls_auc = []
cls_ap = []
for i in records["labels_in_test"]:
   scores = []
   pred = []
   cls = cls+1
   if i != None:
      for v in records[str(cls)]:
         scores.append(v["confidence"])
         pred.append(v["target"])
      auc = roc_auc_score(pred,scores)
      ap = aps(pred,scores)
      print(cls,auc,ap)
      cls_auc.append(auc)
      cls_ap.append(ap)

m_auc = sum(cls_auc)/float(len(cls_auc))
m_ap = sum(cls_ap)/float(len(cls_ap))
print("MEAN AUC : ", m_auc)
print("MEAN AP : ", m_ap)

