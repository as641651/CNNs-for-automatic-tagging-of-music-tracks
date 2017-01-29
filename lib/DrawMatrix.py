import numpy as np
import matplotlib.pyplot as plt

def DrawMatrix(pm,path="cm.png"):

  #fig = plt.figure()
#  plt.clf()
  f, ax = plt.subplots(len(pm))
  i = 0
  for cls,v in pm.iteritems():
      d = np.zeros((1,len(v["p"])))
      d[0] = np.array(v["p"])
    
      res = ax[i].table(cellText=d,colLabels=v["l"],rowLabels = [cls], rowLoc='center', colLoc='center')
      ax[i].axis('off')
      #ax[i].axis('tight')

      i = i + 1
    #  if i > 3:
    #   break
  
#  fig.tight_layout()
  plt.show()
#  plt.savefig(path, format='png')

