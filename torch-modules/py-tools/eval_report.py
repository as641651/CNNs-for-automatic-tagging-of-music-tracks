import numpy as np
import matplotlib.pyplot as plt

def categorize_tags(log,groups):
  category = {}
  category["none"] = []
  for k,v in groups.iteritems():
    category[k] = []
  for k,v in  log["0"]["ap"].iteritems():
     found = False
     for k1,v1 in groups.iteritems():
       if k in v1:
         category[k1].append(k)
         found = True
     if not found:
       category["none"].append(k)

  return category

def categorize(logs,groups,metric,category=None):
   results = {}
   if not category:
     category = categorize_tags(logs,groups)
   for k,v in category.iteritems():
     results[k] = {}
     for k1,v1 in logs.iteritems():
       results[k][k1] = {}
       for l in v:
         results[k][k1][l] = v1[metric][l]

   return results   

class eval_report(object):
   
   def __init__(self,cts):
      self.results = []
      self.trace_id = []
      self.cts = cts
      self.color = ['r','g','b','m','c','y','k']

   def addResult(self,r,i):
      self.results.append(r)
      self.trace_id.append(i)

   def plot(self,title,ylab,ylim,gmean,mean,im_path,diff=False):
     fig, ax = plt.subplots()
     xl = []
     for k,v in self.results[0].iteritems():
        xl.append(k+"(" + str(self.cts[k]) + ")")
     idx = np.arange(len(self.results[0]))
     bar_width = 0.15
     opacity = 0.8
     x_ticks_size = 8

     plt.xlabel('Tags')
     plt.ylabel(ylab)
     plt.title(title)
     ax.tick_params(axis='x', labelsize=x_ticks_size)
     plt.xticks(idx + bar_width, xl,rotation="vertical")
     plt.ylim(ylim[0],ylim[1])
     l1 = 'Group Mean'
     l2 = 'Mean'
     if diff:
       ax.axhline(gmean,color='red',linewidth=1,ls = '--')
       ax.axhline(mean,color='green',linewidth=1,ls = '--')
     else:
       ax.axhline(gmean,color='blue',linewidth=1,label=l1,ls = '--')
       ax.axhline(mean,color='red',linewidth=1,label=l2,ls = '--')
     ax.axhline(0,color='black',linewidth=2)
     rects = []
     for i in range(len(self.results)):
       y = []
       for k,v in self.results[i].iteritems():
         y.append(v)
       j = i%6
       rects.append(plt.bar(idx + i*bar_width, y,bar_width,alpha=opacity,label=self.trace_id[i],color=self.color[j]))

     plt.legend(fontsize=8)
     plt.tight_layout()
     fig.savefig(im_path)


