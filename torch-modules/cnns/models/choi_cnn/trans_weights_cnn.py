import numpy as np
import h5py
import pprint
import PyTorch
from PyTorchAug import nn
import PyTorchAug

net = {}
layer = {}

def addWeights(layer,k,hf):
   layer[str(k)] = {}
   data = hf.get(str(k))
   for k2 in data.keys():
      print "  " + str(k2)
      layer[str(k)][str(k2)] = np.array(data.get(str(k2)))
      print(str(k), str(k2), layer[str(k)][str(k2)].shape)

def transferW(b,a):
  print b.size()
  print a.shape
  for i in range(b.size()[0]):
     for j in range(b.size()[1]):
        b[i][j] = a[i][j]

def transferB(b,a):
  print b.size()
  print a.shape
  for i in range(b.size()[0]):
      b[i] = a[i]

def transferB_2(b,a):
  print b.size()
  print a.shape
  for i in range(b.size()[0]):
      b[i] = a[i]*a[i]


with h5py.File('music_tagger_cnn_weights_theano.h5','r') as hf:
   for k in hf.keys():
      k = str(k)
      print(k)
      
      if(k == 'bn_0_freq'):
         net["bn_0_freq"] = nn.BatchNormalization(1366)
         addWeights(layer,k,hf)
         transferB(net["bn_0_freq"].bias,layer["bn_0_freq"]["bn_0_freq_beta"])
         transferB(net["bn_0_freq"].weight,layer["bn_0_freq"]["bn_0_freq_gamma"])
         transferB(net["bn_0_freq"].running_mean,layer["bn_0_freq"]["bn_0_freq_running_mean"])
         transferB_2(net["bn_0_freq"].running_var,layer["bn_0_freq"]["bn_0_freq_running_std"])
      
      if(k == 'bn1'):
         net["bn1"] = nn.SpatialBatchNormalization(32)
         addWeights(layer,k,hf)
         transferB(net["bn1"].bias,layer["bn1"]["bn1_beta"])
         transferB(net["bn1"].weight,layer["bn1"]["bn1_gamma"])
         transferB(net["bn1"].running_mean,layer["bn1"]["bn1_running_mean"])
         transferB_2(net["bn1"].running_var,layer["bn1"]["bn1_running_std"])
         
      if(k == 'bn2'):
         net["bn2"] = nn.SpatialBatchNormalization(128)
         addWeights(layer,k,hf)
         transferB(net["bn2"].bias,layer["bn2"]["bn2_beta"])
         transferB(net["bn2"].weight,layer["bn2"]["bn2_gamma"])
         transferB(net["bn2"].running_mean,layer["bn2"]["bn2_running_mean"])
         transferB_2(net["bn2"].running_var,layer["bn2"]["bn2_running_std"])
       
      if(k == 'bn3'):
         net["bn3"] = nn.SpatialBatchNormalization(128)
         addWeights(layer,k,hf)
         transferB(net["bn3"].bias,layer["bn3"]["bn3_beta"])
         transferB(net["bn3"].weight,layer["bn3"]["bn3_gamma"])
         transferB(net["bn3"].running_mean,layer["bn3"]["bn3_running_mean"])
         transferB_2(net["bn3"].running_var,layer["bn3"]["bn3_running_std"])
      if(k == 'bn4'):
         net["bn4"] = nn.SpatialBatchNormalization(192)
         addWeights(layer,k,hf)
         transferB(net["bn4"].bias,layer["bn4"]["bn4_beta"])
         transferB(net["bn4"].weight,layer["bn4"]["bn4_gamma"])
         transferB(net["bn4"].running_mean,layer["bn4"]["bn4_running_mean"])
         transferB_2(net["bn4"].running_var,layer["bn4"]["bn4_running_std"])
      if(k == 'bn5'):
         net["bn5"] = nn.SpatialBatchNormalization(256)
         addWeights(layer,k,hf)
         transferB(net["bn5"].bias,layer["bn5"]["bn5_beta"])
         transferB(net["bn5"].weight,layer["bn5"]["bn5_gamma"])
         transferB(net["bn5"].running_mean,layer["bn5"]["bn5_running_mean"])
         transferB_2(net["bn5"].running_var,layer["bn5"]["bn5_running_std"])
       
      if(k == "conv1"):
         net["conv1"] = nn.SpatialConvolutionMM(1,32,3,3)
         addWeights(layer,k,hf)
         transferW(net["conv1"].weight,layer["conv1"]["conv1_W"].reshape(32,9)) 
         transferB(net["conv1"].bias,layer["conv1"]["conv1_b"]) 
      if(k == "conv2"):
         net["conv2"] = nn.SpatialConvolutionMM(32,128,3,3)
         addWeights(layer,k,hf)
         transferW(net["conv2"].weight,layer["conv2"]["conv2_W"].reshape(128,288)) 
         transferB(net["conv2"].bias,layer["conv2"]["conv2_b"]) 
      if(k == "conv3"):
         net["conv3"] = nn.SpatialConvolutionMM(128,128,3,3)
         addWeights(layer,k,hf)
         transferW(net["conv3"].weight,layer["conv3"]["conv3_W"].reshape(128,1152)) 
         transferB(net["conv3"].bias,layer["conv3"]["conv3_b"]) 
      if(k == "conv4"):
         net["conv4"] = nn.SpatialConvolutionMM(192,128,3,3)
         addWeights(layer,k,hf)
         transferW(net["conv4"].weight,layer["conv4"]["conv4_W"].reshape(128,1728)) 
         transferB(net["conv4"].bias,layer["conv4"]["conv4_b"]) 
      if(k == "conv5"):
         net["conv5"] = nn.SpatialConvolutionMM(256,192,3,3)
         addWeights(layer,k,hf)
         transferW(net["conv5"].weight,layer["conv5"]["conv5_W"].reshape(192,2304)) 
         transferB(net["conv5"].bias,layer["conv5"]["conv5_b"]) 
         

PyTorchAug.save("choi_crnn.t7",net)

#print pprint.pprint(layer,width=1)
#print json.dumps(layer,sort_keys=True,indent=4)
        
