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


with h5py.File('weights_theano.hdf5','r') as hf:
   for k in hf.keys():
      k = str(k)
      
      if(k == 'ConvBNEluDr'):
        addWeights(layer,k,hf)
        data = hf.get(str(k))

        net["conv1"] = nn.SpatialConvolutionMM(1,32,3,3,1,1)
        net["bn1"] = nn.SpatialBatchNormalization(32)
        net["conv2"] = nn.SpatialConvolutionMM(32,32,3,3,1,1)
        net["bn2"] = nn.SpatialBatchNormalization(32)
        net["conv3"] = nn.SpatialConvolutionMM(32,32,3,3,1,1)
        net["bn3"] = nn.SpatialBatchNormalization(32)
        net["conv4"] = nn.SpatialConvolutionMM(32,32,3,3,1,1)
        net["bn4"] = nn.SpatialBatchNormalization(32)
        net["conv5"] = nn.SpatialConvolutionMM(32,32,3,3,1,1)
        net["bn5"] = nn.SpatialBatchNormalization(32)

        for k2 in data.keys():
          print(k2)

          if(k2 == 'convolution2d_1_W'):
            transferW(net["conv1"].weight,layer[str(k)][str(k2)].reshape(32,9)) 
          if(k2 == 'convolution2d_1_b'):
            transferB(net["conv1"].bias,layer[str(k)][str(k2)]) 
          if(k2 == 'batchnormalization_1_beta'):
            transferB(net["bn1"].bias,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_1_gamma'):
            transferB(net["bn1"].weight,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_1_running_mean'):
            transferB(net["bn1"].running_mean,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_1_running_std'):
            transferB(net["bn1"].running_var,layer[str(k)][str(k2)])


          if(k2 == 'convolution2d_2_W'):
            transferW(net["conv2"].weight,layer[str(k)][str(k2)].reshape(32,288)) 
          if(k2 == 'convolution2d_2_b'):
            transferB(net["conv2"].bias,layer[str(k)][str(k2)]) 
          if(k2 == 'batchnormalization_2_beta'):
            transferB(net["bn2"].bias,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_2_gamma'):
            transferB(net["bn2"].weight,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_2_running_mean'):
            transferB(net["bn2"].running_mean,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_2_running_std'):
            transferB(net["bn2"].running_var,layer[str(k)][str(k2)])


          if(k2 == 'convolution2d_3_W'):
            transferW(net["conv3"].weight,layer[str(k)][str(k2)].reshape(32,288)) 
          if(k2 == 'convolution2d_3_b'):
            transferB(net["conv3"].bias,layer[str(k)][str(k2)]) 
          if(k2 == 'batchnormalization_3_beta'):
            transferB(net["bn3"].bias,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_3_gamma'):
            transferB(net["bn3"].weight,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_3_running_mean'):
            transferB(net["bn3"].running_mean,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_3_running_std'):
            transferB(net["bn3"].running_var,layer[str(k)][str(k2)])


          if(k2 == 'convolution2d_4_W'):
            transferW(net["conv2"].weight,layer[str(k)][str(k2)].reshape(32,288)) 
          if(k2 == 'convolution2d_4_b'):
            transferB(net["conv4"].bias,layer[str(k)][str(k2)]) 
          if(k2 == 'batchnormalization_4_beta'):
            transferB(net["bn4"].bias,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_4_gamma'):
            transferB(net["bn4"].weight,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_4_running_mean'):
            transferB(net["bn4"].running_mean,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_4_running_std'):
            transferB(net["bn4"].running_var,layer[str(k)][str(k2)])


          if(k2 == 'convolution2d_5_W'):
            transferW(net["conv5"].weight,layer[str(k)][str(k2)].reshape(32,288)) 
          if(k2 == 'convolution2d_5_b'):
            transferB(net["conv5"].bias,layer[str(k)][str(k2)]) 
          if(k2 == 'batchnormalization_5_beta'):
            transferB(net["bn5"].bias,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_5_gamma'):
            transferB(net["bn5"].weight,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_5_running_mean'):
            transferB(net["bn5"].running_mean,layer[str(k)][str(k2)])
          if(k2 == 'batchnormalization_5_running_std'):
            transferB(net["bn5"].running_var,layer[str(k)][str(k2)])

PyTorchAug.save("compact_cnn.t7",net)
#print pprint.pprint(layer,width=1)
#print json.dumps(layer,sort_keys=True,indent=4)

        
