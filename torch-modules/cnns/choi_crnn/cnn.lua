require 'nn'


function input_normalization(model)

   local net = nn.Sequential()
   local bn0 = nn.BatchNormalization(1366)
   bn0.bias = model["bn_0_freq"].bias[{{1,1366}}]
   bn0.weight = model["bn_0_freq"].weight[{{1,1366}}]
   bn0.running_mean = model["bn_0_freq"].running_mean[{{1,1366}}]
   bn0.running_var = model["bn_0_freq"].running_var[{{1,1366}}]
  
   net:add(bn0)

   return net

end

function build_choi_cnn(model)

   local net = nn.Sequential()
 
   net:add(model["conv1"])
   net:add(model["bn1"])
   local elu1 = nn.ELU()
   elu1.inplace = true
   net:add(elu1)
   net:add(nn.SpatialMaxPooling(2,2,2,2,0,0))
   net:add(nn.Dropout(0.1))

   net:add(model["conv2"])
   net:add(model["bn2"])
   local elu2 = nn.ELU()
   elu2.inplace = true
   net:add(elu2)
   net:add(nn.SpatialMaxPooling(3,3,3,3,0,0))
   net:add(nn.Dropout(0.1))

   net:add(model["conv3"])
   net:add(model["bn3"])
   local elu3 = nn.ELU()
   elu3.inplace = true
   net:add(elu3)
   net:add(nn.SpatialMaxPooling(4,4,4,4,0,0))
   net:add(nn.Dropout(0.1))

   net:add(model["conv4"])
   net:add(model["bn4"])
   local elu4 = nn.ELU()
   elu4.inplace = true
   net:add(elu4)
   net:add(nn.SpatialMaxPooling(4,1,4,1,0,0))
--   net:add(nn.Dropout(0.1))

   return net
end


local net = {}
local model = torch.load('cnns/choi_crnn/choi_crnn.t7')

net.input_normalization = input_normalization(model)
net.cnn = build_choi_cnn(model)

function net.type(dtype)
   net.cnn:type(dtype)
   net.input_normalization:type(dtype)
end

function net.forward(input)
 
     local nDim = input:size():size()
     assert(nDim == 4, "Suports only batch mode" )
     assert(input:size(2) == 1, "Suports only mono" )
     assert(input:size(4) == 1366, "Suports only 1366 xdim" )
  
     num_samples = input:size(1)
     feat_len = input:size(3)      

     local norm_input = net.input_normalization:forward(input:view(num_samples*feat_len,1366))
     norm_input = norm_input:view(num_samples,1,feat_len, 1366)
     local cnn_output = net.cnn:forward(norm_input)
     cnn_output = cnn_output:view(cnn_output:size(1),cnn_output:size(2),cnn_output:size(4))

     return cnn_output
end   


--print(net.cnn.modules)
return net

