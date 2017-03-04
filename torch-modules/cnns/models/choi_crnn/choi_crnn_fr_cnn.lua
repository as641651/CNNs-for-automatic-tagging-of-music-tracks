require 'nn'
require 'rnn'

local norm_input = nil
local num_samples = nil
local feat_len = nil

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

function cnn_1(model)

   local net = nn.Sequential()
 
   net:add(model["conv1"])
   net:add(model["bn1"])
   local elu1 = nn.ELU()
   elu1.inplace = true
   net:add(elu1)
   net:add(nn.SpatialMaxPooling(2,2,2,2,0,0))
   --net:add(nn.Dropout(0.1))

   return net
end

function cnn_2(model)

   local net = nn.Sequential()
 
   net:add(model["conv2"])
   net:add(model["bn2"])
   local elu2 = nn.ELU()
   elu2.inplace = true
   net:add(elu2)
   net:add(nn.SpatialMaxPooling(2,2,2,2,0,0))
   --net:add(nn.Dropout(0.1))

   return net
end

function cnn_3(model)

   local net = nn.Sequential()
 
   net:add(model["conv3"])
   net:add(model["bn3"])
   local elu3 = nn.ELU()
   elu3.inplace = true
   net:add(elu3)
   net:add(nn.SpatialMaxPooling(4,4,4,4,0,0))
   --net:add(nn.Dropout(0.1))

   return net
end

function cnn_4(model)

   local net = nn.Sequential()
 
   net:add(model["conv4"])
   net:add(model["bn4"])
   local elu4 = nn.ELU()
   elu4.inplace = true
   net:add(elu4)
   net:add(nn.SpatialMaxPooling(4,3,4,3,0,0))

   return net
end

function gru_1(model)
   return nn.GRU(128,128)
end

function gru_2(model)
   return nn.GRU(128,128)
end

function build_choi_cnn(model)
   local net = nn.Sequential()
   net:add(cnn_1(model))
   net:add(cnn_2(model))
   net:add(cnn_3(model))
   net:add(cnn_4(model))
   net:add(gru_1())
   net:add(gru_2())
   net:add(nn.Dropout(0.3))
   net:training()
   return net
end


local net = {}
net.cache = {}
local model = torch.load('cnns/models/choi_crnn/choi_crnn.t7')
--local model = torch.load('choi_cnn.t7')

net.input_normalization = input_normalization(model)
net.cnn = build_choi_cnn(model)

net.model = nn.Sequential()
net.model:add(net.input_normalization)
net.model:add(net.cnn)
--print(net.model)

function net.type(dtype)
   net.cnn:type(dtype)
   net.input_normalization:type(dtype)
end

function net.forward(input)
     local nDim = input:size():size()
     assert(nDim == 4, "Suports only batch mode" )
     assert(input:size(2) == 1, "Suports only mono" )
     assert(input:size(3) == 96, "Suports only 96 feat len" )
     if input:size(4) < 1366 then
       local tmp = input:clone()
       input = torch.zeros(tmp:size(1),1,tmp:size(3),1366):type(tmp:type())
       input[{{},{},{},{1,tmp:size(4)}}] = tmp
     end
     if input:size(4) > 1366 then
        input = input[{{},{},{},{1,1366}}]:contiguous()
     end

     assert(input:size(4) == 1366, "Suports only 1366 xdim" )
  
     net.cache.num_samples = input:size(1)
     net.cache.feat_len = input:size(3)      

     net.cache.norm_input = net.model:get(1):forward(input:view(net.cache.num_samples*net.cache.feat_len,1366))
     net.cache.norm_input = net.cache.norm_input:view(net.cache.num_samples,1,net.cache.feat_len, 1366)
     net.cache.cnn_1 = net.model:get(2):get(1):forward(net.cache.norm_input)
  --   print("cnn_1 size ", net.cache.cnn_1:size())
     net.cache.cnn_2 = net.model:get(2):get(2):forward(net.cache.cnn_1)
  --   print("cnn_2 size ", net.cache.cnn_2:size())
     net.cache.cnn_3 = net.model:get(2):get(3):forward(net.cache.cnn_2)
  --   print("cnn_3 size ", net.cache.cnn_3:size())
     net.cache.cnn_4 = net.model:get(2):get(4):forward(net.cache.cnn_3)
  --   print("cnn_4 size ", net.cache.cnn_4:size())
     net.cache.cnn_4 = net.cache.cnn_4:view(net.cache.cnn_4:size(1),net.cache.cnn_4:size(2),-1)
     net.cache.cnn_4 = net.cache.cnn_4:permute(3,1,2)
  --   print("cnn_4 size ", net.cache.cnn_4:size())
     net.cache.rnn1 = torch.zeros(net.cache.cnn_4:size(1),net.cache.cnn_4:size(2),net.cache.cnn_4:size(3)):type(net.cache.cnn_4:type())
     net.cache.rnn2 = torch.zeros(net.cache.cnn_4:size(1),net.cache.cnn_4:size(2),net.cache.cnn_4:size(3)):type(net.cache.cnn_4:type())
     for i = 1,net.cache.cnn_4:size(1) do
        net.cache.rnn1[i] = net.model:get(2):get(5):forward(net.cache.cnn_4[i])
        net.cache.rnn2[i] = net.model:get(2):get(6):forward(net.cache.rnn1[i])
     end
  --   print("rnn_1 size ", net.cache.rnn1:size())
  --   print("rnn_2 size ", net.cache.rnn2:size())
     local cnn_output = net.cache.rnn2[net.cache.cnn_4:size(1)]
  --   print("cnn_out size", cnn_output:size())
     net.called_forward = true
     if net.model.train == false then
       net.model:get(2):get(5):forget() 
       net.model:get(2):get(6):forget()
     end
     return cnn_output
end

function net.backward(input, gradOutput)

  assert(net.called_forward, "forward not called")
  if input:size(4) < 1366 then
   local tmp = input:clone()
   input = torch.zeros(tmp:size(1),1,tmp:size(3),1366):type(tmp:type())
   input[{{},{},{},{1,tmp:size(4)}}] = tmp 
  end
  if input:size(4) > 1366 then
   input = input[{{},{},{},{1,1366}}]:contiguous()
  end

  net.cache.grad_rnn2 = net.cache.rnn2.new(#net.cache.rnn2):zero()
  net.cache.grad_rnn2[net.cache.rnn2:size(1)] = gradOutput
  net.cache.grad_rnn1 = net.cache.rnn1.new(#net.cache.rnn1):zero()
  net.cache.grad_cnn_4 = net.cache.cnn_4.new(#net.cache.cnn_4):zero()

  for i = net.cache.cnn_4:size(1),1,-1 do
     net.cache.grad_rnn1[i] = net.model:get(2):get(6):backward(net.cache.rnn1[i],net.cache.grad_rnn2[i])
     net.cache.grad_cnn_4[i] = net.model:get(2):get(5):backward(net.cache.cnn_4[i],net.cache.grad_rnn1[i])
  end

  net.cache.grad_cnn_4 = net.cache.grad_cnn_4:view(1,net.cache.grad_cnn_4:size(1),net.cache.grad_cnn_4:size(2),net.cache.grad_cnn_4:size(3))
 -- print("grad_cnn_4 ", net.cache.grad_cnn_4:size())
  net.cache.grad_cnn_4 = net.cache.grad_cnn_4:permute(3,4,1,2)
 -- print("grad_cnn_4 ", net.cache.grad_cnn_4:size())
  net.model:get(2):get(5):forget()
  net.model:get(2):get(6):forget()

--  net.cache.grad_cnn_3 = net.model:get(2):get(4):backward(net.cache.cnn_3,net.cache.grad_cnn_4)
--  net.cache.grad_cnn_2 = net.model:get(2):get(3):backward(net.cache.cnn_2,net.cache.grad_cnn_3)
--  net.cache.grad_cnn_1 = net.model:get(2):get(2):backward(net.cache.cnn_1,net.cache.grad_cnn_2)
--  net.cache.grad_norm_input = net.model:get(2):get(1):backward(net.cache.norm_input,net.cache.grad_cnn_1)

--  local gradInput = net.model:get(1):backward(input:view(net.cache.num_samples*net.cache.feat_len,1366),net.cache.grad_norm_input:view(net.cache.num_samples*net.cache.feat_len,1366))
--  gradInput = gradInput:view(net.cache.num_samples,1,net.cache.feat_len,1366)
  
  net.called_forward = false
  return net.cache.grad_cnn_4

end


--print(net.cnn.modules)
return net

