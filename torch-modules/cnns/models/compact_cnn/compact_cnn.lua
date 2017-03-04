require 'nn'

local num_samples = nil
local feat_len = nil


function cnn_1(model)

   local net = nn.Sequential()
 
   net:add(model["conv1"])
   net:add(model["bn1"])
   local elu1 = nn.ELU()
   elu1.inplace = true
   net:add(elu1)
   net:add(nn.SpatialMaxPooling(4,2,4,2,0,0))
   net:add(nn.Dropout(0.1))

   return net
end

function cnn_2(model)

   local net = nn.Sequential()
 
   net:add(model["conv2"])
   net:add(model["bn2"])
   local elu2 = nn.ELU()
   elu2.inplace = true
   net:add(elu2)
   net:add(nn.SpatialMaxPooling(4,2,4,2,0,0))
   net:add(nn.Dropout(0.1))

   return net
end

function cnn_3(model)

   local net = nn.Sequential()
 
   net:add(model["conv3"])
   net:add(model["bn3"])
   local elu3 = nn.ELU()
   elu3.inplace = true
   net:add(elu3)
   net:add(nn.SpatialMaxPooling(4,2,4,2,0,0))
   net:add(nn.Dropout(0.1))

   return net
end

function cnn_4(model)

   local net = nn.Sequential()
 
   net:add(model["conv4"])
   net:add(model["bn4"])
   local elu4 = nn.ELU()
   elu4.inplace = true
   net:add(elu4)
   net:add(nn.SpatialMaxPooling(4,2,4,2,0,0))

   return net
end

function cnn_5(model)

   local net = nn.Sequential()
 
   net:add(model["conv5"])
   net:add(model["bn5"])
   local elu4 = nn.ELU()
   elu4.inplace = true
   net:add(elu4)
   net:add(nn.SpatialMaxPooling(2,2,2,2,0,0))

   return net
end

function build_choi_cnn(model)
   local net = nn.Sequential()
   net:add(cnn_1(model))
   net:add(cnn_2(model))
   net:add(cnn_3(model))
   net:add(cnn_4(model))
   net:add(cnn_5(model))
   return net
end


local net = {}
net.cache = {}
local model = torch.load('cnns/models/compact_cnn/weights/compact_cnn.t7')
--local model = torch.load('choi_cnn.t7')

net.cnn = build_choi_cnn(model)

net.model = nn.Sequential()
net.model:add(nn.Linear(1,1)) -- no normalization
net.model:add(net.cnn)
--print(net.model)

function net.type(dtype)
   net.cnn:type(dtype)
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

     net.cache.cnn_1 = net.model:get(2):get(1):forward(input)
--     print("cnn_1 size ", net.cache.cnn_1:size())
     net.cache.cnn_2 = net.model:get(2):get(2):forward(net.cache.cnn_1)
--     print("cnn_2 size ", net.cache.cnn_2:size())
     net.cache.cnn_3 = net.model:get(2):get(3):forward(net.cache.cnn_2)
--     print("cnn_3 size ", net.cache.cnn_3:size())
     net.cache.cnn_4 = net.model:get(2):get(4):forward(net.cache.cnn_3)
--     print("cnn_4 size ", net.cache.cnn_4:size())
     net.cache.cnn_5 = net.model:get(2):get(5):forward(net.cache.cnn_4)
--     print("cnn_5 size ", net.cache.cnn_5:size())

     local cnn_output = net.cache.cnn_5:view(net.cache.cnn_5:size(1),-1)
--     print(cnn_output)
--     os.exit()
     net.called_forward = true
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

  gradOutput = gradOutput:view(net.cache.cnn_5:size())
  net.cache.grad_cnn_4 = net.model:get(2):get(5):backward(net.cache.cnn_4,gradOutput)
  net.cache.grad_cnn_3 = net.model:get(2):get(4):backward(net.cache.cnn_3,net.cache.grad_cnn_4)
  net.cache.grad_cnn_2 = net.model:get(2):get(3):backward(net.cache.cnn_2,net.cache.grad_cnn_3)
  net.cache.grad_cnn_1 = net.model:get(2):get(2):backward(net.cache.cnn_1,net.cache.grad_cnn_2)
  gradInput = net.model:get(2):get(1):backward(input,net.cache.grad_cnn_1)
 
  net.called_forward = false
  return gradInput

end


--print(net.cnn.modules)
return net

