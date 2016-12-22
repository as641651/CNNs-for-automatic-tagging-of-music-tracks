require 'nn'


function build_choi_cnn(model)

   local net = nn.Sequential()
   
   net:add(model["bn_0_freq"])

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
   net:add(nn.SpatialMaxPooling(4,4,4,4,0,0))
   net:add(nn.Dropout(0.1))

   return net
end

local net = {}
local model = torch.load('cnns/choi_crnn/choi_crnn.t7')
net.cnn = build_choi_cnn(model)

--print(net.cnn.modules)
return net

