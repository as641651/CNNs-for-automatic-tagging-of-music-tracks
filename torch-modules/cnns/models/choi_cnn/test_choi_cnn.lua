require 'cutorch'
require 'cunn'

local net = require 'choi_cnn'
local dtype = 'torch.CudaTensor'
net.type(dtype)

local b = 3
local input = torch.randn(b,1,96,1366):type(dtype)
local net_out = net.forward(input)
local grad_net_out = torch.randn(b,1024):type(dtype)
local grad_input = net.backward(input,grad_net_out)

print(net.cache)

return net.model

