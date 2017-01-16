require 'cutorch'
require 'cunn'

dtype = 'torch.CudaTensor'
local rnn = require 'rnns.rnn_seq_to_one'
rnn.init_rnn()
rnn.type(dtype)

local input = torch.randn(3,1024):type(dtype)
local out = rnn.forward(input)
print(out:size())
local grad_out = torch.randn(out:size(1)):type(dtype)
local grad_in = rnn.backward(input,grad_out)
print(grad_in)
local param, grad = rnn.model:getParameters()
--print(grad)
