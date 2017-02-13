local rnn = require 'rnns.rnn_seq_to_seq'
local utils = require 'modules.utils'
rnn.init_rnn()
print(rnn.model)

local cnns = torch.randn(5,1024)
local out = rnn.forward_test_cnn_vecs(cnns)

for i = 1,5 do 
   local rout = rnn.forward_test_vocab(i)
   print(rout:size())
end

local out2 = rnn.forward(cnns)
print(out2)
local gt = torch.zeros(3)
gt[1] = 2
gt[2] = 32
gt[3] = 23

local out3 = rnn.forward(cnns,gt)
local grad3 = rnn.backward(out3)

print(out3)
