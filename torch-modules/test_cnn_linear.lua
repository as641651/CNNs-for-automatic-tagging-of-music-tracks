require 'cutorch'
require 'cunn'

dtype = 'torch.CudaTensor'
classifier = require 'cnn_linear'
classifier.cnn.opt.model = 'cnns.models.choi_cnn.choi_cnn'
classifier.rnn.opt.classifier_vocab_size = 50
classifier.init()
classifier.type(dtype)

local input = torch.randn(3,1,96,1366):type(dtype)
local gt_seq = torch.Tensor(3,3):type(dtype)
gt_seq[1][1] = 43
gt_seq[1][2] = 3
gt_seq[1][3] = 4

gt_seq[2][1] = 35
gt_seq[2][2] = 13
gt_seq[2][3] = 5

gt_seq[3][1] = 35
gt_seq[3][2] = 3
gt_seq[3][3] = 5

print("gt_seq ", gt_seq)
print("input_shape ", input:size())

print("Forward check .. ")
local out = classifier.forward(input)
print(out)

print("backward check ..")
local loss = classifier.forward_backward(input,gt_seq)
print(loss)
--print(classifier.cache.grad_sigmoid_out)
--print(classifier.cache.grad_cnn_out)
