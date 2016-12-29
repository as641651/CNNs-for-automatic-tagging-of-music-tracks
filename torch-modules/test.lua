require 'torch'
require 'DataLoader'
local cnn = require 'cnns.cnn'
local rnn = require 'rnns.rnn_layer'

cnn.opt.model = 'cnns.models.choi_crnn.choi_cnn'
cnn.init_cnn()

rnn.opt.cnn_out_dim = 1664
rnn.opt.input_encoding_size = 512
rnn.opt.rnn_hidden_size = 200
rnn.opt.rnn_layers = 2
rnn.opt.dropout = 0.4
rnn.opt.seq_length = 5
rnn.opt.classifier_vocab_size = 5
rnn.opt.additional_vocab_size = 5
rnn.opt.rnn_model = 'rnns.models.lstm_model1'

local opts = require 'exp_opts'

local opt = opts.parse(arg)
print(opt)
rnn.init_rnn()
local dtype = 'torch.FloatTensor'

torch.setdefaulttensortype(dtype)
torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1) -- note +1 because lua is 1-indexed
  dtype = 'torch.CudaTensor'
  cudnn.convert(cnn.model, cudnn)
  cudnn.convert(rnn.model, cudnn)
end

cnn.type(dtype)
rnn.type(dtype)
-- initialize the data loader class
local loader = DataLoader(opt)

local clip_id
local input
local labels

local net_input = torch.zeros(3,1,96,1366)

clip_id,input1,labels = loader:getBatch(opt)
net_input[{1}] = input1[{1}]

clip_id,input2, labels = loader:getBatch(opt)
net_input[{2}] = input1[{1}]

clip_id,input3,labels = loader:getBatch(opt)
net_input[{3}] = input1[{1}]

print(clip_id, net_input:size())

output = cnn.forward(net_input:type(dtype)) 
--output = cnn.forward(input1:type(dtype)) 
print(output:size())

add_seq = torch.Tensor(2):type(dtype)
add_seq[1] = 1
add_seq[2] = 2
gt_seq = torch.Tensor(2):type(dtype)
gt_seq[1] = 3
gt_seq[2] = 4
labels = rnn.forward(output:view(output:size(1),-1), add_seq, gt_seq)

grad_labels = torch.randn(#labels):type(dtype)
print(labels:size())
grad_cnn = rnn.backward(output:view(output:size(1),-1),grad_labels,1)
grad_cnn = grad_cnn[1]:view(output:size())

print(grad_cnn:size())

gradInput = cnn.backward(net_input:type(dtype), grad_cnn)
print(gradInput:size())


