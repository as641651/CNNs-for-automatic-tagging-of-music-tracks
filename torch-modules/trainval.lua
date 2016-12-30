require 'torch'
require 'modules.DataLoader'
local opts = require 'exp_opts'

--SETTINGS
local opt = opts.parse(arg)
opt.classifier = 'cnn_rnn_end2end'
local classifier = require(opt.classifier)

print("GENERAL OPTIONS : ")
print(opt)

classifier.cnn.opt.model = 'cnns.models.choi_crnn.choi_cnn'
classifier.rnn.opt.rnn_model = 'rnns.models.lstm_model1'
classifier.rnn.opt.cnn_out_dim = 1664
classifier.rnn.opt.input_encoding_size = 512
classifier.rnn.opt.rnn_hidden_size = 200
classifier.rnn.opt.rnn_layers = 2
classifier.rnn.opt.dropout = 0.4
classifier.rnn.opt.seq_length = 5

local loader = DataLoader(opt)
classifier.rnn.opt.classifier_vocab_size = 5 --to be found in dataloader
classifier.rnn.opt.additional_vocab_size = 5 --to be found in dataloader

classifier.init()

--SET DATATYPE AND GPU/CPU SETTINGS
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
  cudnn.convert(classifier.cnn.model, cudnn)
  cudnn.convert(classifier.rnn.model, cudnn)
end
classifier.type(dtype)

--DEBUG
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

add_seq = torch.Tensor(4):type(dtype)
add_seq[1] = 1
add_seq[2] = 2
add_seq[3] = 0
add_seq[4] = 0
gt_seq = torch.Tensor(4):type(dtype)
gt_seq[1] = 3
gt_seq[2] = 4
gt_seq[3] = 0
gt_seq[4] = 0


print("Test Check ")
local labels = classifier.forward(net_input:type(dtype),add_seq)
print(labels)

classifier.clearState()

print("Train Check ")
local loss = classifier.forward_backward(net_input:type(dtype),add_seq,gt_seq)
print("LOSS = ", loss)

classifier.clearState()

local net_input2 = torch.zeros(1,1,96,1366)

clip_id2,input12,labels = loader:getBatch(opt)
net_input2[{1}] = input1[{1}]

print("Train Check ")
local loss = classifier.forward_backward(net_input2:type(dtype),add_seq,gt_seq)
print("LOSS = ", loss)

classifier.clearState()
local add_s = torch.Tensor():type(dtype)
print("Train Check ")
local loss = classifier.forward_backward(net_input:type(dtype),add_s,gt_seq)
print("LOSS = ", loss)

classifier.clearState()
print("Test Check ")
local labels = classifier.forward(net_input2:type(dtype),add_s)
print(labels)
