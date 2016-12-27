require 'torch'
require 'DataLoader'
cnn = require 'cnns.choi_crnn.cnn'
rnn = require 'rnns.densecap_lstm.my_rnn'


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
  cudnn.convert(cnn.cnn, cudnn)
  cudnn.convert(cnn.input_normalization, cudnn)
  cudnn.convert(rnn.rnn, cudnn)
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

labels = rnn.forward(output:view(output:size(1),-1))

print(labels)
