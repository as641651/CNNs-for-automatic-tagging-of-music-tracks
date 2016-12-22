require 'torch'
require 'DataLoader'
net = require 'cnns.choi_crnn.cnn'

local opts = require 'exp_opts'

local opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpu + 1) -- note +1 because lua is 1-indexed
end

-- initialize the data loader class
local loader = DataLoader(opt)

local clip_id
local input
local labels

clip_id,input,labels = loader:getBatch(opt)

print(clip_id, input:size())

output = net.cnn:forward(input[1]) 
print(output:size())

clip_id,input,labels = loader:getBatch(opt)

print(clip_id, input:size())

clip_id,input,labels = loader:getBatch(opt)

print(clip_id, input:size())
