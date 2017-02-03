require 'torch'
require 'nn'
require 'rnn'
require 'cunn'
require 'cudnn'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Converts model to cpu compatable mode. Removes optimization states to reduce memory.')
cmd:text()
cmd:text('Options')

-- Run time opts
cmd:option('-m', '', 'model file path') 
cmd:option('-d', '', 'deploy model path') 
local opt = cmd:parse(arg or {})

if opt.m == '' then print("Please specifiy a model file"); os.exit() end
if opt.d == '' then print("Please specifiy a deploy file"); os.exit() end
print("Reading model " .. opt.m)
local model = torch.load(opt.m)
model.cnn:float()
model.rnn:float()
model.mlp:float()
model.cnn_optim_state = nil
model.rnn_optim_state = nil
model.mlp_optim_state = nil
torch.save(opt.d,model)
print("Wrote Deploy model at " .. opt.d)


  

