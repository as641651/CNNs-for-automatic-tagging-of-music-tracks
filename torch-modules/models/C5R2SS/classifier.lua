require 'nn'
require 'modules.TemporalCrossEntropyCriterion'
utils = require 'modules.utils'

local classifier = {}
local cnn = require 'cnns.cnn'
local rnn = require 'rnns.rnn_seq_to_seq'
classifier.cache = {}
classifier.cnn = cnn
classifier.rnn = rnn

function classifier.setOpts(opt)
   classifier.cnn.opt.model = opt.cnn_model 
   classifier.rnn.opt.rnn_model = opt.rnn_model 
   classifier.rnn.opt.cnn_out_dim = opt.rnn_feature_input_dim 
   classifier.rnn.opt.input_encoding_size = opt.rnn_encoding_dim 
   classifier.rnn.opt.rnn_hidden_size = opt.rnn_hidden_dim 
   classifier.rnn.opt.rnn_layers = opt.rnn_num_layers 
   classifier.rnn.opt.dropout = opt.rnn_dropout 
   classifier.vocab_size = opt.classifier_vocab_size
   classifier.seq_len = opt.seq_len or 7
end

function classifier.init()
   classifier.cnn.init_cnn()
   classifier.rnn.init_rnn()
   classifier.mlp = nn.Sequencer(nn:Sequential():add(nn.Linear(classifier.rnn.opt.rnn_hidden_size,classifier.vocab_size)))
   classifier.mlp:remember('both')
   classifier.mlp:get(1):get(1):get(1).weight:normal(0,1e-3)
   classifier.mlp:get(1):get(1):get(1).bias:fill(0)
   classifier.crit = nn.TemporalCrossEntropyCriterion()
end

function classifier.type(dtype)
   classifier.cnn.type(dtype)
   classifier.rnn.type(dtype)
   classifier.mlp:type(dtype)
   classifier.crit:type(dtype)
end


function classifier.forward_backward(input,add,gt_seq)

--FORWARD
   local cnn_output = classifier.cnn.forward(input)
   local rnn_out = classifier.rnn.forward(cnn_output,gt_seq)
   local S = cnn_output:size(1)
   local E = S+gt_seq:size(1)-1
   classifier.cache.rnn_output = {}
   for i = S,E do
      table.insert(classifier.cache.rnn_output,rnn_out[i])
   end
   classifier.mlp:forget()
   classifier.cache.linear_out = classifier.mlp:forward(classifier.cache.rnn_output)
   local crit_in = utils.table_to_tensor(classifier.cache.linear_out)
   crit_in = crit_in:view(1,crit_in:size(1),crit_in:size(2))

   local loss = classifier.crit:forward(crit_in,gt_seq:view(1,gt_seq:size(1))) 

--BACKWARD
   classifier.cache.grad_linear_out = classifier.crit:backward(crit_in,gt_seq:view(1,gt_seq:size(1)))
   classifier.cache.grad_linear_out = utils.tensor_to_table(classifier.cache.grad_linear_out:view(crit_in:size(2),crit_in:size(3)))

   classifier.cache.grad_rnn_out  = classifier.mlp:backward(classifier.cache.rnn_output, classifier.cache.grad_linear_out)
   local grad_rnn = {}
   for i = 1, utils.count_keys(rnn_out) do
       if i < S then table.insert(grad_rnn,torch.zeros(rnn_out[i]:size(1)):type(rnn_out[i]:type())) end
       if i >= S then table.insert(grad_rnn, classifier.cache.grad_rnn_out[i-S+1]) end
   end
   classifier.cache.grad_cnn_out = classifier.rnn.backward(grad_rnn)
   classifier.cache.grad_cnn_out = classifier.cache.grad_cnn_out[{{1,S}}]
   classifier.cache.grad_input = classifier.cnn.backward(input,classifier.cache.grad_cnn_out)

   return loss
end

function classifier.forward_test(input,add)

   local output = {}
   local cnn_output = classifier.cnn.forward(input)
   local rnn_out = classifier.rnn.forward_test_cnn_vecs(cnn_output)
   classifier.mlp:forget()
   for i = 1,classifier.seq_len do
     local linear_out = classifier.mlp:get(1):forward(rnn_out)
     local sm = nn.SoftMax():type(linear_out:type()):forward(linear_out)
     local Y,cls = torch.max(sm,1)
     output[cls[1]] = Y[1]
     rnn_out = classifier.rnn.forward_test_vocab(cls[1])
   end
   return output
end

function classifier.clearState()
   classifier.cnn.getModel():clearState()
   classifier.rnn.getModel():clearState()
   classifier.mlp:clearState()
end

function classifier.training()
   classifier.cnn.getModel():training()
   classifier.rnn.getModel():training()
   classifier.mlp:training()
end

function classifier.evaluate()
   classifier.cnn.getModel():evaluate()
   classifier.rnn.getModel():evaluate()
   classifier.mlp:evaluate()
end

function classifier.loadCNN(cnn)
  print("Loading checkpoint ..CNN ")
  classifier.cnn.setModel(cnn)
end

function classifier.loadRNN(rnn)
  print("Loading checkpoint ..RNN ")
  classifier.rnn.setModel(rnn)
end

function classifier.loadMLP(mlp)
  print("Loading checkpoint ..MLP ")
  local l1 =  mlp:get(1):get(1):get(mlp:size()):parameters()[2]:size(1)
  local l2 =  mlp:get(1):get(1):get(mlp:size()):parameters()[1]:size(2)
  if classifier.vocab_size ~= l1 then
    print("WARNING :  size of last layer in MLP does not match with vocab size. Weights in last layer not transfered")
    local lastL = nn.Linear(l2,classifier.vocab_size):type(mlp:type())
    lastL.weight:normal(0,1e-3)
    lastL.bias:fill(0)
    mlp:get(1):get(1)["modules"][mlp:size()] = lastL
  else
    classifier.mlp = mlp
  end
end

return classifier
