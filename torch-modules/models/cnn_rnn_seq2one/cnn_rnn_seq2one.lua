require 'nn'
utils = require 'modules.utils'

local classifier = {}
local cnn = require 'cnns.cnn'
local rnn = require 'rnns.rnn_seq_to_one'
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
end

function classifier.init()
   classifier.cnn.init_cnn()
   classifier.rnn.init_rnn()
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(1024,1024))
   mlp:add(nn.Dropout(0.2))
   mlp:add(nn.Sigmoid())
   mlp:add(nn.Linear(1024,classifier.vocab_size))
   classifier.mlp = mlp
   classifier.mlp:get(4).weight:normal(0,1e-3)
   classifier.mlp:get(4).bias:fill(0)
   classifier.sigmoid = nn.Sigmoid()
   classifier.crit = nn.BCECriterion()   
   print(classifier.mlp)
   print(classifier.sigmoid)
end

function classifier.type(dtype)
   classifier.cnn.type(dtype)
   classifier.rnn.type(dtype)
   classifier.mlp:type(dtype)
   classifier.sigmoid:type(dtype)
   classifier.crit:type(dtype)
end

function classifier.forward(input,add)

   classifier.cache.cnn_output = classifier.cnn.forward(input)
   classifier.cache.rnn_output = classifier.rnn.forward(classifier.cache.cnn_output)
   classifier.cache.linear_output = classifier.mlp:forward(classifier.cache.rnn_output)
   local sigmoid_out = classifier.sigmoid:forward(classifier.cache.linear_output)

   return sigmoid_out
end

function classifier.forward_backward(input,add,gt_seq)

--FORWARD
   classifier.cache.sigmoid_out = classifier.forward(input)

   local target = utils.n_of_k(gt_seq,classifier.vocab_size)
   local loss = classifier.crit:forward(classifier.cache.sigmoid_out,target) 

--BACKWARD
   classifier.cache.grad_sigmoid_out = classifier.crit:backward(classifier.cache.sigmoid_out,target)
   classifier.cache.grad_linear_out  = classifier.sigmoid:backward(classifier.cache.linear_output, classifier.cache.grad_sigmoid_out)
   classifier.cache.grad_rnn_out  = classifier.mlp:backward(classifier.cache.rnn_output, classifier.cache.grad_linear_out)
   classifier.cache.grad_cnn_out = classifier.rnn.backward(classifier.cache.cnn_output,classifier.cache.grad_rnn_out)
   classifier.cache.grad_input = classifier.cnn.backward(input,classifier.cache.grad_cnn_out)

   return loss
end

function classifier.forward_test(input,add)

   local output = {}
   local cnn_output = classifier.cnn.forward(input)
   local rnn_output = classifier.rnn.forward(cnn_output)
   local linear_output = classifier.mlp:forward(rnn_output)
   local sigmoid_out = classifier.sigmoid:forward(linear_output)
--   for i = 1,sigmoid_out:size(1) do output[i] = 0 end
   --sort the results and choose the top  10 results greater than certain thresh
   sigmoid_out = sigmoid_out:view(-1)
   local Y, cls_label= torch.sort(sigmoid_out,1,true)
   if cls_label:numel() > 10 then cls_label = cls_label[{{1,10}}] end
   sigmoid_out = sigmoid_out:index(1,cls_label)   
   for i = 1,sigmoid_out:size(1) do
     if sigmoid_out[i] > 0.05 then output[cls_label[i]] = sigmoid_out[i] end
   end

   print(output)
   return output
end

function classifier.clearState()
   classifier.cnn.getModel():clearState()
   classifier.rnn.getModel():clearState()
   classifier.mlp:clearState()
   classifier.sigmoid:clearState()
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
  local l1 =  mlp:get(mlp:size()):parameters()[2]:size(1)
  local l2 =  mlp:get(mlp:size()):parameters()[1]:size(2)
  if classifier.vocab_size ~= l1 then
    print("WARNING :  size of last layer in MLP does not match with vocab size. Weights in last layer not transfered")
    local newMlp = nn.Sequential():type(mlp:type())
    for i = 1, (mlp:size()-1) do 
      newMlp:add(mlp:get(i))
    end
    local lastL = nn.Linear(l2,classifier.vocab_size):type(mlp:type())
    lastL.weight:normal(0,1e-3)
    lastL.bias:fill(0)
    newMlp:add(lastL)
    classifier.mlp = newMlp
  else
    classifier.mlp = mlp
  end
end

return classifier
