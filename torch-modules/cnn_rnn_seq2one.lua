require 'nn'
utils = require 'modules.utils'

local classifier = {}
local cnn = require 'cnns.cnn'
local rnn = require 'rnns.rnn_seq_to_one'
classifier.cache = {}
classifier.cnn = cnn
classifier.rnn = rnn

function classifier.init()
   classifier.cnn.init_cnn()
   classifier.rnn.init_rnn()
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(1024,1024))
   mlp:add(nn.Dropout(0.2))
   mlp:add(nn.Linear(1024,classifier.rnn.opt.classifier_vocab_size))
   classifier.mlp = mlp
   classifier.mlp:get(3).weight:normal(0,1e-3)
   classifier.mlp:get(3).bias:fill(0)
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

   local target = utils.n_of_k(gt_seq,classifier.rnn.opt.classifier_vocab_size)
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

function classifier.loadCheckpoint(checkpoint)
  print("Loading checkpoint .. ")
  classifier.cnn.setModel(checkpoint.cnn)
  classifier.rnn.setModel(checkpoint.rnn)
--  classifier.mlp:get(1).weight:copy(checkpoint.mlp:get(1).weight)
--  classifier.mlp:get(1).bias:copy(checkpoint.mlp:get(1).bias)
  classifier.mlp = checkpoint.mlp
end


return classifier
