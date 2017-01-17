require 'nn'
utils = require 'modules.utils'

local classifier = {}
local cnn = require 'cnns.cnn'
classifier.cache = {}
classifier.cnn = cnn
classifier.rnn = {} --should find a better way to code this.. 
classifier.rnn.opt = {}

function classifier.init()
   classifier.cnn.init_cnn()
   local rnn_model = nn.Sequential()
   rnn_model:add(nn.Linear(1024,1024))
   rnn_model:add( nn.Linear(1024,classifier.rnn.opt.classifier_vocab_size))
   classifier.rnn.model = rnn_model
   classifier.rnn.model:get(2).weight:normal(0,1e-3)
   classifier.rnn.model:get(2).bias:fill(0)
   classifier.sigmoid = nn.Sigmoid()
   classifier.crit = nn.BCECriterion()
   classifier.mlp = nn.Sequential() --dummy
   print(classifier.rnn.model)
   print(classifier.sigmoid)
end

function classifier.type(dtype)
   classifier.cnn.type(dtype)
   classifier.rnn.model:type(dtype)
   classifier.sigmoid:type(dtype)
   classifier.crit:type(dtype)
   classifier.mlp:type(dtype)
end

function classifier.forward(input,add)

   classifier.cache.cnn_output = classifier.cnn.forward(input)
   classifier.cache.linear_output = classifier.rnn.model:forward(classifier.cache.cnn_output)
   local sigmoid_out = classifier.sigmoid:forward(classifier.cache.linear_output)

   return sigmoid_out
end

function classifier.forward_backward(input,add,gt_seq)

--FORWARD
   classifier.cache.sigmoid_out = classifier.forward(input)

--   print(gt_seq)   
   local target = utils.n_of_k(gt_seq,classifier.rnn.opt.classifier_vocab_size)
--   print(target)
   local loss = classifier.crit:forward(classifier.cache.sigmoid_out,target) 

--BACKWARD
   classifier.cache.grad_sigmoid_out = classifier.crit:backward(classifier.cache.sigmoid_out,target)
   classifier.cache.grad_linear_out  = classifier.sigmoid:backward(classifier.cache.linear_output, classifier.cache.grad_sigmoid_out)
   classifier.cache.grad_cnn_out  = classifier.rnn.model:backward(classifier.cache.cnn_output, classifier.cache.grad_linear_out)
--   classifier.cache.grad_input = classifier.cnn.backward(input,classifier.cache.grad_cnn_out)

   return loss
end

function classifier.forward_test(input,add)

   assert(input:size(1) == 1, "one sample for forward test")

   local output = {}
   local cnn_output = classifier.cnn.forward(input)
   local linear_output = classifier.rnn.model:forward(cnn_output)
   local sigmoid_out = classifier.sigmoid:forward(linear_output)
--   for i = 1,sigmoid_out:size(1) do output[i] = 0 end

   --sort the results and choose the top  10 results greater than certain thresh
   sigmoid_out = sigmoid_out:view(-1)
   local Y, cls_label= torch.sort(sigmoid_out,1,true)
   if cls_label:numel() > 10 then cls_label = cls_label[{{1,10}}] end
   sigmoid_out = sigmoid_out:index(1,cls_label)   
   for i = 1,sigmoid_out:size(1) do
     if sigmoid_out[i] > 0.5 then output[cls_label[i]] = sigmoid_out[i] end
   end

   print(output)
   return output
end

function classifier.clearState()
   classifier.cnn.model:clearState()
   classifier.rnn.model:clearState()
   classifier.sigmoid:clearState()
end


return classifier
