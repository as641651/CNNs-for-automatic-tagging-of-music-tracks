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
   classifier.rnn.model = nn.Linear(1024,classifier.rnn.opt.classifier_vocab_size)
   classifier.rnn.model.weight:normal(0,1e-3)
   classifier.rnn.model.bias:fill(0)
   classifier.sigmoid = nn.Sigmoid()
   classifier.crit = nn.BCECriterion()   
   print(classifier.rnn.model)
   print(classifier.sigmoid)
end

function classifier.type(dtype)
   classifier.cnn.type(dtype)
   classifier.rnn.model:type(dtype)
   classifier.sigmoid:type(dtype)
   classifier.crit:type(dtype)
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
   classifier.cache.grad_input = classifier.cnn.backward(input,classifier.cache.grad_cnn_out)

   return loss
end


function classifier.clearState()
   classifier.cnn.model:clearState()
   classifier.rnn.model:clearState()
   classifier.sigmoid:clearState()
end


return classifier
