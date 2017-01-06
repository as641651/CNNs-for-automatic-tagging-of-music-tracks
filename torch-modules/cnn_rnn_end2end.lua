require 'modules.TemporalCrossEntropyCriterion'
require 'nn'

local classifier = {}
local cnn = require 'cnns.cnn'
local rnn = require 'rnns.rnn_layer'

classifier.cnn = cnn
classifier.rnn = rnn

function classifier.init()
   classifier.cnn.init_cnn()
   classifier.rnn.init_rnn()
   classifier.crit = nn.TemporalCrossEntropyCriterion()   
   --classifier.crit = nn.SequencerCriterion(nn.CrossEntropyCriterion())   
end

function classifier.type(dtype)
   classifier.cnn.type(dtype)
   classifier.rnn.type(dtype)
   classifier.crit:type(dtype)
end

function classifier.forward_backward(input,additional_seq, gt_seq)

--FORWARD
   local cnn_output = classifier.cnn.forward(input)
   local rnn_output = classifier.rnn.forward(cnn_output:view(cnn_output:size(1),-1), additional_seq,gt_seq)
   local loss = classifier.crit:forward(rnn_output:view(1,rnn_output:size(1),rnn_output:size(2)),classifier.rnn.target_tokens:view(1,classifier.rnn.target_tokens:size(1))) 
   --local loss = classifier.crit:forward(rnn_output,classifier.rnn.target_tokens) 

--BACKWARD
   local grad_rnn_output = classifier.crit:backward(rnn_output:view(1,rnn_output:size(1),rnn_output:size(2)),classifier.rnn.target_tokens:view(1,classifier.rnn.target_tokens:size(1))) 
   --local grad_rnn_output = classifier.crit:backward(rnn_output,classifier.rnn.target_tokens) 
   grad_rnn_output = grad_rnn_output:view(rnn_output:size(1),rnn_output:size(2))

   local grad_cnn_output = classifier.rnn.backward(cnn_output:view(cnn_output:size(1),-1),grad_rnn_output,1)
   grad_cnn_output = grad_cnn_output[1]:view(cnn_output:size())

   local grad_input = classifier.cnn.backward(input,grad_cnn_output)

   return loss
end

function classifier.forward(input,additional_seq)

   local cnn_output = classifier.cnn.forward(input)
   local labels = classifier.rnn.forward(cnn_output:view(cnn_output:size(1),-1), additional_seq)

   return labels
end

function classifier.clearState()
   classifier.rnn.model:clearState()
   classifier.cnn.model:clearState()
end

return classifier
