require 'nn'

local classifier = {}
local cnn = require 'cnns.cnn'
local rnn = require 'rnns.rnn_seq_to_one'

classifier.cnn = cnn
classifier.rnn = rnn

function classifier.init()
   classifier.cnn.init_cnn()
   classifier.rnn.init_rnn()
   classifier.sigmoid = nn.Sigmoid()
   classifier.crit = nn.BCECriterion()   
end

function classifier.type(dtype)
   classifier.cnn.type(dtype)
   classifier.rnn.type(dtype)
   classifier.sigmoid:type(dtype)
   classifier.crit:type(dtype)
end

function classifier.forward(input,additional_seq)

   local cnn_output = classifier.cnn.forward(input)
   local rnn_output = classifier.rnn.forward(cnn_output:view(cnn_output:size(1),-1), additional_seq)
   local sigmoid_out = classifier.sigmoid:forward(rnn_output)

   return sigmoid_out
end

function classifier.forward_backward(input,additional_seq, gt_seq)

--FORWARD
   local cnn_output = classifier.cnn.forward(input)
   local rnn_output = classifier.rnn.forward(cnn_output:view(cnn_output:size(1),-1), additional_seq)
   local sigmoid_out = classifier.sigmoid:forward(rnn_output)
   
   local target = classifier.rnn.get_target(gt_seq)
   local loss = classifier.crit:forward(sigmoid_out,target) 

--BACKWARD
   local grad_sigmoid_out = classifier.crit:backward(sigmoid_out,target)
   local grad_rnn_output = classifier.sigmoid:backward(rnn_output, grad_sigmoid_out)
--   grad_rnn_output = grad_rnn_output:view(rnn_output:size(1),rnn_output:size(2))

   local grad_cnn_output = classifier.rnn.backward(cnn_output:view(cnn_output:size(1),-1),additional_seq,grad_rnn_output,1)
   grad_cnn_output = grad_cnn_output[1]:view(cnn_output:size())

   local grad_input = classifier.cnn.backward(input,grad_cnn_output)

   return loss
end


function classifier.clearState()
   classifier.rnn.model:clearState()
   classifier.cnn.model:clearState()
end

return classifier
