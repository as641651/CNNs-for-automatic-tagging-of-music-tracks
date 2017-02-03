require 'nn'
utils = require 'modules.utils'

local classifier = {}
local cnn = require 'cnns.cnn'
local rnn = require 'rnns.rnn_seq_to_one'
classifier.cache = {}
classifier.cnn = cnn
classifier.rnn = rnn

function classifier.setOpts(opt)
   classifier.cnn.opt.model = opt.cnn_model --
   classifier.rnn.opt.rnn_model = opt.rnn_model --
   classifier.rnn.opt.cnn_out_dim = opt.rnn_feature_input_dim --
   classifier.rnn.opt.input_encoding_size = opt.rnn_encoding_dim --
   classifier.rnn.opt.rnn_hidden_size = opt.rnn_hidden_dim --
   classifier.rnn.opt.rnn_layers = opt.rnn_num_layers --
   classifier.rnn.opt.dropout = opt.rnn_dropout --
   classifier.vocab_size = opt.classifier_vocab_size
   classifier.loader_info = opt.loader_info
   classifier.sigmoid_wt =  opt.sigmoid_wt --
   classifier.seq_wt = opt.seq_wt --
end

function classifier.init()
   classifier.cnn.init_cnn()
   classifier.rnn.init_rnn()
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(1024,classifier.vocab_size))
   classifier.mlp = mlp
   classifier.mlp:get(1).weight:normal(0,1e-3)
   classifier.mlp:get(1).bias:fill(0)
   classifier.sigmoid = nn.Sigmoid()
   classifier.crit = nn.BCECriterion()   
end

function classifier.type(dtype)
   classifier.cnn.type(dtype)
   classifier.rnn.type(dtype)
   classifier.mlp:type(dtype)
   classifier.sigmoid:type(dtype)
   classifier.crit:type(dtype)
end

function get_seq_prob(seq_l)
   seq_prob = torch.zeros(classifier.vocab_size)
   for i=1,classifier.vocab_size do
     local pw = 1./3.
     local u = classifier.loader_info.unigrams[i]/classifier.loader_info.num_instances
     local b = 0
     if classifier.loader_info.unigrams[seq_l[1]] ~= 0 then 
        local b1 = classifier.loader_info.bigrams[i][seq_l[1]]/classifier.loader_info.unigrams[seq_l[1]]
        local b2 = classifier.loader_info.bigrams[i][seq_l[2]]/classifier.loader_info.unigrams[seq_l[2]]
        b = 0.5*b1 + 0.5*b2
     end
     local t = 0
     if classifier.loader_info.bigrams[seq_l[1]][seq_l[2]] ~= 0 then
        t = classifier.loader_info.trigrams[i][seq_l[1]][seq_l[2]]/classifier.loader_info.bigrams[seq_l[1]][seq_l[2]]
     end
             
     seq_prob[i] = pw*u + pw*b + pw*t
   end

   return seq_prob
end

function smooth_with_seq(sigmoid_out)
   local Y, cls_label= torch.sort(sigmoid_out,1,true)
   cls_label = cls_label[{{1,2}}]
   local seq_prob = get_seq_prob(cls_label)

   sigmoid_out = sigmoid_out:view(-1)
   seq_prob = seq_prob:type(sigmoid_out:type())
   local smooth_out = sigmoid_out:mul(classifier.sigmoid_wt) + seq_prob:mul(classifier.seq_wt)
   return smooth_out
end

function classifier.forward(input,add)

   classifier.cache.cnn_output = classifier.cnn.forward(input)
   classifier.cache.rnn_output = classifier.rnn.forward(classifier.cache.cnn_output)
   classifier.cache.linear_output = classifier.mlp:forward(classifier.cache.rnn_output)
   local sigmoid_out = classifier.sigmoid:forward(classifier.cache.linear_output)
   local smooth_out = smooth_with_seq(sigmoid_out)
   return smooth_out
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
   local smooth_out = smooth_with_seq(sigmoid_out)

   --sort the results and choose the top  10 results greater than certain thresh
   local Y, cls_label= torch.sort(smooth_out,1,true)
   if cls_label:numel() > 10 then cls_label = cls_label[{{1,10}}] end
   smooth_out = smooth_out:index(1,cls_label)
   for i = 1,smooth_out:size(1) do
     if smooth_out[i] > 0.2 then output[cls_label[i]] = smooth_out[i] end
   end
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
