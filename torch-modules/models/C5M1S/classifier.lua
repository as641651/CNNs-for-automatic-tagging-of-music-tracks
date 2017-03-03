require 'nn'
utils = require 'modules.utils'

local classifier = {}
local cnn = require 'cnns.cnn'
local rnn = require 'rnns.rnn_seq_to_one'
classifier.cache = {}
classifier.cnn = cnn
classifier.rnn = rnn

function classifier.setOpts(opt)
   assert(opt.group_batch or not opt.group, "should group batch or disable group for this classifier") 
   classifier.cnn.opt.model = opt.cnn_model --
   classifier.vocab_size = opt.classifier_vocab_size
   classifier.loader_info = opt.loader_info
   classifier.linear_hidden = opt.linear_hidden or 1024
end

function classifier.init()
   classifier.cnn.init_cnn()
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(classifier.linear_hidden,classifier.vocab_size))
   classifier.mlp = mlp
   classifier.mlp:get(1).weight:normal(0,1e-3)
   classifier.mlp:get(1).bias:fill(0)
   classifier.sigmoid = nn.Sigmoid()
   classifier.wts = torch.zeros(classifier.vocab_size)
   for i = 1,classifier.vocab_size do
     classifier.wts[i] = classifier.loader_info.vocab_weights[i]*classifier.vocab_size
   end
   classifier.crit = nn.BCECriterion(classifier.wts)
end

function classifier.type(dtype)
   classifier.cnn.type(dtype)
   classifier.mlp:type(dtype)
   classifier.sigmoid:type(dtype)
   classifier.crit:type(dtype)
end


function classifier.forward(input,add)
   --print("input: ", input:size())
   classifier.cache.cnn_output = classifier.cnn.forward(input)
   --print("cnn_out: ", classifier.cache.cnn_output:size())
   classifier.cache.linear_output = classifier.mlp:forward(classifier.cache.cnn_output)
   --print("linear_out: ", classifier.cache.linear_output:size())
   local sigmoid_out = classifier.sigmoid:forward(classifier.cache.linear_output)
   return sigmoid_out
end

function classifier.forward_backward(input,add,gt_seq)

--FORWARD
   classifier.cache.sigmoid_out = classifier.forward(input)
   
   local target = nil
   if type(gt_seq) == "table" then
     target = torch.zeros(utils.count_keys(gt_seq), classifier.vocab_size):type(gt_seq[1]:type())
     for k,v in pairs(gt_seq) do target[tonumber(k)] = utils.n_of_k(v,classifier.vocab_size)  end
   else
     target = utils.n_of_k(gt_seq,classifier.vocab_size)
   end

   local loss = classifier.crit:forward(classifier.cache.sigmoid_out,target) 
   
--BACKWARD
   classifier.cache.grad_sigmoid_out = classifier.crit:backward(classifier.cache.sigmoid_out,target)
   classifier.cache.grad_linear_out  = classifier.sigmoid:backward(classifier.cache.linear_output, classifier.cache.grad_sigmoid_out)
   classifier.cache.grad_cnn_out  = classifier.mlp:backward(classifier.cache.cnn_output, classifier.cache.grad_linear_out)
   classifier.cache.grad_input = classifier.cnn.backward(input,classifier.cache.grad_cnn_out)
   return loss
end

function classifier.forward_test(input,add)

   local output = {}
   local cnn_output = classifier.cnn.forward(input)
   local linear_output = classifier.mlp:forward(cnn_output)
   local sigmoid_out = classifier.sigmoid:forward(linear_output)
   sigmoid_out = sigmoid_out:view(-1)
   --sort the results and choose the top  10 results greater than certain thresh
   local Y, cls_label= torch.sort(sigmoid_out,1,true)
   if cls_label:numel() > 10 then cls_label = cls_label[{{1,10}}] end
   sigmoid_out = sigmoid_out:index(1,cls_label)
   for i = 1,sigmoid_out:size(1) do
     if sigmoid_out[i] > 0.2 then output[cls_label[i]] = sigmoid_out[i] end
   end
   return output
end

function classifier.clearState()
   classifier.cnn.getModel():clearState()
   classifier.mlp:clearState()
   classifier.sigmoid:clearState()
end

function classifier.training()
   classifier.cnn.getModel():training()
   classifier.mlp:training()
end

function classifier.evaluate()
   classifier.cnn.getModel():evaluate()
   classifier.mlp:evaluate()
end

function classifier.loadCNN(cnn)
  print("Loading checkpoint ..CNN ")
  classifier.cnn.setModel(cnn)
end

function classifier.loadRNN(rnn)
  print("no RNN to load  ")
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
