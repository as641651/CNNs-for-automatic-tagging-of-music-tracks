require 'nn'
local utils = require 'modules.utils'

local net = {}
net.opt = {}
net.opt.cnn_out_dim = 1024
net.opt.rnn_hidden_size = 1024
net.opt.input_encoding_size = 1024
net.opt.rnn_layers = 2
net.opt.dropout = 0
net.called_forward = false
net.opt.rnn_model = 'rnns.models.lstm_model1'
net.opt.classifier_vocab_size = 50
net.cache = {}
local lookup_tabel = nil


function net.init_rnn()
  
  print("RNN OPTS : ")
  print(net.opt)
  net.model = nn.Sequential()

  lookup_tabel = nn.LookupTable(net.opt.classifier_vocab_size, net.opt.input_encoding_size)

  local rnn_model = require(net.opt.rnn_model)
 
  net.model:add(nn.Sequencer(rnn_model.get_rnn(net.opt)))
  net.model:get(1):remember('both')
  
end

function net.type(dtype)
   net.model:type(dtype)
   lookup_tabel:type(dtype)
end

function net.forward_test_cnn_vecs(cnn_vectors)
    net.model:get(1):forget()
    local nvec = nil
    for ci = 1,cnn_vectors:size(1) do
       nvec = net.model:get(1):get(1):forward(cnn_vectors[ci])
    end
    return nvec
end

function net.forward_test_vocab(vocab)
   local word = torch.Tensor(1):zero():type(net.model:type())
   word[1] = vocab
   local invec = lookup_tabel:forward(word)
   local outvec = net.model:get(1):get(1):forward(invec[1])
   return outvec
end

function net.forward(cnn_vectors,vocab_seq)
    net.model:get(1):forget()
    local input_tensor = cnn_vectors
      
    if vocab_seq ~= nil then
      if vocab_seq:size(1) > 1 then 
        local vocab_vectors = lookup_tabel:forward(vocab_seq[{{1,vocab_seq:size(1)-1}}])
        input_tensor = torch.cat(cnn_vectors,vocab_vectors,1)
      end
    end
    local net_input = utils.tensor_to_table(input_tensor)
    output = net.model:forward(net_input)
    net.called_forward = true
    net.cache.net_input = net_input
    return output
end

function net.backward(gradOutput,scale)
  assert(net.called_forward == true, "forward not called")
  assert(scale == nil or scale == 1.0)

  local gradInput = net.model:backward(net.cache.net_input, gradOutput,scale)
  net.called_forward = false

  return utils.table_to_tensor(gradInput)
end

function net.setModel(model)
   net.model = model
end

function net.getModel()
  return net.model
end

return net
