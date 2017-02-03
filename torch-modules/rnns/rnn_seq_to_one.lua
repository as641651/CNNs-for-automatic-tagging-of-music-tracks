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


function net.init_rnn()
  
  print("RNN OPTS : ")
  print(net.opt)
  net.model = nn.Sequential()

  local rnn_model = require(net.opt.rnn_model)
 
  net.rnn_model = nn.Sequential()
  net.rnn_model:add(nn.Sequencer(rnn_model.get_rnn(net.opt)))
  net.rnn_model:add(nn.SelectTable(-1)) -- selects the last time step
  net.model:add(net.rnn_model)
  
end


function net.type(dtype)
   net.model:type(dtype)
end


function net.forward(cnn_vectors)
    local net_input = utils.tensor_to_table(cnn_vectors)
    output = net.model:forward(net_input)
    net.called_forward = true
    return output
end

function net.backward(cnn_vectors,gradOutput,scale)
  assert(net.called_forward == true, "forward not called")
  assert(scale == nil or scale == 1.0)

  local net_input = utils.tensor_to_table(cnn_vectors)
  local gradInput = net.model:backward(net_input, gradOutput,scale)
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
