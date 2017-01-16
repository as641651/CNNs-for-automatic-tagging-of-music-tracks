require 'nn'

local net = {}
net.opt = {}
net.opt.cnn_out_dim = 1664
net.opt.input_encoding_size = 512
net.opt.classifier_vocab_size = 50
net.opt.additional_vocab_size = 0
net.opt.rnn_hidden_size = 512 
net.opt.rnn_layers = 2
net.opt.dropout = 0
net.called_forward = false
net.opt.rnn_model = 'rnns.models.lstm_model1'


function get_cnn_encoder(D,W)

   local cnn_encoder = nn.Sequential()
   cnn_encoder:add(nn.Linear(D,W))
   cnn_encoder:add(nn.ReLU(true))

   return cnn_encoder
end

function get_vocab_encoder(V,W)

   local lookup_table = nn.LookupTable(V,W)
   return lookup_table

end


-- net.rnn maps a table {cnn_vecs, add_seq} to word probabilities
function net.init_rnn()
  
  print("RNN OPTS : ")
  print(net.opt)
  net.model = nn.Sequential()
  local parallel = nn.ParallelTable()
  net.cnn_encoder = get_cnn_encoder(net.opt.cnn_out_dim, net.opt.input_encoding_size) 
  parallel:add(net.cnn_encoder)
  net.vocab_encoder = get_vocab_encoder(net.opt.additional_vocab_size+1, net.opt.input_encoding_size)
  parallel:add(net.vocab_encoder)

  net.model:add(parallel)
  net.model:add(nn.JoinTable(1, 2))

  local rnn_model = require(net.opt.rnn_model)

  net.rnn_model = nn.Sequential()
  net.rnn_model:add(nn.Sequencer(rnn_model.get_rnn(net.opt)))
--  net.rnn_model:add(nn.SelectTable(1)) -- selects the last time step
  net.rnn_model:add(nn.Linear(net.opt.rnn_hidden_size, net.opt.classifier_vocab_size))

  net.model:add(net.rnn_model)
  
  print("RNN MODEL : ")
  print(net.model)

end


function net.type(dtype)
   net.model:type(dtype)
end


function net.forward(cnn_vectors, add_sequence)
    
    C = cnn_vectors:size(1)
    L = 0
    local mask = nil
    if add_sequence ~= nil then
       add_sequence = add_sequence[add_sequence:gt(0)]
       if add_sequence:numel() > 0 then L = add_sequence:size(1) else add_sequence = torch.Tensor(1):type(cnn_vectors:type()):fill(1) end
    else
       L = 1
       add_sequence = torch.Tensor(1):type(cnn_vectors:type()):fill(1)
    end
    local net_input = {cnn_vectors,add_sequence}
    output = net.model:forward(net_input)
    net.called_forward = true
    return output[{{C+L}}]
end

function net.backward(cnn_vectors,add_sequence,gradOutput,scale)
  assert(net.called_forward == true, "forward not called")
  assert(scale == nil or scale == 1.0)

  local gradO = output.new(#output):zero()
  gradO[{{C+L}}]:copy(gradOutput)
  if add_sequence ~= nil then
     add_sequence = add_sequence[add_sequence:gt(0)]
  else
     add_sequence = torch.Tensor(1):type(cnn_vectors:type()):fill(1)
  end
  local net_input = {cnn_vectors,add_sequence}
  local gradInput = net.model:backward(net_input, gradO,scale)
  gradInput[2]:zero()
  net.called_forward = false

  return gradInput
end

function net.getParameters(dtype)
   local fakenet = nn.Sequential():type(dtype)
   fakenet:add(net.cnn_encoder)
   fakenet:add(net.rnn_model)
   return fakenet:getParameters()
end

function net.get_target(gt_seq)

    local target = torch.zeros(net.opt.classifier_vocab_size):type(gt_seq:type())
    for i = 1,gt_seq:size(1) do target[gt_seq[i]] = 1 end
    return target

end
return net
