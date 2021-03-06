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
net.opt.seq_length = 7
net.target_tokens = nil
net.called_forward = false
net.opt.rnn_model = 'rnns.models.lstm_model1'

local START_TOKEN = net.opt.classifier_vocab_size  + 1
local NULL_TOKEN = net.opt.classifier_vocab_size + 2
local label_tokens = nil

--local rnn_view_in = nn.View(1, 1, -1):setNumInputDims(3)
--local rnn_view_out = nn.View(1, -1):setNumInputDims(2)


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


-- net.rnn maps a table {cnn_vecs, gt_seq} to word probabilities
function net.init_rnn()
  
  print("RNN OPTS : ")
  print(net.opt)
  net.model = nn.Sequential()
  local parallel = nn.ParallelTable()
  net.cnn_encoder = get_cnn_encoder(net.opt.cnn_out_dim, net.opt.input_encoding_size) 
  parallel:add(net.cnn_encoder)
  parallel:add(net.start_token_gen)
  net.vocab_encoder = get_vocab_encoder(net.opt.classifier_vocab_size+2+net.opt.additional_vocab_size, net.opt.input_encoding_size)
  parallel:add(net.vocab_encoder)

  net.model:add(parallel)
  net.model:add(nn.JoinTable(1, 2))

  local rnn_model = require(net.opt.rnn_model)

  net.rnn_model = nn.Sequential()
  net.rnn_model:add(nn.Sequencer(rnn_model.get_rnn(net.opt)))
  net.rnn_model:add(nn.Linear(net.opt.rnn_hidden_size, net.opt.classifier_vocab_size+1))
  
  net.model:add(net.rnn_model)
  
  print("RNN MODEL : ")
  print(net.model)

  START_TOKEN = net.opt.classifier_vocab_size + net.opt.additional_vocab_size+ 1
  NULL_TOKEN = net.opt.classifier_vocab_size + net.opt.additional_vocab_size + 2

end

--for debugging
function forward_train(cnn_vectors,vocab_sequence)

   local encoded_cnn = net.cnn_encoder:forward(cnn_vectors)
   local encoded_vocab = net.vocab_encoder:forward(vocab_sequence)

   local join_table = nn.JoinTable(1,2):type(cnn_vectors:type())

   local rnn_input = join_table:forward{encoded_cnn, encoded_vocab}

   print(rnn_input:size())

   local lstm_out = net.recurrent_hidden:forward(rnn_input)

   local rnn_out = net.recurrent_out:forward(lstm_out)

   print(rnn_out:size())
   os.exit()
end

function net.type(dtype)
   net.model:type(dtype)
end

function sample(cnn_vectors, add_sequence)
  local C,T = cnn_vectors:size(1), net.opt.seq_length
  local L = 0
  if add_sequence ~= nil then 
    add_sequence = add_sequence[add_sequence:gt(0)]
    if add_sequence:numel()>0 then L = add_sequence:size(1) end
  end

  local seq = torch.Tensor(T+1):zero():type(cnn_vectors:type())
  local prob = torch.Tensor(T+1):zero():type(cnn_vectors:type())
  local word = torch.Tensor(1):zero():type(cnn_vectors:type())
  local softmax = nn.SoftMax():type(cnn_vectors:type())
  local step_scores = nil
  -- During sampling we want our LSTM modules to remember states
--[[  for i = 1, #net.rnn_model do
    local layer = net.rnn_model:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end--]]

  net.rnn_model:get(1):remember('both')


  -- First C+L timesteps: ignore output
  for ci = 1,C do
     local cnn_vecs_encoded = net.cnn_encoder:forward(cnn_vectors[ci])
     net.rnn_model:get(1):get(1):forward(cnn_vecs_encoded)
  end
  for li = 1,L do
     word[1] = add_sequence[li]
     local vocab_encoded = net.vocab_encoder:forward(word)
     net.rnn_model:get(1):get(1):forward(vocab_encoded)
  end

  -- Now feed words through RNN
  for t = 1, T do
    if t == 1 then
      -- On the first timestep, feed START tokens
      word[1] = START_TOKEN
    else
      -- On subsequent timesteps, feed previously sampled words
      word[1] = seq[t-1]
    end
    local wordvec = net.vocab_encoder:forward(word)
    local net_out = net.rnn_model:get(2):forward(net.rnn_model:get(1):get(1):forward(wordvec)):view(-1)
    local scores = nn.SoftMax():type(net_out:type()):forward(net_out)
    if t > 1 then 
       step_scores = torch.cat(step_scores,scores,2)
    else
       step_scores = scores:clone()
    end

    local idx = nil
    local p = nil
    p, idx = torch.max(scores,1)
    seq[{{t}}]:copy(idx)
    prob[{{t}}]:copy(p)
  end

  net.rnn_model:get(1):forget()
--[[  -- After sampling stop remembering states
  for i = 1, #net.rnn_model do
    local layer = net.rnn_model:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end
--]]
  --return {seq,prob}
  return step_scores:t()

end


function net.forward(cnn_vectors, add_sequence, gt_sequence)
    
  if gt_sequence ~= nil then
    -- Add a start token to the start of the gt_sequence, and replace
    -- 0 with NULL_TOKEN
    local T = gt_sequence:size(1)
    local L = 0
    local mask = nil
    if add_sequence ~= nil then
       add_sequence = add_sequence[add_sequence:gt(0)]
       L = add_sequence:numel()
    end
    local C = cnn_vectors:size(1)

    label_tokens = gt_sequence.new(L+T+2)
    if L>0 then label_tokens[{{1,L}}]:copy(add_sequence) end
    label_tokens[{{L+1,L+1}}]:fill(START_TOKEN)
    label_tokens[{{L+2,L+T+1}}]:copy(gt_sequence)
    label_tokens[{{L+T+2,L+T+2}}]:fill(START_TOKEN)
    --mask = torch.eq(label_tokens, 0)
    --label_tokens[mask] = NULL_TOKEN

    net.target_tokens = gt_sequence.new(C+L+T+2)
    net.target_tokens[C+L+1] = 0 -- START_TOKEN - net.opt.additional_vocab_size
    net.target_tokens[C+L+T+2] = 0 --START_TOKEN - net.opt.additional_vocab_size
    net.target_tokens[{{C+L+2,C+L+T+1}}]:copy(gt_sequence)
    --mask = torch.eq(net.target_tokens,0)
    --net.target_tokens[mask] = START_TOKEN - net.opt.additional_vocab_size
    net.target_tokens[{{1,C+L}}]:fill(0)
    --net.target_tokens = net.target_tokens[{{C+L+1,C+L+T+1}}]
    --print(label_tokens,net.target_tokens,gt_sequence)

    --rnn_view_in:resetSize(L+T+C+1,-1)
    --rnn_view_out:resetSize(1,C+L+T+1,-1)
    local net_input = {cnn_vectors,label_tokens}
    local output = net.model:forward(net_input)
    --forward_train(cnn_vectors, net.gt_tokens)
    net.called_forward = true
    return output
  else
    return sample(cnn_vectors,add_sequence)
  end
end

function net.backward(cnn_vectors,gradOutput,scale)
  assert(net.called_forward == true, "forward with gt not called")
  assert(scale == nil or scale == 1.0)

  local net_input = {cnn_vectors,label_tokens}
  local gradInput = net.model:backward(net_input, gradOutput,scale)
  gradInput[2]:zero()
  net.called_forward = false

  return gradInput
end

function net.getParameters(dtype)
   local fakenet = nn.Sequential():type(dtype)
   fakenet:add(net.model:get(1):get(1))
   fakenet:add(net.model:get(3))
   return fakenet:getParameters()
end

return net
