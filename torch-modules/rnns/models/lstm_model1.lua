require 'nn'
require 'rnn'

local net = {}

function net.get_rnn(opt)

  local rnn = nn.Sequential()
  net.recurrent_hidden = nn.Sequential()
  net.recurrent_out = nn.Sequential()

  for i = 1, opt.rnn_layers do
    local input_dim = opt.rnn_hidden_size
    if i == 1 then
      input_dim = opt.input_encoding_size
    end
    net.recurrent_hidden:add(nn.LSTM(input_dim, opt.rnn_hidden_size))
    if opt.dropout > 0 then
      net.recurrent_hidden:add(nn.Dropout(opt.dropout))
    end
  end

  net.recurrent_out:add(nn.Linear(opt.rnn_hidden_size, opt.classifier_vocab_size+1))

  rnn:add(net.recurrent_hidden)
  rnn:add(net.recurrent_out)

  return rnn

end

return net

