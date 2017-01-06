require 'nn'
require 'rnn'

local net = {}

function net.get_rnn(opt)

  local rnn = nn.Sequential()

  for i = 1, opt.rnn_layers do
    local input_dim = opt.rnn_hidden_size
    if i == 1 then
      input_dim = opt.input_encoding_size
    end
    rnn:add(nn.LSTM(input_dim, opt.rnn_hidden_size))
    if opt.dropout > 0 then
      rnn:add(nn.Dropout(opt.dropout))
    end
  end

  return rnn

end

return net

