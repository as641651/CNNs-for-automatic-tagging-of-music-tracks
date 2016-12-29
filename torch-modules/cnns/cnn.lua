
local net = {}
net.opt = {}
net.opt.model = 'cnns.models.choi_crnn.choi_cnn'
local cnn_model = nil

function net.init_cnn()

   print("CNN OPTS: ")
   print(net.opt)

   cnn_model = require(net.opt.model)
   net.model = cnn_model.model

   print("CNN MODEL :")
   print(net.model)

end

function net.type(dtype)
   cnn_model.type(dtype)
end
  
function net.forward(input)
   local output = cnn_model.forward(input)
   return output
end

function net.backward(input, gradOutput)
   local gradInput = cnn_model.backward(input,gradOutput)
   return gradInput
end

return net
