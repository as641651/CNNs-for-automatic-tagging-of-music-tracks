
local net = {}
net.opt = {}
net.opt.model = 'cnns.models.choi_crnn.choi_cnn'
local cnn_model = nil

function net.init_cnn()

   print("CNN OPTS: ")
   print(net.opt)

   cnn_model = require(net.opt.model)

   print("CNN MODEL :")
   print(cnn_model.model)

end

function net.type(dtype)
   cnn_model.model:type(dtype)
end
  
function net.forward(input)
   local output = cnn_model.forward(input)
   return output
end

function net.backward(input, gradOutput)
   local gradInput = cnn_model.backward(input,gradOutput)
   return gradInput
end

function net.setModel(model)
   cnn_model.model = model
--   net.model = model
end

function net.getModel()
  return cnn_model.model
end

return net
