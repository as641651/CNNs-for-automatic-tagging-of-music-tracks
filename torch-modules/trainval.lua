require 'torch'
require 'modules.DataLoader'
require 'modules.optim_updates'
local utils = require 'modules.utils'
local platform_opts = require 'platform_opts'
local eval_utils = require 'modules.eval_utils'

--SETTINGS
local platform = platform_opts.parse(arg)

if platform.c == '' then 
   print("Please specify a config file")
   os.exit()
end

local opt = utils.read_json(platform.c)
opt.platform = platform
opt.optim_state = {}
opt.cnn_optim_state = {}

local classifier = require(opt.classifier)

print("GENERAL OPTIONS : ")
print("Seed : " .. tostring(opt.seed))
print(opt)

classifier.cnn.opt.model = opt.cnn_model
classifier.rnn.opt.rnn_model = opt.rnn_model
classifier.rnn.opt.cnn_out_dim = opt.rnn_feature_input_dim
classifier.rnn.opt.input_encoding_size = opt.rnn_encoding_dim
classifier.rnn.opt.rnn_hidden_size = opt.rnn_hidden_dim
classifier.rnn.opt.rnn_layers = opt.rnn_num_layers
classifier.rnn.opt.dropout = opt.rnn_dropout
classifier.rnn.opt.seq_length = opt.rnn_test_time_seq_length

local loader = DataLoader(opt)
classifier.rnn.opt.classifier_vocab_size = loader:get_vocab_size()
classifier.rnn.opt.additional_vocab_size = loader:get_info_vocab_size()

classifier.init()

--SET DATATYPE AND GPU/CPU SETTINGS
local dtype = 'torch.FloatTensor'
torch.setdefaulttensortype(dtype)
torch.manualSeed(opt.seed)
if opt.platform.gpu >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.platform.gpu + 1) -- note +1 because lua is 1-indexed
  dtype = 'torch.CudaTensor'
  if opt.platform.cudnn >= 0 then
    cudnn.convert(classifier.cnn.model, cudnn)
    cudnn.convert(classifier.rnn.model, cudnn)
  end
end
classifier.type(dtype)
loader:type(dtype)

if opt.fine_tune_cnn then
    cnn_params, cnn_grad_params = classifier.cnn.model:get(2):getParameters()
end
local rnn_params, rnn_grad_params = classifier.rnn.getParameters(dtype)

print('total number of parameters in RNN: ', rnn_grad_params:nElement())
if opt.fine_tune_cnn then
   print('total number of parameters in CNN: ', cnn_grad_params:nElement())
end

local function lossFun()
   rnn_grad_params:zero()
   if opt.fine_tune_cnn then cnn_grad_params:zero() end
   
   local data = {}
   loader:train()
   data.sample_id, data.input,data.gt,data.info_tags = loader:getSample()
--   print("Loaded sample :" .. tostring(data.sample_id))
   local loss = classifier.forward_backward(data.input,nil,data.gt)


   if opt.weight_decay > 0 then
     rnn_grad_params:add(opt.weight_decay, rnn_params)
     if opt.fine_tune_cnn then
        if cnn_grad_params then cnn_grad_params:add(opt.weight_decay, cnn_params) end
     end
   end

   return loss
end

local loss0
local iter = 1
while true do
    
    local loss = lossFun()
    print("iter " .. tostring(iter) .. " Loss : " .. tostring(loss))

    if iter == 200 then opt.learning_rate = 1e-4 end
    if iter == 500 then opt.learning_rate = 1e-5 end
    
    adam(rnn_params,rnn_grad_params,opt.learning_rate,opt.optim_alpha,opt.optim_beta,opt.optim_epsilon,opt.optim_state)

    if opt.fine_tune_cnn then
      adam(cnn_params,cnn_grad_params,opt.learning_rate,opt.optim_alpha,opt.optim_beta,opt.optim_epsilon,opt.cnn_optim_state)
    end
   
  classifier.clearState()

  --periodic validation
  if (iter > 0 and iter % opt.save_checkpoint_every == 0) or (iter+1 == opt.max_iters) then
    --[[ loader:val()
     local clip_id,input1,gt_tags,info_tags = loader:getSample()
     local labels_prob = classifier.forward(input1,nil)
     print("val check for sample_id : " .. tostring(clip_id) )
     print(labels_prob[1]:view(1,labels_prob[1]:size(1)), labels_prob[2]:view(1,labels_prob[2]:size(1)),gt_tags:view(1,gt_tags:size(1))) --]]

     
    local eval_kwargs = {
      model=classifier,
      loader=loader,
      split='val',
      max_samples=210,
      dtype=dtype,
    }
    local results = eval_utils.eval_split(eval_kwargs)
   end

  -- stopping criterions
  iter = iter + 1
  -- Collect garbage every so often
  if iter % 33 == 0 then collectgarbage() end
  if loss0 == nil then loss0 = loss end
  if loss > loss0 * 100 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end

end

