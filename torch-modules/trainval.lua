require 'torch'
require 'modules.DataLoader'
require 'modules.optim_updates'
local utils = require 'modules.utils'
local platform_opts = require 'platform_opts'
local eval_utils = require 'modules.eval_utils'

--SETTINGS
local platform = platform_opts.parse(arg)

local checkpoint_start = nil
if platform.s ~= '' then
   require 'nn'
   require 'rnn'
   require 'cunn'
   require 'cudnn'
   print("Loading from check point " .. platform.s)
   checkpoint_start = torch.load(platform.s)
   print("CNN :")
   print(checkpoint_start.cnn)
   print("RNN :")
   print(checkpoint_start.rnn)
   print("MLP :")
   print(checkpoint_start.mlp)
   print("opts :")
   print(checkpoint_start.opt)
   print("Iters trained :")
   print(checkpoint_start.iter)
   print("LOSS : ")
   print(checkpoint_start.loss)
   os.exit()
end 
  
local opt

if platform.m ~= '' then 
   require 'nn'
   require 'rnn'
   require 'cunn'
   require 'cudnn'
   print("starting from check point " .. platform.m)
   checkpoint_start = torch.load(platform.m)
   if platform.c == '' then 
     opt = checkpoint_start.opt 
     opt.max_iters = 0
     opt.load_cnn_chpt = true
     opt.load_rnn_chpt = true
     opt.load_mlp_chpt = true
   else
     opt = utils.read_json(platform.c)
     opt.platform = platform
   end
else
  if platform.c == '' then 
     print("Please specify a config file")
     os.exit()
  end

  opt = utils.read_json(platform.c)
  opt.platform = platform
end

rnn_optim_state = {}
mlp_optim_state = {}
cnn_optim_state = {}

local iter = 1
if opt.platform.m ~= '' and not opt.fresh_optim and opt.max_iters > 0 then 
   mlp_optim_state = checkpoint_start.mlp_optim_state
   opt.mlp_optim = checkpoint_start.mlp_optim
   rnn_optim_state = checkpoint_start.rnn_optim_state
   opt.rnn_optim = checkpoint_start.rnn_optim
   cnn_optim_state = checkpoint_start.cnn_optim_state
   opt.cnn_optim = checkpoint_start.cnn_optim
   iter = checkpoint_start.iter
   print("Continuing Optimization ... ")
end

local classifier = require(opt.classifier)
opt.checkpoint_save_path = opt.platform.c .. ".t7"
if opt.platform.save ~= '' then
   opt.checkpoint_save_path = opt.platform.save
end

print("GENERAL OPTIONS : ")
print("Seed : " .. tostring(opt.seed))
opt.loader_info = nil
print(opt)

local loader = DataLoader(opt)
opt.classifier_vocab_size = loader:get_vocab_size()
opt.additional_vocab_size = loader:get_info_vocab_size()
opt.loader_info = loader.info

classifier.setOpts(opt)
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
    cudnn.convert(classifier.cnn.getModel(), cudnn)
    cudnn.convert(classifier.rnn.getModel(), cudnn)
  end
end

if checkpoint_start ~= nil then
  if opt.load_cnn_chpt then classifier.loadCNN(checkpoint_start.cnn) end
  if opt.load_rnn_chpt then classifier.loadRNN(checkpoint_start.rnn) end
  if opt.load_mlp_chpt then classifier.loadMLP(checkpoint_start.mlp) end
end

classifier.type(dtype)
loader:type(dtype)

if opt.fine_tune_cnn then
    cnn_params, cnn_grad_params = classifier.cnn.getModel():get(2):getParameters()
end
if opt.fine_tune_rnn then
    rnn_params, rnn_grad_params = classifier.rnn.getModel():getParameters()
end
local mlp_params, mlp_grad_params = classifier.mlp:getParameters()

print('total number of parameters in MLP: ', mlp_grad_params:nElement())
if opt.fine_tune_rnn then
   print('total number of parameters in RNN: ', rnn_grad_params:nElement())
end
if opt.fine_tune_cnn then
   print('total number of parameters in CNN: ', cnn_grad_params:nElement())
end

local function lossFun()
   mlp_grad_params:zero()
   if opt.fine_tune_rnn then rnn_grad_params:zero() end
   if opt.fine_tune_cnn then cnn_grad_params:zero() end
   
   local data = {}
   loader:train()
   data.sample_id, data.input,data.gt,data.info_tags = loader:getSample()
--   print("Loaded sample :" .. tostring(data.sample_id))
--   print(data.input:size())
--     print(data.gt)
   local loss = classifier.forward_backward(data.input,nil,data.gt)

   if opt.weight_decay > 0 then
     mlp_grad_params:add(opt.weight_decay, mlp_params)
     if opt.fine_tune_cnn then cnn_grad_params:add(opt.weight_decay, cnn_params) end
     if opt.fine_tune_rnn then rnn_grad_params:add(opt.weight_decay, rnn_params) end
   end

   return loss
end

local loss0
local loss = 0
local avgLoss = 0
while true do
   
  if opt.max_iters > 0 then 
      loss = lossFun()

      if iter%opt.avg_loss_every == 0 then
         avgLoss = avgLoss/opt.avg_loss_every
         print("iter " .. tostring(iter) .. " Loss : " .. tostring(avgLoss))
         avgLoss = 0
      else
         avgLoss = avgLoss + loss
      end
      
      if iter%opt.cnn_step == 0 then opt.cnn_learning_rate = opt.cnn_learning_rate*opt.cnn_gamma end
      if iter%opt.rnn_step == 0 then opt.rnn_learning_rate = opt.rnn_learning_rate*opt.rnn_gamma end
      if iter%opt.mlp_step == 0 then opt.mlp_learning_rate = opt.mlp_learning_rate*opt.mlp_gamma end
       
      if mlp_params:numel() > 0 then 
         if opt.mlp_optim == 'adam' then
            adam(mlp_params,mlp_grad_params,opt.mlp_learning_rate,opt.optim_alpha,opt.optim_beta,opt.optim_epsilon,mlp_optim_state) 
         elseif opt.mlp_optim == 'sgdm' then
            sgdm(mlp_params,mlp_grad_params,opt.mlp_learning_rate,opt.optim_alpha,mlp_optim_state)
         elseif opt.mlp_optim == 'sgdmom' then
            sgdmom(mlp_params,mlp_grad_params,opt.mlp_learning_rate,opt.optim_alpha,mlp_optim_state)
         else
            error('optim un available')
         end            
      end
      if opt.fine_tune_rnn then 
         if opt.rnn_optim == 'adam' then
            adam(rnn_params,rnn_grad_params,opt.rnn_learning_rate,opt.optim_alpha,opt.optim_beta,opt.optim_epsilon,rnn_optim_state) 
         elseif opt.rnn_optim == 'sgdm' then
            sgdm(rnn_params,rnn_grad_params,opt.rnn_learning_rate,opt.optim_alpha,rnn_optim_state)
         elseif opt.rnn_optim == 'sgdmom' then
            sgdmom(rnn_params,rnn_grad_params,opt.rnn_learning_rate,opt.optim_alpha,rnn_optim_state)
         else
            error('optim un available')
         end            
      end

      if opt.fine_tune_cnn then
         if opt.cnn_optim == 'adam' then
            adam(cnn_params,cnn_grad_params,opt.cnn_learning_rate,opt.optim_alpha,opt.optim_beta,opt.optim_epsilon,cnn_optim_state) 
         elseif opt.cnn_optim == 'sgdm' then
            sgdm(cnn_params,cnn_grad_params,opt.cnn_learning_rate,opt.optim_alpha,cnn_optim_state)
         elseif opt.cnn_optim == 'sgdmom' then
            sgdmom(cnn_params,cnn_grad_params,opt.cnn_learning_rate,opt.optim_alpha,cnn_optim_state)
         else
            error('optim un available')
         end
      end
  else
    print("Running evaluation ... ")
  end

  classifier.clearState()

  --periodic validation
  if (iter > 0 and iter % opt.save_checkpoint_every == 0) or (iter+1 == opt.max_iters) or (opt.max_iters == 0) or (iter == 1) then
     
     local eval_kwargs = {
      model=classifier,
      loader=loader,
      split='val',
      max_samples=opt.val_images_use,
      dtype=dtype,
      vocab_size = opt.classifier_vocab_size
      }
     local results = eval_utils.eval_split(eval_kwargs)
     local model = {}
     model.cnn = classifier.cnn.getModel()
     model.rnn = classifier.rnn.getModel()
     model.mlp = classifier.mlp

     model.cnn_optim_state = cnn_optim_state
     model.cnn_optim = opt.cnn_optim
     model.rnn_optim_state = rnn_optim_state
     model.rnn_optim = opt.rnn_optim
     model.mlp_optim_state = mlp_optim_state
     model.mlp_optim = opt.mlp_optim

     model.opt = opt

     model.iter = iter
     model.loss = loss
--     model.results = results --TODO

     if opt.max_iters > 0 and iter > 2 then
       torch.save(opt.checkpoint_save_path, model)
       print('wrote checkpoint ' .. opt.checkpoint_save_path)
     end
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
  if opt.max_iters == 0 then break end
end


