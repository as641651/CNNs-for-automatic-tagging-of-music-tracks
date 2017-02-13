require 'cunn'
local classifier = require 'models.C5R2SS.classifier'
opt = {}
opt.cnn_model = "cnns.models.choi_cnn.choi_cnn" 
opt.rnn_model = "rnns.models.lstm_model1"
opt.rnn_feature_input_dim  = 1024
opt.rnn_encoding_dim = 1024
opt.rnn_hidden_dim = 1024
opt.rnn_num_layers = 2
opt.rnn_dropout = 0.1
opt.classifier_vocab_size = 50
opt.seq_len = 3

classifier.setOpts(opt)
classifier.init()
dtype = 'torch.CudaTensor'
classifier.type(dtype)
print(classifier.rnn.model)

local input = torch.randn(5,1,96,1366):type(dtype)
local t1 = classifier.forward_test(input,nil)
print(t1)

local gt = torch.zeros(3):type(dtype)
gt[1] = 23
gt[2] = 2
gt[3] =15

local t2 = classifier.forward_backward(input,nil,gt)
print(t2)

