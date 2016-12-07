local M = { }

function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a model.')
  cmd:text()
  cmd:text('Options')

  -- Core ConvNet settings
  cmd:option('-backend', 'cudnn', 'nn|cudnn')
  

  -- Data input settings
  cmd:option('-data_h5', '../databases/dd_new.h5', 
    'HDF5 file containing the preprocessed dataset (from proprocess.py)')
  cmd:option('-data_json', '../experiment.json',
    'JSON file containing additional dataset info (from preprocess.py)')
  cmd:option('-debug_max_train_images', -1,
    'Use this many training images (for debugging); -1 to use all images')

  -- Misc
  cmd:option('-id', '',
    'an id identifying this run/job; useful for cross-validation')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpu', -1, 'which gpu to use. -1 = use CPU')
  cmd:option('-timing', false, 'whether to time parts of the net')

  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M
