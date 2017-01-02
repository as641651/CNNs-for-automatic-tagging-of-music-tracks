local M = { }

function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Platform configs.')
  cmd:text()
  cmd:text('Options')

  -- Run time opts
  cmd:option('-c', '', 'config file') 
  cmd:option('-cudnn', 1, '-1 to not use cudnn. cudnn is not supported by fermi gpus') 
  cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU')
  cmd:option('-timing', false, 'whether to time parts of the net')

  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M
