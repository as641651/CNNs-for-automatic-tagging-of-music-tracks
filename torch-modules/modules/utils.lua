local json = require 'cjson'

local utils = {}


--[[
Utility function to set up GPU imports and pick datatype based on commmand line
arguments.

Inputs:
- gpu: Index of GPU requested on the command line; zero-indexed. gpu < 0 means
  CPU-only mode. If gpu >= 0 then we will import cutorch and cunn and set the
  current device using cutorch.
- use_cudnn: Whether cuDNN was requested on the command line; either 0 or 1.
  We will import cudnn if gpu >= 0 and use_cudnn == 1.

Returns:
- dtype: torch datatype that should be used based on the arguments
- actual_use_cudnn: Whether cuDNN should actually be used; this is equal to
  gpu >= 0 and use_cudnn == 1.
--]]
function utils.setup_gpus(gpu, use_cudnn)
  local dtype = 'torch.FloatTensor'
  local actual_use_cudnn = false
  if gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(gpu + 1)
    dtype = 'torch.CudaTensor'
    if use_cudnn == 1 then
      require 'cudnn'
      actual_use_cudnn = true
    end
  end
  return dtype, actual_use_cudnn
end


-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end


-- Check to make sure a required key is present
function utils.ensureopt(opt, key)
  if opt == nil or opt[key] == nil then
    error('error: required key ' .. key .. ' was not provided.')
  end
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read("*a")
  file:close()
  local info = json.decode(text)
  return info
end

function utils.write_json(path, j)
--  cjson.encode_sparse_array(true, 2, 10)
  local text = json.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end


-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end


-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end


function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end


function utils.n_of_k(gt_seq,k)
    local target = nil
    if gt_seq:size():size() == 2 then
      target = torch.zeros(gt_seq:size(1),k):type(gt_seq:type())
      for j = 1,gt_seq:size(1) do
        for i = 1,gt_seq:size(2) do target[j][gt_seq[j][i]] = 1 end
      end
    else
      target = torch.zeros(k):type(gt_seq:type())
      for i = 1,gt_seq:size(1) do target[gt_seq[i]] = 1 end
    end
    return target
end


function utils.apply_thresh(scores,thresh)
   local idx = scores:view(-1):gt(thresh)
  
   local ii = torch.LongTensor(idx:size(1)):zero()
   local count = 0
   for i=1,idx:size(1) do 
     count = count + 1
     if idx[i] ==1 then ii[i] = count end
   end
   ii = ii[ii:gt(0)]
   return ii
end

function utils.tensor_to_table(input)
  t = {}
  for i = 1,input:size(1) do table.insert(t,input:select(1,i)) end
  return t
end

function utils.table_to_tensor(t)
  local tensor = torch.zeros(utils.count_keys(t),t[1]:size(1)):type(t[1]:type())
  local idx = 1
  for k,v in pairs(t) do 
     tensor[idx] = v
     idx = idx + 1
  end
  return tensor
end

function utils.table_to_4Dtensor(t)
  local tensor = torch.zeros(utils.count_keys(t),t[1]:size(1),t[1]:size(2),t[1]:size(3)):type(t[1]:type())
  local idx = 1
  for k,v in pairs(t) do 
     tensor[idx] = v
     idx = idx + 1
  end
  return tensor
end

--iterates a table by sorting values in decending order
function utils.spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end
-- Stash global statistics here.
-- Since loading files with require caches and does not reload the same file,
-- all places that require 'utils' will have access to this table.
-- Loading with dofile will not cache, and will therefore give a fresh copy.
utils.__GLOBAL_STATS__ = {}

return utils
