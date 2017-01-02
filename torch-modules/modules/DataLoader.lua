require 'hdf5'
local utils = require 'modules.utils'


local DataLoader = torch.class('DataLoader')


function DataLoader:__init(opt)
  self.h5_file = opt.data_h5 -- required h5file with images and other (made with prepro script)
  self.json_file = opt.split_info_path -- required json file with vocab etc. (made with prepro script)
  self.group = opt.group
  self.feature_xdim = opt.feature_xdim
  self.feature_ydim = opt.feature_ydim
  self.debug_max_train_samples = utils.getopt(opt, 'debug_max_train_samples', -1)

  print('DataLoader loading json file: ', self.json_file)
  self.info = utils.read_json(self.json_file)
  self.vocab_size = utils.count_keys(self.info.idx_to_token)
  self.info_vocab_size = utils.count_keys(self.info.info_idx_to_token)

  self.train_size = utils.count_keys(self.info.train_idxs)

  -- Convert keys in idx_to_token from string to integer
  local idx_to_token = {}
  for k, v in pairs(self.info.idx_to_token) do
    idx_to_token[tonumber(k)] = v
  end
  self.info.idx_to_token = idx_to_token
  
  local info_idx_to_token = {}
  for k, v in pairs(self.info.info_idx_to_token) do
    info_idx_to_token[tonumber(k)] = v
  end
  self.info.info_idx_to_token = info_idx_to_token

  self.iterators = {[0]=1,[1]=1,[2]=1} -- iterators (indices to split lists) for train/val/test
  print(string.format('assigned %d/%d/%d images to train/val/test.', #self.info.train_idxs, #self.info.val_idxs, #self.info.test_idxs))
  
  -- open the hdf5 file
  print('DataLoader loading h5 file: ', self.h5_file)
  self.h5_file = hdf5.open(self.h5_file, 'r')
  self.split = 0 --train
  self.dtype = 'torch.FloatTensor'
end

function DataLoader:type(dtype)
   self.dtype = dtype
end

function DataLoader:get_vocab_size()
   return self.vocab_size
end

function DataLoader:get_info_vocab_size()
   return self.info_vocab_size
end

function DataLoader:getBatch(opt)
  local split = utils.getopt(opt, 'split', 0)
  local iterate = utils.getopt(opt, 'iterate', true)

  assert(split == 0 or split == 1 or split == 2, 'split must be integer, either 0 (train), 1 (val) or 2 (test)')
  local split_ix
  if split == 0 then split_ix = self.info.train_idxs end
  if split == 1 then split_ix = self.info.val_idxs end
  if split == 2 then split_ix = self.info.test_idxs end
  assert(#split_ix > 0, 'split is empty?')
  
  -- pick an index of the datapoint to load next
  local ri -- ri is iterator position in local coordinate system of split_ix for this split
  local max_index = #split_ix
  if self.debug_max_train_samples > 0 then max_index = self.debug_max_train_samples end
  if iterate then
    ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1 end -- wrap back around
    self.iterators[split] = ri_next
  else
    -- pick an index randomly
    ri = torch.random(max_index)
  end
  ix = split_ix[ri]
  assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)

  local clip_id = ix

  local input = self.h5_file:read("/" .. tostring(clip_id)):all()
  
  local labels = self.info.gt[tostring(clip_id)]

  return clip_id,input:view(1,input:size(1),input:size(2),input:size(3)),labels
end

function DataLoader:train()   
  self.split = 0
end

function DataLoader:val()   
  self.split = 1
end

function DataLoader:test()   
  self.split = 2
end

--[[
function DataLoader:getBatch(batch_size)

  if not self.info.group then
    local input = torch.zeros(batch_size,1,self.info.feature_ydim,self.info.feature_xdim)
    for i = 1,batch_size do 
--]]

function DataLoader:getSample(iterate)

  if not iterate then iterate = true end

  local split_ix
  if self.split == 0 then split_ix = self.info.train_idxs end
  if self.split == 1 then split_ix = self.info.val_idxs end
  if self.split == 2 then split_ix = self.info.test_idxs end
  assert(#split_ix > 0, 'split is empty?')
  
  -- pick an index of the datapoint to load next
  local ri -- ri is iterator position in local coordinate system of split_ix for this split
  local max_index = #split_ix
  if self.debug_max_train_samples > 0 then max_index = self.debug_max_train_samples end
  if iterate then
    ri = self.iterators[self.split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1 end -- wrap back around
    self.iterators[self.split] = ri_next
  else
    -- pick an index randomly
    ri = torch.random(max_index)
  end
  ix = split_ix[ri]
  assert(ix ~= nil, 'bug: split ' .. self.split .. ' was accessed out of bounds with ' .. ri)

  if not self.group then
     local input, labels, info_tags = self:getClip(ix)
     return ix,input:view(1,input:size(1),input:size(2),input:size(3)):type(self.dtype),self:tableToTensor(labels):type(self.dtype),self:tableToTensor(info_tags):type(self.dtype)
  else
     local clips = self.info.song_clips[tostring(ix)]
     local num_clips = utils.count_keys(clips)
     local input = torch.zeros(num_clips,1,self.feature_ydim,self.feature_xdim)
     local labels_table = {}
     local info_tags_table = {}
     for i = 1,num_clips do 
        input[{i}],labels_table[i],info_tags_table[i] = self:getClip(clips[i])
     end
     local labels = self:tableToTensor(self:unionOfLabels(labels_table))
     local info_tags = self:tableToTensor(self:unionOfLabels(info_tags_table))
     return ix,input:type(self.dtype),labels:add(1):type(self.dtype),info_tags:add(1+self.vocab_size):type(self.dtype)
  end           
end

function DataLoader:unionOfLabels(labels) -- takes a table and returns a tensor
  local union = {}
  for k,l in pairs(labels) do 
      for k2,l2 in pairs(l) do union[l2] = 1 end
  end
  return union
end

function DataLoader:getClip(clip_id)
  local input = self.h5_file:read("/" .. tostring(clip_id)):all()
  local labels = self.info.gt[tostring(clip_id)]
  local info_tags = self.info.info_tags[tostring(clip_id)]

  return input,labels,info_tags
end

function DataLoader:tableToTensor(label_table)
  local labels = torch.zeros(utils.count_keys(label_table))
  local idx = 1
  for k,v in pairs(label_table) do 
     labels[idx] = k
     idx = idx + 1
  end
  return labels
end
