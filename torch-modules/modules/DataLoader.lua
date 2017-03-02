require 'hdf5'
local utils = require 'modules.utils'
local ngrams = require 'modules.ngrams'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(opt,debug)
  --self.h5_file = opt.data_h5 -- required h5file with images and other (made with prepro script)
  --self.json_file = opt.split_info_path -- required json file with vocab etc. (made with prepro script)
  self.group = opt.group
  self.feature_xdim = opt.feature_xdim
  self.feature_ydim = opt.feature_ydim
  self.max_clips_per_song = opt.max_clips_per_song
  self.debug_max_train_samples = utils.getopt(opt, 'debug_max_train_samples', -1)
 
  cmd = 'python ../lib/create_experiment.py -c ' .. opt.platform.c
  os.execute(cmd)

  comm = utils.read_json('../cache/tmp.json')
  self.json_file = tostring(comm.split_info_path)
  self.h5_file = tostring(comm.h5_file)
  self.additional_info_file = tostring(comm.additional_info_file)

  print('DataLoader loading json file: ', self.json_file)

  self.info = utils.read_json(self.json_file)
  self.debug_info = nil
  if debug then self.debug_info = utils.read_json(self.additional_info_file) end

  self.vocab_size = utils.count_keys(self.info.idx_to_token)
  self.info_vocab_size = utils.count_keys(self.info.info_idx_to_token)

  self.info.unigrams = nil
  self.info.bigrams = nil
  self.info.trigrams = nil
  self.info.num_instances = nil
  print("Computing ngrams ...")
  self.info.unigrams, self.info.bigrams, self.info.trigrams,self.info.num_instances = unpack(ngrams.compute_ngrams(self.info.gt,self.vocab_size))
  self.train_size = utils.count_keys(self.info.train_idxs)

  -- Convert keys in idx_to_token from string to integer
  local idx_to_token = {}
  for k, v in pairs(self.info.idx_to_token) do
    idx_to_token[tonumber(k)+1] = v
  end
  self.info.idx_to_token = idx_to_token
  
  local idx_to_wts = {}
  for k, v in pairs(self.info.vocab_weights) do
    idx_to_wts[tonumber(k)+1] = v
  end
  self.info.vocab_weights = idx_to_wts

  local info_idx_to_token = {}
  for k, v in pairs(self.info.info_idx_to_token) do
    info_idx_to_token[tonumber(k)+1] = v
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
     local input, labels = self:getClip(ix)
     input = self:splitInput(input,self.feature_xdim)
     local song_id = self.info.clips_song[tostring(ix)]
     local info_tags = self.info.info_tags[tostring(song_id)]
     return ix,input:type(self.dtype),self:tableToTensor(labels):add(1):type(self.dtype),self:tableToTensor(info_tags):add(1+self.vocab_size):type(self.dtype)
  else
     local clips = self.info.song_clips[tostring(ix)]
     local num_clips = math.min(utils.count_keys(clips),self.max_clips_per_song)
     
     local input = torch.zeros(num_clips,1,self.feature_ydim,self.feature_xdim)
     local labels_table = {}

     for i = 1,num_clips do 
       input1 ,labels_table[i]= self:getClip(clips[i])
       if input1:size(3) < self.feature_xdim then
          local tmp = input1:clone()
          input1 = torch.zeros(1,tmp:size(2),self.feature_xdim):type(tmp:type())
          input1[{{},{},{1,tmp:size(3)}}] = tmp 
       end
       if input1:size(3) > self.feature_xdim then
          input1 = input1[{{},{},{1,self.feature_xdim}}]:contiguous()
       end--]]
       input[{i}] = input1
     end
     local labels = self:tableKeyToTensor(self:unionOfLabels(labels_table))
     local info_tags = self:tableToTensor(self.info.info_tags[tostring(ix)])
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
  return input,labels
end

function DataLoader:tableToTensor(label_table)
  local labels = torch.zeros(utils.count_keys(label_table))
  local idx = 1
  for k,v in pairs(label_table) do 
     labels[idx] = v
     idx = idx + 1
  end
  return labels
end

function DataLoader:tableKeyToTensor(label_table)
  local labels = torch.zeros(utils.count_keys(label_table))
  local idx = 1
  for k,v in pairs(label_table) do 
     labels[idx] = k
     idx = idx + 1
  end
  return labels
end

function DataLoader:splitInput(input,offset)
   if input:size(3) > (self.feature_xdim + offset) then 
        local tmp = {}
        local count = 1
        local st = 1
        if input:size(3) > self.feature_xdim*8 and self.split == 0 then st = math.random(1,6) end
        offset = st*offset
        while offset + self.feature_xdim < input:size(3) do
           tmp[count] = input:narrow(3,offset,self.feature_xdim)
           offset = offset + self.feature_xdim
           count = count + 1
           if count > self.max_clips_per_song then break end
        end
        return utils.table_to_4Dtensor(tmp)
   else
        return input:view(1,input:size(1),input:size(2),input:size(3))
   end
end 

function DataLoader:printSongInfo(id)
   if self.debug_info then
      print(id)
      local song_name = self.debug_info.song_id_to_name[tostring(id)]
      print(self.debug_info[song_name])
   end
end
      
