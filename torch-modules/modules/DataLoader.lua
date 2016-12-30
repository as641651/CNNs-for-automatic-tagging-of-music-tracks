require 'hdf5'
local utils = require 'modules.utils'


local DataLoader = torch.class('DataLoader')


function DataLoader:__init(opt)
  self.h5_file = utils.getopt(opt, 'data_h5') -- required h5file with images and other (made with prepro script)
  self.json_file = utils.getopt(opt, 'data_json') -- required json file with vocab etc. (made with prepro script)

  self.debug_max_train_samples = utils.getopt(opt, 'debug_max_train_samples', -1)
  print('DataLoader loading json file: ', self.json_file)
  self.info = utils.read_json(self.json_file)
  self.vocab_size = utils.count_keys(self.info.idx_to_token)

  self.train_size = utils.count_keys(self.info.train_idxs)

  -- Convert keys in idx_to_token from string to integer
  local idx_to_token = {}
  for k, v in pairs(self.info.idx_to_token) do
    idx_to_token[tonumber(k)] = v
  end
  self.info.idx_to_token = idx_to_token

  self.iterators = {[0]=1,[1]=1,[2]=1} -- iterators (indices to split lists) for train/val/test
  print(string.format('assigned %d/%d/%d images to train/val/test.', #self.info.train_idxs, #self.info.val_idxs, #self.info.test_idxs))
  
  -- open the hdf5 file
  print('DataLoader loading h5 file: ', self.h5_file)
  self.h5_file = hdf5.open(self.h5_file, 'r')
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

