utils = require 'modules.utils'

local eval_utils = {}

function eval_utils.eval_split(kwargs)
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local split = utils.getopt(kwargs, 'split', 'val')
  local max_samples = utils.getopt(kwargs, 'max_samples', -1)
  local id = utils.getopt(kwargs, 'id', '')
  local thresh = utils.getopt(kwargs, 'thresh', 1)
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  assert(thresh <= 1 and thresh >= 0, "Threshold must be between 0 and 1")

  print('using split ', split)
  
  if split == 'val' then loader:val() else loader:test() end
  local split_to_int = {val=1, test=2}
  --loader:resetIterator()
  local evaluator = Evaluator{id=id}

  local counter = 0
  while true do
    counter = counter + 1
    loader:val() 
    -- Grab a batch of data and convert it to the right dtype
    local data = {}
    data.clip_id,data.input,data.gt_tags,data.info_tags = loader:getSample()

    -- Call forward_test to make predictions, and pass them to evaluator
    local label_prob = model.forward(data.input,data.info_tags)
    print(label_prob[1]:view(1,label_prob[1]:size(1)), label_prob[2]:view(1,label_prob[2]:size(1)),data.gt_tags:view(1,data.gt_tags:size(1))) --]]
    evaluator:addResult(data.clip_id,label_prob[1],label_prob[2], data.gt_tags)
    model.clearState()
    
    -- Print a message to the console
    local msg = 'Processed sample %s (%d / %d) of split %s'
    local num_samples = max_samples --todo
    if max_samples > 0 then num_samples = math.min(num_samples, max_samples) end
    print(string.format(msg, data.clip_id, counter, num_samples, split))

    -- Break out if we have processed enough images
    if max_samples > 0 and counter >= max_samples then break end
  end

   
  local results = evaluator:evaluate()
  print(string.format('Recall: %f', 100 * results[1]))
  print(string.format('Precission: %f', 100 * results[2]))
  
  return results
end


local Evaluator = torch.class('Evaluator')
function Evaluator:__init(opt)
  self.records = {}
  self.id = utils.getopt(opt, 'id', '')
  self.n = 0
end

function Evaluator:addResult(sample_id, labels, confidence, targets)
  assert(labels:size(1) == confidence:size(1))

    local record = {}
    record.labels = labels
    record.confidence = confidence
    record.targets = targets
    record.id = sample_id
    record.imgid = self.n
    table.insert(self.records, record)
  
    self.n = self.n + 1
end

function Evaluator:evaluate(verbose)
  if verbose == nil then verbose = true end
  
  --local min_threshs = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
  collectgarbage()

  -- lets now do the evaluation

  --local tpr = {}
  --local fpr = {}

  --for foo, th in pairs(min_threshs) do
  local tp = 0    
  local fp = 0    
  local fn = 0    
  for i=1,self.n do
    -- pull up the relevant record
    local r = self.records[i]
    for l = 1,r.labels:size(1) do
       if r.targets[r.targets:eq(r.labels[l])]:numel() > 0 then  tp = tp + 1 else fp = fp + 1 end
    end
    for l = 1,r.targets:size(1) do 
       if r.labels[r.labels:eq(r.targets[l])]:numel() == 0 then fn = fn + 1 end
    end
  end
    
  local rec = tp/(tp+fn)
  local prec = tp/(fp + tp)

  local results = {rec, prec}
  return results
end

function Evaluator:numAdded()
  return self.n - 1
end

return eval_utils
