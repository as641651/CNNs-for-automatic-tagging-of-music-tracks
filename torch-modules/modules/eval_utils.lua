local utils = require 'modules.utils'
local roc = require 'modules.roc'

function addResult(sample_id,confidence,cls,targets,records)
     local record = {}
     if confidence ~= nil then 
       record.confidence = confidence
     else
       record.confidence = 0
     end

     if targets[targets:eq(cls)]:numel() > 0 then
        record.target = 1
     else
        record.target = 0
     end

     record.id = sample_id
     record.cls = cls
     table.insert(records, record)
end

local eval_utils = {}

function eval_utils.eval_split(kwargs)
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local split = utils.getopt(kwargs, 'split', 'val')
  local max_samples = utils.getopt(kwargs, 'max_samples', -1)
  local id = utils.getopt(kwargs, 'id', '')
  local thresh = utils.getopt(kwargs, 'thresh', 1)
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  local num_cls = utils.getopt(kwargs, 'vocab_size', 50)
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  assert(thresh <= 1 and thresh >= 0, "Threshold must be between 0 and 1")

  print('using split ', split)
  
  if split == 'val' then loader:val() else loader:test() end
  local split_to_int = {val=1, test=2}
  --loader:resetIterator()
  local evaluator = {}
  local labels_in_test = {} -- to exclude eval of lables not in test set
  for cls = 1,num_cls do
     evaluator[cls] = {}
  end

  local counter = 0
  while true do
    counter = counter + 1
    loader:val() 
    -- Grab a batch of data and convert it to the right dtype
    local data = {}

    data.clip_id,data.input,data.gt_tags,data.info_tags = loader:getSample()
    for i = 1,data.gt_tags:size(1) do labels_in_test[data.gt_tags[i]] = 1 end

    -- Call forward_test to make predictions, and pass them to evaluator
    local label_prob = model.forward_test(data.input,data.info_tags)

    for cls = 1,num_cls do
       addResult(data.clip_id,label_prob[cls],cls,data.gt_tags,evaluator[cls])
    end
    print("gt ", data.gt_tags)
 --   for cls = 1,data.gt_tags:size(1) do
 --      evaluator[data.gt_tags[cls]]:addResult(data.clip_id,1.0,data.gt_tags[cls],data.gt_tags)
 --   end

    model.clearState()
    
    -- Print a message to the console
    local msg = 'Processed sample %s (%d / %d) of split %s'
    local num_samples = max_samples --todo
    if max_samples > 0 then num_samples = math.min(num_samples, max_samples) end
    print(string.format(msg, data.clip_id, counter, num_samples, split))

    -- Break out if we have processed enough images
    if max_samples > 0 and counter >= max_samples then break end
  end
  print(labels_in_test)
  evaluator["labels_in_test"] = labels_in_test
  utils.write_json("tmp.json",evaluator)
  os.execute('python eval.py')
  local results = {}
  results.ap_results = {}
  results.auc_results = {}
--[[  for cls,v in pairs(labels_in_test) do 
     results.ap_results[cls], results.auc_results[cls] = unpack(evaluator[cls]:evaluate())
  end
   
  print(results.ap_results, results.auc_results)
  results.MAP = utils.average_values(results.ap_results)
  results.MAUC = utils.average_values(results.auc_results)
  print(string.format('MAP: %f', 100 * results.MAP))
  print(string.format('Mean AUC: %f', 100 * results.MAUC))
  --]]
  return results
end


return eval_utils
