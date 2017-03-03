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
     record.gt = utils.tensor_to_table(targets:float())
     table.insert(records, record)
end

function toJsonTable(t)
  local l = {}
  for i,v in pairs(t) do table.insert(l,v) end
  return l
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
  local log_path = utils.getopt(kwargs, 'log_path', '')
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  assert(thresh <= 1 and thresh >= 0, "Threshold must be between 0 and 1")

  print('using split ', split)
  if max_samples <= 0 then max_samples = #loader.info.val_idxs end  
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
    loader:printSongInfo(data.clip_id)

    print("Predictions :")
    if type(data.gt_tags) ~= "table"  then
       for i = 1,data.gt_tags:size(1) do labels_in_test[data.gt_tags[i]] = data.gt_tags[i]  end

      -- Call forward_test to make predictions, and pass them to evaluator
       local label_prob = model.forward_test(data.input,data.info_tags)
       for k,v in utils.spairs(label_prob,function(t,a,b) return t[b] < t[a] end) do
          print(loader.info.idx_to_token[k],v)
       end
    
       for cls = 1,num_cls do
          addResult(data.clip_id,label_prob[cls],cls,data.gt_tags,evaluator[cls])
       end
      model.clearState()
    else
      for k,v in pairs(data.gt_tags) do
        print("Clip .. " .. tostring(k))
        for i = 1,v:size(1) do labels_in_test[v[i]] = v[i]  end
        local d_input = data.input[tonumber(k)]
        d_input = d_input:view(1,d_input:size(1),d_input:size(2),d_input:size(3))
        local label_prob = model.forward_test(d_input,data.info_tags)
        for k1,v1 in utils.spairs(label_prob,function(t,a,b) return t[b] < t[a] end) do
          print(loader.info.idx_to_token[k1],v1)
        end
        for cls = 1,num_cls do
          addResult(data.clip_id,label_prob[cls],cls,v,evaluator[cls])
        end
        model.clearState()
      end
    end
    -- Print a message to the console
    local msg = 'Processed sample %s (%d / %d) of split %s'
    local num_samples = max_samples --todo
    if max_samples > 0 then num_samples = math.min(num_samples, max_samples) end
    print(string.format(msg, data.clip_id, counter, num_samples, split))

    -- Break out if we have processed enough images
    if max_samples > 0 and counter >= max_samples then break end
  end
  labels_in_test = toJsonTable(labels_in_test)
  evaluator["labels_in_test"] = labels_in_test
  evaluator["info_json"] = loader.json_file
  evaluator["log_path"] = log_path
  utils.write_json("tmp.json",evaluator)
  os.execute('python eval.py')
  --os.remove("tmp.json")
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
