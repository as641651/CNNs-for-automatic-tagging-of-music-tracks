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
     evaluator[cls] = Evaluator{id=id}
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
       if label_prob[cls] ~= nil then evaluator[cls]:addResult(data.clip_id,label_prob[cls],cls,data.gt_tags) end
    end
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

  local results = {}
  results.ap_results = {}
  results.auc_results = {}
  for cls,v in pairs(labels_in_test) do 
     results.ap_results[cls], results.auc_results[cls] = unpack(evaluator[cls]:evaluate())
  end
   
  print(results.ap_results, results.auc_results)
  results.MAP = utils.average_values(results.ap_results)
  results.MAUC = utils.average_values(results.auc_results)
  print(string.format('MAP: %f', 100 * results.MAP))
  print(string.format('Mean AUC: %f', 100 * results.MAUC))
  
  return results
end


local Evaluator = torch.class('Evaluator')
function Evaluator:__init(opt)
  self.records = {}
  self.labels_in_test = {} -- we want to exclude the labels that not have not occured in gt in the test set
  self.id = utils.getopt(opt, 'id', '')
  self.n = 0
  self.num_labels = nil
  self.dtype = 'torch.FloatTensor'
end

function Evaluator:addResult(sample_id,confidence,cls,targets)

    local record = {}
    record.confidence = confidence
    record.targets = targets:type(self.dtype)
    record.id = sample_id
    record.evalid = self.n
    record.cls = cls
    table.insert(self.records, record)
 -- if self.num_labels == nil then self.num_labels = confidence:size(1) end
  
--    for i = 1,targets:size(1) do self.labels_in_test[targets[i]] = 1 end
    self.n = self.n + 1
end

function Evaluator:evaluate(verbose)
  if verbose == nil then verbose = true end

  if self.n == 0 then return {0,0} end
        
  local min_threshs = {10,20,30,40,50,60,70,80,90}
  collectgarbage()
  --self.num_labels = utils.count_keys(self.labels_in_test)
  --print("Total lables under test :", utils.count_keys(self.labels_in_test))
  --print("Evaluating :")

  local fpr = {}
  local prc = {}
  local tpr = {}
  -- lets now do the evaluation
  for foo, th in pairs(min_threshs) do
     local fpr_t = 0
     local prc_t = 0
     local tpr_t = 0
     local gt = false   
     local tp = 0    
     local fp = 0    
     local fn = 0 
     local tn = 0
     for i=1,self.n do
      -- pull up the relevant record
       local r = self.records[i]
       if  r.targets[r.targets:eq(r.cls)]:numel() > 0 then gt = true else gt = false end
       if r.confidence >= (th*0.01) then
         if gt then  tp = tp + 1 else fp = fp+1 end
       else
         if gt then fn = fn+1 else tn = tn+1 end
       end
     end
  	   
     if (tp+fn) ~= 0 then tpr_t = tp/(tp+fn) end 
     if (tp+fp) ~= 0 then prc_t = tp/(fp+tp) end
     if (fp+tn) ~= 0 then fpr_t = fp/(fp+tn) end
       -- if fp > 0 then  print(tp,fp,tn,fn) end
     
     tpr[th] = tpr_t
     prc[th] = prc_t
     fpr[th] = fpr_t   
  end
  
  local ap = 0
  local auc = 0
  for foo, th in pairs(min_threshs) do
    ap = ap + prc[th]*0.11112
    local prv_fpr = 1.0
    if th ~= 10 then prv_fpr = fpr[(th-10)] end
    local nxt_tpr = 0
    if th == 90 then nxt_tpr = tpr[th] else nxt_tpr = tpr[(th+10)] end
    auc = auc + ((prv_fpr - fpr[th])*(tpr[th] + nxt_tpr))/2.0
  end
   

  --print(ap)
  --print(auc)    
  
  return {ap,auc}
end

function Evaluator:numAdded()
  return self.n - 1
end

return eval_utils
