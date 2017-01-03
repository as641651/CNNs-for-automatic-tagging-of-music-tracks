utils = require 'modules.utils'

local eval_utils = {}

function eval_utils.eval_split(kwargs)
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local split = utils.getopt(kwargs, 'split', 'val')
  local max_samples = utils.getopt(kwargs, 'max_images', -1)
  local id = utils.getopt(kwargs, 'id', '')
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  local split_to_int = {val=1, test=2}
  split = split_to_int[split]
  print('using split ', split)
  
  if split == 'val' then loader:val() else loader:test() end
  loader:resetIterator()
  local evaluator = DenseCaptioningEvaluator{id=id}

  local counter = 0
  while true do
    counter = counter + 1
    
    -- Grab a batch of data and convert it to the right dtype
    local data = {}
    data.clip_id,data.input,data.gt_tags,data.info_tags = loader:getSample()

    -- Call forward_test to make predictions, and pass them to evaluator
    local label_prob = model.forward(data.input,data.info_tags)
    evaluator:addResult(label_prob, data.gt_tags)
    
    -- Print a message to the console
    local msg = 'Processed sample %s (%d / %d) of split %d'
    local num_samples = max_samples --todo
    if max_samples > 0 then num_samples = math.min(num_samples, max_samples) end
    print(string.format(msg, info.filename, counter, num_images, split))

    -- Break out if we have processed enough images
    if max_samples > 0 and counter >= max_samples then break end
  end

   
  local recall_results = unpack(evaluator:evaluate())
  print(string.format('Recall: %f', 100 * recall_results))
  
  local out = {
    recall_results=ap_results,
  }
  return out
end


local DenseCaptioningEvaluator = torch.class('DenseCaptioningEvaluator')
function DenseCaptioningEvaluator:__init(opt)
  self.all_scores = {}
  self.records = {}
  self.n = 1
  self.npos = 0
  self.id = utils.getopt(opt, 'id', '')
end

-- boxes is (B x 4) are xcycwh, scores are (B, ), target_boxes are (M x 4) also as xcycwh.
-- these can be both on CPU or on GPU (they will be shipped to CPU if not already so)
-- predict_text is length B list of strings, target_text is length M list of strings.
function DenseCaptioningEvaluator:addResult(scores, boxes, target_boxes, class)
  assert(scores:size(1) == boxes:size(1))
  assert(boxes:nDimension() == 2)

  -- convert both boxes to x1y1x2y2 coordinate systems
  boxes = box_utils.xcycwh_to_x1y1x2y2(boxes)
  if target_boxes == nil then
    target_boxes = boxes.new()
  else
    target_boxes = box_utils.xcycwh_to_x1y1x2y2(target_boxes)
  end

  -- make sure we're on CPU
  boxes = boxes:float()
  scores = scores:double() -- grab the positives class (1)
  target_boxes = target_boxes:float()

  -- merge ground truth boxes that overlap by >= 0.7
  --local mergeix = box_utils.merge_boxes(target_boxes, 0.7) -- merge groups of boxes together
  --local merged_boxes, merged_text = pluck_boxes(mergeix, target_boxes, target_text)
  --merged_boxes = target_boxes
  --merged_text = {}
  --for k,v in pairs(target_text) do table.insert(merged_text, {v}) end

  -- 1. Sort detections by decreasing confidence
  local Y,IX = torch.sort(scores,1,true) -- true makes order descending
  
  local nd = scores:size(1) -- number of detections
  local nt = 0
  if target_boxes:numel() ~= 0 then
    nt = target_boxes:size(1) -- number of gt boxes
  end
  local used = torch.zeros(nt)
  for d=1,nd do -- for each detection in descending order of confidence
    local ii = IX[d]
    local bb = boxes[ii]
    
    -- assign the box to its best match in true boxes
    local ovmax = 0
    local jmax = -1
    for j=1,nt do
      local bbgt = target_boxes[j]
      local bi = {math.max(bb[1],bbgt[1]), math.max(bb[2],bbgt[2]),
                  math.min(bb[3],bbgt[3]), math.min(bb[4],bbgt[4])}
      local iw = bi[3]-bi[1]+1
      local ih = bi[4]-bi[2]+1
      if iw>0 and ih>0 then
        -- compute overlap as area of intersection / area of union
        local ua = (bb[3]-bb[1]+1)*(bb[4]-bb[2]+1)+
                   (bbgt[3]-bbgt[1]+1)*(bbgt[4]-bbgt[2]+1)-iw*ih
        local ov = iw*ih/ua
        if ov > ovmax then
          ovmax = ov
          jmax = j
        end
      end
    end

    local ok = 1
    if jmax ~= -1 and used[jmax] == 0 then
      used[jmax] = 1 -- mark as taken
    else
      ok = 0
    end

    -- record the best box, the overlap, and the fact that we need to score the language match
    local record = {}
    record.ok = ok -- whether this prediction can be counted toward a true positive
    record.ov = ovmax
    if jmax ~= -1 then
      record.candidate = class
    end
    -- Replace nil with empty table to prevent crash in meteor bridge
    --if record.references == nil then record.references = {} end
    record.imgid = self.n
    table.insert(self.records, record)
  end
  
  -- keep track of results
  self.n = self.n + 1
  self.npos = self.npos + nt
  table.insert(self.all_scores, Y:double()) -- inserting the sorted scores as double
end

function DenseCaptioningEvaluator:evaluate(verbose)
  if verbose == nil then verbose = true end
  --local min_overlaps = {0.3, 0.4, 0.5, 0.6, 0.7}
  local min_overlaps = {0.5}

  -- concatenate everything across all images
  local scores = torch.cat(self.all_scores, 1) -- concat all scores
  -- call python to evaluate all records and get their BLEU/METEOR scores
  -- local blob = eval_utils.score_labels(self.records) -- replace in place (prev struct will be collected)
  -- local scores = blob.scores -- scores is a list of scores, parallel to records
  collectgarbage()
  collectgarbage()

  -- prints/debugging
--  print("records ", #self.records)
  if verbose then
    for k=1,#self.records do
      local record = self.records[k]
      if record.ov > 0 and record.ok == 1 and k % 1000 == 0 then
        print(string.format('IMG %d PRED: %s, OK:%f, OV: %f SCORE: %f',
              record.imgid, record.candidate, record.ok, record.ov, scores[k]))
      end  
    end
  end

  -- lets now do the evaluation
  local y,ix = torch.sort(scores,1,true) -- true makes order descending

  local ap_results = {}
  local recall_results = {}
  local pr_curves = {}
  for foo, min_overlap in pairs(min_overlaps) do
    -- go down the list and build tp,fp arrays
    local n = y:nElement()
    local tp = torch.zeros(n)
    local fp = torch.zeros(n)

    for i=1,n do
      -- pull up the relevant record
      local ii = ix[i]
      local r = self.records[ii]

      if not r.candidate then 
        fp[i] = 1 -- nothing aligned to this predicted box in the ground truth
      else
        -- ok something aligned. Lets check if it aligned enough, and correctly enough
        local score = scores[ii]
        if r.ov >= min_overlap and r.ok == 1 then
          tp[i] = 1
        else
          fp[i] = 1
        end
      end
    end

    fp = torch.cumsum(fp,1)
    tp = torch.cumsum(tp,1)
    local rec = torch.div(tp, self.npos)
    local prec = torch.cdiv(tp, fp + tp)

    -- compute max-interpolated average precision
    local ap = 0
    local apn = 0
    pr_curves['ov' .. min_overlap] = {}
    for t=0,1,0.01 do
      local mask = torch.ge(rec, t):double()
      local prec_masked = torch.cmul(prec:double(), mask)
      local p = torch.max(prec_masked)
      table.insert(pr_curves['ov' .. min_overlap], p)
      ap = ap + p
      apn = apn + 1
    end
    ap = ap / apn

    -- store it
    ap_results['ov' .. min_overlap] = ap
    recall_results['ov' .. min_overlap] = rec[rec:size()]
  end

  --local map = utils.average_values(ap_results)
  --local detmap = utils.average_values(det_results)

  -- lets get out of here
  local results = {ap_results, pr_curves, recall_results}
  return results
end

function DenseCaptioningEvaluator:numAdded()
  return self.n - 1
end

return eval_utils
