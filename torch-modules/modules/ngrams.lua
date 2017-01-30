local ngram = {}

function tableToTensor(label_table)
  local labels = torch.zeros(utils.count_keys(label_table))
  local idx = 1
  for k,v in pairs(label_table) do 
     labels[idx] = v
     idx = idx + 1
  end
  return labels
end

function ngram.compute_ngrams(gt,vocab_size)
   local dtype = "torch.FloatTensor"
   local u = torch.zeros(vocab_size):type(dtype)
   local b = torch.zeros(vocab_size,vocab_size):type(dtype)
   local t = torch.zeros(vocab_size,vocab_size,vocab_size):type(dtype)
   local num_instances = 0
   for _,v in pairs(gt) do 
      v = tableToTensor(v)
      num_instances = num_instances + v:size(1)
      v:add(1)
      for i=1,v:size(1) do
        local id1 = v[i]
        u[id1] =  u[id1] + 1.0
        for j = i,v:size(1) do 
          local id2 = v[j]
          if id2 ~= id1 then      
            b[id1][id2] =  b[id1][id2] + 1.0
            b[id2][id1] =  b[id2][id1] + 1.0
            for k = j,v:size(1) do
              local id3 = v[k]
              if id2 ~= id3 and id1~=id3 then
                t[id1][id2][id3] =  t[id1][id2][id3] + 1.0
                t[id1][id3][id2] =  t[id1][id3][id2] + 1.0
                t[id2][id1][id3] =  t[id2][id1][id3] + 1.0
                t[id2][id3][id1] =  t[id2][id3][id1] + 1.0
                t[id3][id1][id2] =  t[id3][id1][id2] + 1.0
                t[id3][id2][id1] =  t[id3][id2][id1] + 1.0
              end
            end
          end
        end
      end
   end
   return {u,b,t,num_instances}
end


return ngram
