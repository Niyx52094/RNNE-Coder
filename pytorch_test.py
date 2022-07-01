# 这个函数将会返回一个不可使用的词表
# 生成n-gram的巧妙方式大家可以借鉴一下
# 下面是一个3-gram的例子
a = [1,2,3,4,5]
for ngram in zip(*[a[i:] for i in range(2)]):
   print(ngram)
import torch

batch_size = 5
parallel_paths = 3
new_order = torch.arange(batch_size).view(-1, 1).repeat(1, parallel_paths).view(-1)
new_order = new_order.long()
print(new_order)

import torch
output_log_prob = torch.randn(5,10)
print(output_log_prob)
best_log_prob, best_index = torch.topk(output_log_prob,k=5,dim=-1)

print(best_log_prob.shape)
print(best_index.shape)
print(best_index)

indices = torch.arange(batch_size, dtype=torch.long)
indices = indices.repeat_interleave(5)

decode_input_ids = torch.full(
   (batch_size, 1),
   0,
   dtype=torch.long,
)

decode_input_ids = decode_input_ids.index_select(dim=0, index=indices)  # [batch_size * num_beams , length]
print(decode_input_ids.shape)

list_1 = [1,2,3]
print(list_1)

list_1 = sorted(list_1, reverse=True)
print(list_1)