# # 这个函数将会返回一个不可使用的词表
# # 生成n-gram的巧妙方式大家可以借鉴一下
# # 下面是一个3-gram的例子
# a = [1,2,3,4,5]
# for ngram in zip(*[a[i:] for i in range(2)]):
#    print(ngram)
# import torch
#
# batch_size = 5
# parallel_paths = 3
# new_order = torch.arange(batch_size).view(-1, 1).repeat(1, parallel_paths).view(-1)
# new_order = new_order.long()
# print(new_order)
#
# topk 第一个返回值，第二个返回索引
# import torch
# output_log_prob = torch.randn(5,10)
# print(output_log_prob)
# best_log_prob, best_index = torch.topk(output_log_prob,k=5,dim=-1)
#
# print(best_log_prob.shape)
# print(best_index.shape)
# print(best_index)
#
## 生成batch_size * beam 的长度
# indices = torch.arange(batch_size, dtype=torch.long)
# indices = indices.repeat_interleave(5)
#
# decode_input_ids = torch.full(
#    (batch_size, 1),
#    0,
#    dtype=torch.long,
# )
#
# decode_input_ids = decode_input_ids.index_select(dim=0, index=indices)  # [batch_size * num_beams , length]
# print(decode_input_ids.shape)
#
# list_1 = [1,2,3]
# print(list_1)
#
# list_1 = sorted(list_1, reverse=True)
# print(list_1)
##
# causal_mask = torch.zeros(5, 5).fill_(float("-inf"))
# causal_mask = causal_mask.triu(diagonal=1)
# print(causal_mask)

# tokens之后的0全是padding，因为1是eos, 在pipe中规定的
import torch
list_ = [[2,3,1,0,0,0,0,0],
         [2,3,4,19,10,1,0,0],
         [2,3,4,19,24,5,123,1]]
batch_tokens = torch.tensor(list_).view(3,-1)
cumsum = batch_tokens.eq(1)
print(cumsum.shape)
cumsum = cumsum.flip(dims=[1])
print(cumsum.shape)
cumsum = cumsum.cumsum(dim=-1)
print(cumsum[:, -1:])
tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])
print(tgt_pad_mask)

static_kv: bool = False
print(static_kv)