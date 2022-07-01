# -*- coding: UTF-8 -*-
import torch
from Encoder_Decoder.CopyRNN import CopyRNN
# from deep_keyphrase.dataloader import (TOKENS, TOKENS_LENS, TOKENS_OOV,
#                                        OOV_COUNT, OOV_LIST, EOS_WORD, UNK_WORD)
class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, **kwargs):
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty # 长度惩罚的指数系数
        self.num_beams = num_beams # beam size
        self.beams = [] # 存储最优序列及其累加的log_prob score
        self.worst_score = 1e9 # 将worst_score初始为无穷大。

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty # 计算惩罚后的score
        if len(self) < self.num_beams or score > self.worst_score:
            # 如果类没装满num_beams个序列
            # 或者装满以后，但是待加入序列的score值大于类中的最小值
            # 则将该序列更新进类中，并淘汰之前类中最差的序列
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                # 如果没满的话，仅更新worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        # 当解码到某一层后, 该层每个结点的分数表示从根节点到这里的log_prob之和
        # 此时取最高的log_prob, 如果此时候选序列的最高分都比类中最低分还要低的话
        # 那就没必要继续解码下去了。此时完成对该句子的解码，类中有num_beams个最优序列。
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
class BeamSearch(object):
    def __init__(self, model, num_beam, max_length, length_penalty, min_length=1, tokenizer=None,early_stopping=False,unk_token_id = 0, bos_token_id=1, eos_token_id=1,pad_token_id=0):
        self.model = model
        self.num_beam = num_beam

        self.min_length = min_length
        self.max_length = max_length

        self.length_penalty = length_penalty
        self.early_stop = early_stopping

        # tokenizer的特殊自如，bos，句子开始。eos句子结束，pad填充字符，unk不在vocab内的字符
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self.tokenizer=tokenizer
        if isinstance(model, CopyRNN):
            self.vocab_size = 50

        else:
            #TODO 加入bart tokenizer的vocab_size
            pass

    def beam_search(self, src_batch, repetition_penalty=1.0, length_penalty=None):
        device = None
        oov_lists = src_batch['oov_lists']
        encoder_tokens = src_batch['encoder_tokens'].cuda()
        encoder_token_length = src_batch['encoder_token_length']
        encoder_input_mask = src_batch['src_input_mask'].cuda()

        src_with_oov_ids = src_batch['encoder_token_with_oov'].cuda()
        oov_max_lengths = src_batch['batch_max_oov_length']

        decode_inputs = src_batch['decoder_input'].cuda()

        encode_state = self.model.prepare_for_decode(encoder_tokens, encoder_token_length, encoder_input_mask)

        batch_size = encode_state.encoder_output.size(1)

        # self.model.decoder(decode_inputs, encode_state)

        # 建立beam容器，每个样本一个
        generated_hyps = [
            BeamHypotheses(self.num_beam, self.max_length, self.length_penalty, early_stopping=self.early_stop)
            for _ in range(batch_size)
        ]

        # 每个beam容器的得分，共batch_size*num_beams个
        beam_scores = torch.zeros((batch_size, self.num_beam), dtype=torch.float, device=device)
        beam_scores = beam_scores.view(-1)

        # 每个样本是否完成生成，共batch_size个
        done = [False for _ in range(batch_size)]
        # 为了并行计算，一次生成batch_size*num_beams个序列
        # 第一步自动填入bos_token
        token_ids = torch.full(
            (batch_size * self.num_beam, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        # 当前长度设为1
        cur_len = 1
        #
        # # TODO 复制input_state 到 batch_size * num_beam 的程度
        # # 根据index来做顺序的调转
        indices = torch.arange(batch_size, dtype=torch.long).to(device)
        indices = indices.repeat_interleave(self.num_beam)
        encode_state.reorder_state(indices)
        #
        # best_index_mask = best_index >= self.vocab_size
        # result = best_index_mask.unsqueeze(2)  #[B, num_beams, 1]  最后一位用来cat
        #
        # next_tokens = best_index.masked_fill(best_index_mask, self.unk_token_id)  #[B, num_beams]
        # next_tokens=next_tokens.view(-1, 1)  #[B * num_beams, 1]

        # token_ids = torch.cat([decode_input_ids, next_tokens], dim=-1)

        while cur_len < self.max_length:
            # 将编码器得到的上下文向量和当前结果输入解码器
            output = self.model.decoder.decode(token_ids, encode_state)
            # 输出矩阵维度为：(batch*num_beams), vocab_size + oov
            #
            # # 取出最后一个时间步的各token概率，即当前条件概率
            # # (batch*num_beams)*vocab_size
            # scores = next_token_logits = output[:, -1, :]

            scores = torch.log(output)
            vocab_size = output.size(1)

            ###########################
            # 这里可以做一大堆操作减少重复 #
            ###########################
            if repetition_penalty != 1.0:
                token_scores = scores.gather(dim=1, index=token_ids)
                lt_zero_mask = token_scores.lt(0).float()
                ge_zero_mask = lt_zero_mask.eq(0).float()
                token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
                scores.scatter_(dim=1, index=token_ids, src=token_scores)

            # 计算序列条件概率的，因为取了log，所以直接相加即可
            # (batch_size * num_beams, vocab_size)
            next_scores = scores + beam_scores[:, None].expand_as(scores)

            # 为了提速，将结果重排
            next_scores = next_scores.view(
                batch_size, self.num_beam * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            # 取出分数最高的token（图中黑点）和其对应得分
            # sorted=True，保证返回序列是有序的
            #TODO 核函数
            _next_scores, _next_tokens = torch.topk(next_scores, 2 * self.num_beam, dim=1, largest=True, sorted=True)  # [batch_size , 2 * num_beam]

            # 下一个时间步整个batch的beam列表
            # 列表中的每一个元素都是三元组
            # (分数, token_id, beam_id)
            next_batch_beam = []

            # 对每一个样本进行扩展
            for batch_idx in range(batch_size):

                # 检查样本是否已经生成结束
                if done[batch_idx]:
                    # 对于已经结束的句子，待添加的是pad token
                    next_batch_beam.extend([(0, self.pad_token_id, 0)] * self.num_beam)  # pad the batch
                    continue

                # 当前样本下一个时间步的beam列表
                next_sent_beam = []

                # 对于还未结束的样本需要找到分数最高的num_beams个扩展
                # 注意，next_scores和next_tokens是对应的
                # 而且已经按照next_scores排好顺序
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(_next_tokens[batch_idx], _next_scores[batch_idx])
                ):
                    # get beam and word IDs
                    # 这两行可参考图中3进行理解
                    # beam_id = beam_token_id // self.vocab_size
                    # token_id = beam_token_id % self.vocab_size

                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * self.num_beam + beam_id

                    # 如果出现了EOS token说明已经生成了完整句子
                    if (self.eos_token_id is not None) and (token_id.item() == self.eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beam
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        # 往容器中添加这个序列
                        generated_hyps[batch_idx].add(
                            token_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted word if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # 扩展num_beams个就够了
                    if len(next_sent_beam) == self.num_beam:
                        break

                # 检查这个样本是否已经生成完了，有两种情况
                # 1. 已经记录过该样本结束
                # 2. 新的结果没有使结果改善
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len
                )

                # 把当前样本的结果添加到batch结果的后面
                next_batch_beam.extend(next_sent_beam)

            # 如果全部样本都已经生成结束便可以直接退出了
            if all(done):
                break

            # 把三元组列表再还原成三个独立列表
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = token_ids.new([x[1] for x in next_batch_beam])
            beam_idx = token_ids.new([x[2] for x in next_batch_beam])

            # 准备下一时刻的解码器输入
            # 取出实际被扩展的beam
            token_ids = token_ids[beam_idx, :]
            # copy 机制下讲token_ids 先超过vocab的置为unk_token_id
            token_next_id_mask = beam_tokens >= self.vocab_size
            beam_tokens_masked = beam_tokens.masked_fill(token_next_id_mask, self.unk_token_id)
            # 在这些beam后面接上新生成的token
            token_ids = torch.cat([token_ids, beam_tokens_masked.unsqueeze(1)], dim=-1)

            # 更新当前长度
            cur_len = cur_len + 1
            # end of length while


        # 将未结束的生成结果结束，并置入容器中
        for batch_idx in range(batch_size):
            # 已经结束的样本不需处理
            if done[batch_idx]:
                continue

            # 把结果加入到generated_hyps容器
            for beam_id in range(self.num_beam):
                effective_beam_id = batch_idx * self.num_beam + beam_id
                final_score = beam_scores[effective_beam_id].item()
                # TODO token_ids 改为copy之后的超过vocab_size的
                final_tokens = token_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # select the best hypotheses，最终输出
        # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = token_ids.new(output_batch_size)
        best = []

        # 对每个样本取出最好的output_num_return_sequences_per_batch个句子
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # 如果长短不一则pad句子，使得最后返回结果的长度一样
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
            # 先把输出矩阵填满PAD token
            decoded = token_ids.new(output_batch_size, sent_max_len).fill_(self.pad_token_id)

            # 填入真正的内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                # 填上eos token
                if sent_lengths[i] < self.max_length:
                    decoded[i, sent_lengths[i]] = self.eos_token_id
        else:
            # 所有生成序列都还没结束，直接堆叠即可
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        # 返回的结果包含BOS token
        return decoded


    def __idx2result_beam(self, delimiter, oov_list, result_sequences):
        results = []
        for batch_idx, batch in enumerate(result_sequences):
            beam_list = []
            item_oov_list = oov_list[batch_idx]
            for beam in batch:
                phrase = []
                for idx in beam:
                    if self.id2vocab.get(idx) == EOS_WORD:
                        break
                    if idx in self.id2vocab:
                        phrase.append(self.id2vocab[idx])
                    else:
                        oov_idx = idx - len(self.id2vocab)
                        if oov_idx < len(item_oov_list):
                            phrase.append(item_oov_list[oov_idx])
                        else:
                            phrase.append(UNK_WORD)

                if delimiter is not None:
                    phrase = delimiter.join(phrase)
                if phrase not in beam_list:
                    beam_list.append(phrase)
            results.append(beam_list)
        return results

    def expand_encoder_output(self, encoder_output_dict, batch_size):
        beam_batch_size = batch_size * self.beam_size
        encoder_output = encoder_output_dict['encoder_output']
        encoder_mask = encoder_output_dict['encoder_padding_mask']
        encoder_hidden_state = encoder_output_dict['encoder_hidden']
        max_len = encoder_output.size(-2)
        hidden_size = encoder_hidden_state[0].size(-1)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
        encoder_output = encoder_output.reshape(beam_batch_size, max_len, -1)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(1, self.beam_size, 1)
        encoder_mask = encoder_mask.reshape(beam_batch_size, -1)
        encoder_hidden_state0 = encoder_hidden_state[0].unsqueeze(2).repeat(1, 1, self.beam_size, 1)
        encoder_hidden_state0 = encoder_hidden_state0.reshape(-1, beam_batch_size, hidden_size)
        encoder_hidden_state1 = encoder_hidden_state[1].unsqueeze(2).repeat(1, 1, self.beam_size, 1)
        encoder_hidden_state1 = encoder_hidden_state1.reshape(-1, beam_batch_size, hidden_size)
        encoder_output_dict['encoder_output'] = encoder_output
        encoder_output_dict['encoder_padding_mask'] = encoder_mask
        encoder_output_dict['encoder_hidden'] = [encoder_hidden_state0, encoder_hidden_state1]
        return encoder_output_dict

    def greedy_search(self, src_dict, delimiter=None):
        """

        :param src_dict:
        :param delimiter:
        :return:
        """
        oov_list = src_dict[OOV_LIST]
        batch_size = len(src_dict[TOKENS])
        encoder_output_dict = None
        hidden_state = None
        prev_output_tokens = [[self.bos_idx]] * batch_size
        decoder_state = torch.zeros(batch_size, self.model.decoder.target_hidden_size)
        result_seqs = None

        for target_idx in range(self.max_target_len):
            model_output = self.model(src_dict=src_dict,
                                      prev_output_tokens=prev_output_tokens,
                                      encoder_output_dict=encoder_output_dict,
                                      prev_decoder_state=decoder_state,
                                      prev_hidden_state=hidden_state)
            decoder_prob, encoder_output_dict, decoder_state, hidden_state = model_output
            best_probs, best_indices = torch.topk(decoder_prob, 1, dim=1)
            if result_seqs is None:
                result_seqs = best_indices
            else:
                result_seqs = torch.cat([result_seqs, best_indices], dim=1)
            prev_output_tokens = result_seqs[:, -1].unsqueeze(1)
        result = self.__idx2result_greedy(delimiter, oov_list, result_seqs)

        return result

    def __idx2result_greedy(self, delimiter, oov_list, result_seqs):
        result = []
        for batch_idx, batch in enumerate(result_seqs.numpy().tolist()):
            item_oov_list = oov_list[batch_idx]
            phrase = []
            for idx in batch:
                if self.id2vocab.get(idx) == EOS_WORD:
                    break
                if idx in self.id2vocab:
                    phrase.append(self.id2vocab[idx])
                else:
                    oov_idx = idx - len(self.id2vocab)
                    if oov_idx < len(item_oov_list):
                        phrase.append(item_oov_list[oov_idx])
                    else:
                        phrase.append(UNK_WORD)
            if delimiter is not None:
                phrase = delimiter.join(phrase)
            result.append(phrase)
        return result
