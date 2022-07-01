import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, method, hidden_size, coverage=False):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        self.hidden_h = nn.Linear(self.hidden_size, hidden_size, bias=False)
        self.hidden_s = nn.Linear(self.hidden_size, hidden_size)
        if coverage:
            self.hidden_c = nn.Linear(1, hidden_size,  bias=False)

        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.coverage = coverage

    def forward(self, last_hidden, encoder_outputs, encoder_mask=None, coverage=None):

        '''
        :param last_hidden:[tgt_len, B, H], default [1, B, H]
            previous hidden state or output of the decoder, in shape (1,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (src_len,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape [B, src_len]
        :param coverage:
            coverage_vector.
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = last_hidden.expand(max_len, -1, -1).transpose(1, 0)  # [B, src_len, H]
        attn_energies = self.score(H, encoder_outputs, coverage)  # 计算attentionScore

        if encoder_mask is not None:
            mask = (torch.ByteTensor(encoder_mask)).unsqueeze(1).cuda()  # [B, 1, T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)

        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

        # '''
        # :param last_hidden: size [B, H]
        # :param encoder_outputs: size [T,B,H]
        # :return: attention weight about encoder_outputs
        # 我的写法少了mask， 必须mask掉 要不然softmax回出问题
        # '''
        # last_hidden =last_hidden.unsqueeze(2)  # [B, H, 1]
        # attention = torch.bmm(encoder_outputs.transpose(0,1), last_hidden).squeeze(2) # [B, T, 1]
        # attention_score = F.softmax(attention, dim=1)
        # return attention_score.squeeze(1)  # [B, 1, T]
    def score(self, hidden, encoder_outputs, coverage=None):
        '''
        :param hidden:  [B, src_len, H]
        :param encoder_outputs: [src_len, B, H]
        :param coverage: [B,1, src_len]
        :return:
        '''
        src_len, batch_size, hidden_size = list(encoder_outputs.size())
        decode_energy = self.hidden_h(encoder_outputs.transpose(1, 0).contiguous().view(-1, self.hidden_size))

        decode_energy += self.hidden_s(hidden.contiguous().view(-1, self.hidden_size))
        energy = F.tanh(decode_energy)  # dim[B*T, H]
        if self.coverage:
            coverage = torch.zeros((batch_size, 1, src_len)).cuda() if coverage is None else coverage
            energy = F.tanh(decode_energy + self.hidden_c(coverage.transpose(2,1)).contiguous().view(-1, self.hidden_size))  # dim[B*T, H]
        attn = self.v(energy).contiguous().view(batch_size,src_len, 1) # [B,src_len, 1]
        return attn.transpose(2, 1).contiguous()

class RnnDecoder(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout=0.1):
        super(RnnDecoder, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.embedding = nn.Embedding(output_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.attn = Attn('concat', hidden_size)

        #这一时刻的输入input 维度与 上一时刻的encoder 的输出concatenate 再输入到rnn中
        self.rnn = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        pass
        # #得到last input word 的embedding
        # word_embbed = self.embedding(word_input).view(1, word_input.size(0), -1)  #(1, B, V)
        # word_embbed = self.drop(word_embbed)
        #
        # # 计算attention weight
        # attn_weight = self.attn(last_hidden[-1], encoder_outputs)
        # context = attn_weight.bmm(encoder_outputs.transpose(0,1))  #(B, 1, V)
        # context = context.transpose(0, 1)  # (1, B, V)
        # rnn_input = torch.cat((word_embbed, context), 2)  # (1, B, V) -> (B. V)
        #
        # output, hidden = self.rnn(rnn_input, last_hidden)
        # output = output.squeeze(0)
        # output = F.log_softmax(self.out(output))
        # return output, hidden





