import torch
import torch.nn as nn



class RnnEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size, n_layers=1, dropout=0.2):
        super(RnnEncoder,self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
    def forward(self, input_seqs, input_length, hidden=None):
        '''
        :param input_seqs:
            Variable of shape [B, Src_len], sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        embedding = self.embedding(input_seqs)
        # 将已经padding填充好0的序列再回复成原样，否则回浪费GRU层的计算资源
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedding, input_length)
        outputs, hidden =self.rnn(packed, hidden)
        # 经过gru层之后又需要变为badding填充后可以进行矩阵操作。
        outputs, output_length = torch.nn.utils.rnn. pad_packed_sequence(outputs)
        # 双向rnn之后输出的是2H维度，之后进行相加。
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

