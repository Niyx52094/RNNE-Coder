import torch
import torch.nn as nn
from Encoder_Decoder.RnnDecoder import Attn, RnnDecoder
from Encoder_Decoder.RnnEncoder import RnnEncoder
import torch.nn.functional as F
from state import GRUCopyState

class CopyAttn(Attn):
    def __init__(self, method, hidden_size, coverage=False):
        super(CopyAttn,self).__init__(method, hidden_size, coverage=coverage)


    def forward(self, last_hidden, encoder_outputs, encoder_mask=None, coverage=None):

        '''
        :param last_hidden:[tgt_len, B, H], default [1, B, H]
            previous hidden state or output of the decoder, in shape (1,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (src_len,B,H)
        :param encoder_mask:
            used for masking. NoneType or tensor in shape [B, src_len]
        :param coverage:
            coverage_vector.
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = last_hidden.repeat(max_len, 1, 1).transpose(1, 0)  # [B, src_len, H]
        attn = self.score(H, encoder_outputs, coverage)  # 计算attentionScore # [B, 1, Src_len]

        if encoder_mask is not None:
            # mask = (torch.ByteTensor(encoder_mask)).unsqueeze(1).cuda()  # [B, 1, T]
            attn = attn.masked_fill(encoder_mask.unsqueeze(1) == 0, -1e18)
        attn = F.softmax(attn, dim=-1)  #[B, 1, src_len]
        # context_vector = torch.einsum("blh, blt->bht", encoder_outputs.transpose(1,0), attn)  #[B, H, 1]
        context_vector = torch.bmm(attn, encoder_outputs.transpose(1, 0)).transpose(1, 0).contiguous() # [1,B,H]
        return context_vector, attn # normalize with softmax

class CopyRNN(nn.Module):

    def __init__(self, args):
        super(CopyRNN, self).__init__()
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.embed_size = args.embed_size
        self.max_src_len = args.max_src_len
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.is_copy = args.is_copy
        self.encoder = CopyRnnEncoder(args, self.vocab_size, self.hidden_size, self.embed_size, self.n_layers, self.dropout)
        self.encoder_to_decoder_init_state_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_to_decoder_init_state_2 = nn.Linear(self.hidden_size, self.hidden_size)
        embedding = self.encoder.embedding

        self.decoder = CopyRnnDecoder(args, self.hidden_size, self.embed_size, embedding, self.n_layers, self.dropout)

    # def encoder_mask_through_len(self, src_length, src_len):
    #     '''
    #     :param src_length: [B]
    #     :param src_len: max_src_len
    #     :return:
    #     '''
    #     batch_size = src_length.size(0)
    #     ids = torch.arange(0, src_len).unsqueeze(0).expand(batch_size, -1)
    #     mask = ids >= src_length.unsqueeze(1).expand(-1, src_len)
    #     return mask
    def prepare_for_decode_state(self, src_tokens, src_lengths, src_enc_mask, src_with_ood_ids, oov_max_lengths):

        encoder_ouputs, encoder_hidden_state = self.encoder(src_tokens, src_lengths)
        state = GRUCopyState(encoder_ouputs, src_enc_mask, src_with_ood_ids, encoder_hidden_state, oov_max_lengths)
        return state

    def forward(self, src_tokens, src_lengths, src_enc_mask, src_with_ood_ids, oov_max_lengths,decode_inputs):

        state = self.prepare_for_decode_state(src_tokens, src_lengths, src_enc_mask, src_with_ood_ids,oov_max_lengths)
        state.input_feed = self.encoder_to_decoder_init_state_1(state.input_feed.squeeze(0)).unsqueeze(0)
        state.hidden = self.encoder_to_decoder_init_state_2(state.hidden.squeeze(0)).unsqueeze(0)
        # # [Src_len, B, H]
        # encoder_ouputs, encoder_hidden_state = self.encoder(src_tokens, src_lengths)
        #
        # self.decoder.init_state(encoder_ouputs.size(1))
        # decoder_input, encoder_outputs, encoder_mask, extra_vocab_size, encoder_token_with_oov

        final_dist = self.decoder(decode_inputs, state)

        return final_dist

class CopyRnnEncoder(RnnEncoder):
    def __init__(self, args, vocab_size, hidden_size, embed_size, n_layers=1, dropout=0.2):
        super(CopyRnnEncoder, self).__init__(vocab_size, hidden_size, embed_size, n_layers=1, dropout=0.2)

        self.max_src_len = args.max_src_len
        self.hidden_trasform = nn.Linear(hidden_size * 2, hidden_size)

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
        embedding = self.embedding(input_seqs)  # [B, src_len, H]
        # 将已经padding填充好0的序列再回复成原样，否则回浪费GRU层的计算资源
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedding.transpose(1, 0), input_length, enforce_sorted=True)
        outputs, hidden = self.rnn(packed, hidden)
        # 经过gru层之后又需要变为badding填充后可以进行矩阵操作。
        outputs, output_length = torch.nn.utils.rnn.pad_packed_sequence(outputs,total_length = self.max_src_len)

        outputs = self.hidden_trasform(outputs)  # [L, B, H]
        hidden = torch.sum(hidden, dim=0, keepdim=True)
        return outputs, hidden  # hidden: [1, B, H]

class CopyRnnDecoder(nn.Module):

    def __init__(self, args,hidden_size, embed_size, embedings, n_layers=1, dropout=0.1, use_souce_input_feed = False):
        super(CopyRnnDecoder, self).__init__()

        # Define parameters
        self.coverage = args.coverage
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = args.vocab_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.is_copy = args.is_copy
        if args.share_embedding:
            self.embedding = embedings
        else:
            self.embedding = nn.Embedding(args.vocab_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.attn = CopyAttn('concat', hidden_size, args.coverage)

        # 这一时刻的输入input 维度与encoder 的输出concatenate 再输入到rnn中
        self.rnn = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size,  self.output_size)
        self.gen_layer = nn.Linear(self.hidden_size * 3 + self.embed_size, 1)

        self.use_original_input_feed = use_souce_input_feed

        if self.use_original_input_feed:
            self.linear_input_feed = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def decode(self, decoder_input,  extra_vocab_size, state):
        return self(decoder_input,  extra_vocab_size, state)[:, -1]


    def prepare_for_decode(self, ):
        pass

    def forward(self, decoder_input, state):
        '''
        :param decoder_input: [B, Tgt_len]
        what state includes:
            :param encoder_outputs: [Src_len, B, H]
            :param encoder_mask: [B, Src_len]
            :param extra_vocab_size:  max_extra_length, in a Batch, int
            :param encoder_token_with_oov:  [B, Src_len]
        :return:
        '''

        encoder_outputs = state.encoder_output
        encoder_mask = state.encoder_mask
        encoder_token_with_oov = state.encoder_token_with_oov

        encoder_max_len, batch_size1, hidden_size = list(encoder_outputs.size())
        batch_size2, decoder_input_max_len = list(decoder_input.size())
        assert batch_size2 == batch_size1, 'batch_size of encoder_ouptus and decoder_outpus is not equal!'

        coverage_vector = None   # [B, 1, src_len]

        if self.coverage:
            self.coverage_vectors = []
            self.attn_vectors = []


        decoder_input_embed = self.embedding(decoder_input) #[B, Tgt_len, Embed]

        decoder_input_embed = decoder_input_embed.transpose(1, 0)

        final_dist = []  #[B, Tgt_len, Vocab_size + batch_extra_size]
        final_attn_dist = []


        for time_embed in decoder_input_embed.split(1, dim=0):  # time_embed: [1, B, Embed]
            decoder_single_input = torch.cat([time_embed, state.input_feed], dim=2)  #[1, B, embed + hidden_size]

            # gru_output,[1, B, H], hidden_state:[1, B, H]
            gru_output, hidden_state = self.rnn(decoder_single_input, state.hidden)

            # next_context_vector:[1, B, H], time_attn: [B,1,Src_len]
            next_context_vector, time_attn = self.attn(gru_output, encoder_outputs,
                      encoder_mask,coverage_vector)  # Attn:[B, Tgt_len, Src_len], context_vector:[B, Tgt_len, Src_len, H]

            if self.use_original_input_feed:
                source_context_vector = torch.cat([gru_output,next_context_vector], dim=2).view(-1, self.hidden_size * 2)  # [1, B, 2H]
                next_context_vector = self.linear_input_feed(source_context_vector).view(-1, batch_size1, self.hidden_size)  # [1, B, H]

            if self.coverage:
                # coverage_loss = torch.sum(torch.min(time_attn, self.coverage_vector),-1).squeeze()
                coverage_vector = time_attn if coverage_vector is None else coverage_vector + time_attn
                self.coverage_vectors += [coverage_vector.squeeze(1)]
                self.attn_vectors += [time_attn.squeeze(1)] #[ B]

            state.input_feed = next_context_vector
            state.hidden = hidden_state
            # self.state = [hidden_state, next_context_vector]


            # get generator point
            gen = torch.cat((next_context_vector, decoder_single_input, gru_output), dim=-1)
            gen_point = torch.sigmoid(self.gen_layer(gen.view(-1, gen.size(2))))


            vocab_dist = self.out(self.drop(gru_output).view(-1, self.hidden_size))  # [B,Vocab_size]
            vocab_dist = F.softmax(vocab_dist, dim=-1)  # [B, Vocab_size]

            vocab_dist_ = vocab_dist * gen_point  # [B, vocab_size]
            attn_dist_ = time_attn.view(time_attn.size(0),time_attn.size(2)) * (1 - gen_point)  # [B, Src_len]

            final_dist.append(vocab_dist_)

            final_attn_dist.append(attn_dist_)



        final_dist = torch.stack(final_dist, dim=0).transpose(1, 0) # [ B,Tgt_len, Vocab_size]
        final_attn_dist = torch.stack(final_attn_dist, dim=0).transpose(1, 0)  # [ B,Tgt_len, Src_len]

        if self.is_copy:
            extra_zero = torch.zeros((batch_size1, decoder_input_max_len, extra_vocab_size)).cuda()
            final_dist = torch.cat((final_dist, extra_zero), dim=-1)

            encoder_token_with_oov = encoder_token_with_oov.unsqueeze(1).expand(-1, decoder_input_max_len,-1)  #[B, Tgt_len, Src_len]
            # generator_distribution
            final_dist.scatter_add_(2, encoder_token_with_oov, final_attn_dist)  # [B, Tgt_len, Vocab_size + batch_extra_size]
        # if self.coverage:
        #     coverage_loss = torch.stack(self.coverage_vectors_loss).transpose(1, 0)  # [B, Tgt_len]
        # else:
        #     coverage_loss = None

        return final_dist

