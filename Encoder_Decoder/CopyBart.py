from modeing_bart import BartModel,BartEncoder, BartDecoder
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

def seq_len_to_mask(seq_len, max_len=None):
    """
    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.
    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

class CopyBartEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(CopyBartEncoder).__init__()
        assert isinstance(encoder, BartEncoder), "the encoder type is not BartEncoder!"
        self.model = encoder

    def forward(self, src_tokens, src_lengths):
        mask = seq_len_to_mask(src_lengths)
        encoder_dict = self.model(src_tokens, mask)
        encoder_output = encoder_dict.last_hidden_state
        hidden_states = encoder_dict.hidden_state

        return encoder_output, mask, hidden_states

class CopyBartDecoder(torch.nn.Module):
    def __init__(self, args, decoder, padding_token_id, avg_features=False,use_encoder_mlp=None):
        super(CopyBartDecoder).__init__()
        assert isinstance(decoder, BartDecoder), "the decoder type is not BartDecoder!"
        # Define parameters
        self.coverage = args.coverage
        self.is_copy = args.is_copy

        self.decoder = decoder
        self.padding_token_id = padding_token_id
        self.avg_features = avg_features

        causal_mask = torch.zeros(512,512).fill_(float("-inf"))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer("causal_mask", causal_mask.float())

        hidden_size = decoder.embed_tokens.weight.size(1)

        self.use_encoder_mlp = use_encoder_mlp
        if self.use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))
        if self.is_copy:
            self.decoder_encoder_hidden_transform = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, decoder_input, encoder_state):

        #
        # mask = decoder_input.eq(self.padding_token_id)
        cumsum = decoder_input.eq(self.eos_token_id).flip(dims=[1]).cumsum(dim=-1)  # 和上一行有啥区别？paddin_token_id 不知道？
        decoder_padding_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])
        decoder_input = decoder_input.masked_fill(decoder_padding_mask, self.padding_token_id)

        encoder_hidden_states = encoder_state.encoder_output
        encoder_pad_mask = encoder_state.encoder_output


        if self.training:
            dict = self.decoder(input_ids=decoder_input,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_padding_mask=encoder_pad_mask,
                            decoder_padding_mask=decoder_padding_mask,
                            decoder_causal_mask=self.causal_mask[:decoder_input.size(1), :decoder_input.size(1)],
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True)

        else:
            past_key_value = encoder_state.past_key_value

            dict = self.decoder(input_ids=decoder_input,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_padding_mask=encoder_pad_mask,
                            decoder_padding_mask=None,
                            decoder_causal_mask=None,
                            past_key_value=past_key_value,
                            output_attentions=True,
                            output_hidden_states=True,
                            use_cache=True,
                            return_dict=True)
        if not self.training:
            encoder_state.past_key_value = dict.past_key_value

        hidden_state = dict.last_hidden_state  #[B, T, H]
        if self.is_copy:
            copy_attn = dict.encoder_decoder_attention[-1]  # [B, Head, T, S]  # 使用cross attention 进行copy

            # copy_attn = self.get_copy_score(hidden_state, encoder_state)  # 使用output进行copy, [B, T, S]
        else:
            copy_attn = None

        if self.is_copy and self.coverage:

            pass
        if self.coverage and not self.is_copy:
            ValueError("use coverage should in copy mechanism situation.")

        return hidden_state, copy_attn

    def get_copy_score(self, decoder_hidden_state, state):
        '''

        :param decoder_hidden_state: [B, T, H]
        :param state: class contains encoder_output, encoder_mask ,etc.
        :return: attns [B, T, S]
        '''
        encoder_ouptut = state.encoder_output
        encoder_mask = state.encoder_mask

        decoder_encoder_state = F.tanh(self.decoder_encoder_hidden_transform(decoder_hidden_state))  # [B, T, H]

        copy_score = torch.einsum("bth,blh->btl", decoder_encoder_state, encoder_ouptut)  # [B, T, S]

        copy_score.masked_fill_(encoder_mask.unsqueeze(1),float('-inf'))

        copy_scpre = F.softmax(copy_score,dim=-1)

        return copy_scpre





