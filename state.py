from ctypes import Union

import torch

class State:
    def __init__(self, encoder_output=None, encoder_mask=None, **kwargs):
        """
        每个Decoder都有对应的State对象用来承载encoder的输出以及当前时刻之前的decode状态。

        :param Union[torch.Tensor, list, tuple] encoder_output: 如果不为None，内部元素需要为torch.Tensor, 默认其中第一维是batch
            维度
        :param Union[torch.Tensor, list, tuple] encoder_mask: 如果部位None，内部元素需要torch.Tensor, 默认其中第一维是batch
            维度
        :param kwargs:
        """
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def num_samples(self):
        """
        返回的State中包含的是多少个sample的encoder状态，主要用于Generate的时候确定batch的大小。

        :return:
        """
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

    @property
    def decode_length(self):
        """
        当前Decode到哪个token了，decoder只会从decode_length之后的token开始decode, 为0说明还没开始decode。

        :return:
        """
        return self._decode_length

    @decode_length.setter
    def decode_length(self, value):
        self._decode_length = value

    def _reorder_state(self, state, indices, dim):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)


class GRUCopyState(State):
    def __init__(self, encoder_output, encoder_mask, encoder_token_with_oov, hidden,extra_vocab_size):
        """
        GRUDecoder对应的State，保存encoder的输出以及GRU解码过程中的一些中间状态

        :param torch.FloatTensor encoder_output: bsz x src_seq_len x encode_output_size，encoder的输出
        :param torch.BoolTensor encoder_mask: bsz x src_seq_len, 为0的地方是padding
        :param torch.FloatTensor hidden: num_layers x bsz x hidden_size, 上个时刻的hidden状态
        :param torch.FloatTensor hidden: num_layers x bsz x hidden_size, 上个时刻的hidden状态
        :param torch.LongTensor encoder_token_with_oov: [B, src_len]
        :param torch.LongTensor extra_vocab_size: int
        """
        super().__init__(encoder_output, encoder_mask)
        self._hidden = hidden
        self._input_feed = hidden[0]  # 默认是上一个时刻的输出
        self.encoder_token_with_oov = encoder_token_with_oov
        self.extra_vocab_size = extra_vocab_size
    @property
    def extra_vacab_size(self):
        return self.extra_vocab_size

    @property
    def input_feed(self):
        """
        LSTMDecoder中每个时刻的输入会把上个token的embedding和input_feed拼接起来输入到下个时刻，在LSTMDecoder不使用attention时，
            input_feed即上个时刻的hidden state, 否则是attention layer的输出。GRU decoder 同
        :return: torch.FloatTensor, bsz x hidden_size
        """
        return self._input_feed

    @input_feed.setter
    def input_feed(self, value):
        self._input_feed = value

    @property
    def hidden(self):
        """
        LSTMDecoder中每个时刻的输入会把上个token的embedding和input_feed拼接起来输入到下个时刻，在LSTMDecoder不使用attention时，
            input_feed即上个时刻的hidden state, 否则是attention layer的输出。GRU decoder 同
        :return: torch.FloatTensor, bsz x hidden_size
        """
        return self._hidden

    @hidden.setter
    def hidden(self, value):
        self._hidden = value

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        if self._hidden is not None:
            self._hidden = self._reorder_state(self._hidden, indices, dim=1)

        if self.encoder_token_with_oov is not None:
            self.encoder_token_with_oov = self._reorder_state(self.encoder_token_with_oov, indices, dim=1)

        if self._input_feed is not None:
            self._input_feed = self._reorder_state(self._input_feed, indices, dim=1)


class BartCopyState(State):
    def __init__(self, encoder_output, encoder_mask, encoder_token_with_oov,extra_vocab_size):
        """
        BartDecoder对应的State，保存encoder的输出以及GRU解码过程中的一些中间状态
        :param torch.FloatTensor encoder_output: bsz x src_seq_len x encode_output_size，encoder的输出
        :param torch.BoolTensor encoder_mask: [B, src_len], 为0的地方是padding
        :param torch.LongTensor encoder_token_with_oov: [B, src_len]
        """
        super().__init__(encoder_output, encoder_mask)
        self.encoder_token_with_oov = encoder_token_with_oov
        self.extra_vocab_size = extra_vocab_size


    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        if self.encoder_token_with_oov is not None:
            self.encoder_token_with_oov = self._reorder_state(self.encoder_token_with_oov, indices, dim=1)




