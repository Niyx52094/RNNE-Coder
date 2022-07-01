import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import utils
from utils import (PAD_WORD, UNK_WORD, SEQ_WORD, BOS_WORD, EOS_WORD, DIGIT_WORD)
import json

TOKENS = 'tokens'
TOKENS_LENS = 'tokens_len'
TOKENS_OOV = 'tokens_with_oov'
OOV_COUNT = 'oov_count'
OOV_LIST = 'oov_list'
TARGET_LIST = 'targets'
TARGET = 'target'
RAW_BATCH = 'raw'

TRAIN_MODE = 'train'
EVAL_MODE = 'eval'
INFERENCE_MODE = 'inference'


class Kp20DataSet(Dataset):
    def __init__(self,args,vocab, vocab2id, mode=None):
        # super(Kp20DataSet, self).__init__()
        self.vocab = vocab
        self.vocab2id = vocab2id
        self.max_src_len = args.max_src_len
        utils.max_src_len = args.max_src_len
        utils.PAD_ID = self.vocab2id[PAD_WORD]
        self.max_target_len = args.max_target_len
        utils.max_target_len = args.max_target_len

        self.token_field = args.token_field
        self.keyphrase_field = args.keyphrase_field
        self.max_oov_count = args.max_oov_count
        if mode == 'valid':
            self.data = self.load_data(args.valid_filename)
        elif mode == 'test':
            self.data = self.load_data(args.test_filename)
        else:
            self.data = self.load_data(args.train_filename)

    def load_data(self, file_path):
        result = []
        with open(file_path, 'r') as f:
            data = f.readlines()
        for line in tqdm(data):
            result.append(json.loads(line))
        return result
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_data = self.data[idx]

        src_data = one_data[self.token_field][:self.max_src_len]
        keyphrase_data = one_data[self.keyphrase_field]
        actual_src_length = len(src_data)
        actual_keyphrase_length = len(keyphrase_data)

        # change to id
        token_ids = []  # 一行token对应的id
        token_len_ids = []  # 一行token 对应的长度，要mask用, X
        token_ids_with_oov = []  # 带oov_id的token list， 因为vocab大小只有50000，大概率有些重要词不在里面但却很需要
        oov_list = []  # 把oov的词加进来
        oov_len_count = []  # 计算oov_list的长度, X 一个batch的时候要用，这里只要记录len(oov_list)
        decoder_input_ids = []  # decoder_input keyphrase 的ids (用unk_word)
        decoder_target_ids = [] # decoder_input keyphrase 的ids (用oov的word)
        torget_len_list = []  # keyphrase 真实的长度，做mask用, X 一样，后面batch的时候再弄即可
        for src_token in src_data:
            src_token = src_token.lower()
            if src_token in self.vocab2id.keys():
                token_ids.append(self.vocab2id.get(src_token))
                token_ids_with_oov.append(self.vocab2id.get(src_token))
            else:
                token_ids.append(self.vocab2id[UNK_WORD])
                # 看这个unknown的token是不是再oov_list里面
                if src_token in oov_list:
                    token_ids_with_oov.append(len(self.vocab) + oov_list.index(src_token))
                else:
                    if len(oov_list) >= self.max_oov_count:
                        token_ids_with_oov.append(len(self.vocab) + self.max_oov_count - 1)
                    else:
                        token_ids_with_oov.append(len(self.vocab) + len(oov_list))
                        oov_list.append(src_token)


        decoder_input_ids.append(self.vocab2id[BOS_WORD])

        for keyphrase_token in keyphrase_data:
            keyphrase_token = keyphrase_token.lower()
            single_key_id = self.vocab2id.get(keyphrase_token, self.vocab2id[UNK_WORD])

            #  说明是oov
            if single_key_id == self.vocab2id[UNK_WORD]:
                if keyphrase_token in oov_list:
                    decoder_target_ids.append(len(self.vocab) + oov_list.index(keyphrase_token))
                else:
                    decoder_target_ids.append(single_key_id)
            else:
                decoder_target_ids.append(single_key_id)
            decoder_input_ids.append(single_key_id)


        # decoder_input_ids.append(self.vocab2id[EOS_WORD])
        decoder_target_ids.append(self.vocab2id[EOS_WORD])

        decoder_input_ids = decoder_input_ids[:self.max_target_len]

        decoder_target_ids = decoder_target_ids[:self.max_target_len]

        final_retults = {
            'encoder_tokens_list': token_ids,
            'encoder_token_with_oov_list': token_ids_with_oov,
            'oov_length': len(oov_list),
            'oov_list': oov_list,
            'decoder_input_ids_list': decoder_input_ids,
            'decoder_target_ids_list': decoder_target_ids
        }
        return final_retults

def collate_fn(raw_batch_data):  # for padding
    oov_list = []
    oov_len_list = []

    decoder_input_list = []
    decoder_input_length_list = []

    decoder_target_list = []
    decoder_target_len_list = []

    tokens_list = []
    tokens_length_list = []
    token_ids_with_oov_list = []
    token_ids_with_oov_len_list = []

    pad_id = utils.PAD_ID

    src_input_mask = []
    dec_enc_mask = []

    for raw_data in raw_batch_data:
        # 计算长度做padding
        token_len = len(raw_data['encoder_tokens_list'])
        decoder_input_len = len(raw_data['decoder_input_ids_list'])
        decoder_target_len = len(raw_data['decoder_target_ids_list'])

        assert decoder_input_len == decoder_target_len, 'the decoder input and decoder output is not equal!!'

        # encoder 文本padding
        token_ids = raw_data['encoder_tokens_list'] + [pad_id] * (utils.max_src_len - token_len)
        src_input_mask.append([1] * token_len + [pad_id] * (utils.max_src_len - token_len))
        tokens_list.append(token_ids)
        tokens_length_list.append(token_len)

        token_ids_with_oov = raw_data['encoder_token_with_oov_list'] + [pad_id] * (utils.max_src_len - token_len)
        token_ids_with_oov_list.append(token_ids_with_oov)

        # oov_list 保存
        oov_list.append((raw_data['oov_list'], token_len))
        oov_len_list.append(raw_data['oov_length'])

        ## decoder input padding
        decoder_input_ids = raw_data['decoder_input_ids_list'] + [pad_id] * (utils.max_target_len - decoder_input_len)
        decoder_input_list.append(decoder_input_ids)
        decoder_input_length_list.append(decoder_input_len)

        ## decoder target padding
        decoder_target_ids = raw_data['decoder_target_ids_list'] + [pad_id] * (utils.max_target_len - decoder_target_len)
        dec_enc_mask.append([1] * decoder_target_len + [pad_id] * (utils.max_target_len - decoder_target_len))

        decoder_target_list.append(decoder_target_ids)
        decoder_target_len_list.append(decoder_target_len)


    tokens_list_ts = torch.tensor(tokens_list, dtype=torch.long)  # B X max_len
    tokens_length_list_ts = torch.tensor(tokens_length_list, dtype=torch.long)  #[B]
    src_input_mask = torch.tensor(src_input_mask)

    token_ids_with_oov_list_ts = torch.tensor(token_ids_with_oov_list, dtype=torch.long)

    oov_len_list_ts = torch.tensor(oov_len_list, dtype=torch.long)  # [B]

    decoder_inputs_list_ts = torch.tensor(decoder_input_list, dtype=torch.long)
    decoder_input_length_ts = torch.tensor(decoder_input_length_list, dtype=torch.long)  #[B]

    decoder_target_ts = torch.tensor(decoder_target_list, dtype=torch.long)
    decoder_target_length_ts = torch.tensor(decoder_target_len_list, dtype=torch.long)
    dec_mask = torch.tensor(dec_enc_mask)

    sorted_value, sorted_idices = torch.sort(tokens_length_list_ts, dim=-1,descending=True)

    # 按长度排序
    tokens_list_ts = tokens_list_ts.index_select(dim=0,index=sorted_idices)
    tokens_length_list_ts = tokens_length_list_ts.index_select(dim=0, index=sorted_idices)
    src_input_mask = src_input_mask.index_select(dim=0, index=sorted_idices)
    token_ids_with_oov_list_ts = token_ids_with_oov_list_ts.index_select(dim=0, index=sorted_idices)
    oov_len_list_ts = oov_len_list_ts.index_select(dim=0, index=sorted_idices)

    #decoder
    decoder_inputs_list_ts = decoder_inputs_list_ts.index_select(dim=0, index=sorted_idices)
    decoder_input_length_ts = decoder_input_length_ts.index_select(dim=0, index=sorted_idices)

    decoder_target_ts = decoder_target_ts.index_select(dim=0, index=sorted_idices)
    decoder_target_length_ts = decoder_target_length_ts.index_select(dim=0, index=sorted_idices)
    dec_mask = dec_mask.index_select(dim=0, index=sorted_idices)

    oov_list = sorted(oov_list, key=lambda x:x[1],reverse=True)

    oov_list = [singel_tup[0] for singel_tup in oov_list]

    processed_batch_data = {
        'encoder_tokens': tokens_list_ts,
        'encoder_token_length': tokens_length_list_ts,
        'encoder_token_with_oov': token_ids_with_oov_list_ts,
        'src_input_mask': src_input_mask,

        'oov_length': oov_len_list_ts,
        'oov_lists': oov_list,

        'decoder_input': decoder_inputs_list_ts,
        'decoder_input_lengths': decoder_input_length_ts,
        'decoder_target':decoder_target_ts,
        'decoder_target_lengths': decoder_target_length_ts,
        'decoder_mask':dec_mask,

        'batch_max_oov_length': max(oov_len_list)

    }
    return processed_batch_data





