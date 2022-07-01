import json
import json
from utils import (PAD_WORD, SEQ_WORD, BOS_WORD, EOS_WORD,DIGIT_WORD, UNK_WORD)
# from pysenal import get_chunk, read_jsonline_lazy, append_jsonlines, write_lines
from preprocess.KP20DataSet import Kp20DataSet
from torch.utils.data import DataLoader

class Kp20Engine(object):
    def __init__(self, args, model, optimizer,scheduler):

        self.vocab_size = args.vocab_size
        self.vocab, self.vocab2id, self.id2vocab = self.load_vocab(args.vocab_path)
        print('vocab length is %d'%len(self.vocab))
        # self.dataset = Kp20DataSet(args, self.vocab, self.vocab2id)
        self.model = self.load_model(model)
        self.dataset = None
        self.dataloader = None
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler

    def load_model(self,model):
        if model is not None:
            return model

    def load_vocab(self, vocab_path):
        vocab = []
        vocab2id = {}
        id2vocab = {}
        print(vocab_path)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            # print(json.load(f, strict=False))
            files = f.read().split('\n')
            print(files[:10])
            for line in files:
                vocab.append(str(line))
                if len(vocab) >= self.vocab_size:
                    break

        print(vocab[:10])
        if SEQ_WORD not in vocab:
            print("lost seq word")
        if PAD_WORD not in vocab:
            print("lost padding word")
        if BOS_WORD not in vocab:
            print("lost bps word")
        if EOS_WORD not in vocab:
            print("lost eos word")
        if DIGIT_WORD not in vocab:
            print("lost digit word")
        if UNK_WORD not in vocab:
            print("lost unknown word")
        for idx, token in enumerate(vocab):
            vocab2id[token] = idx
            id2vocab[idx] = token
        return vocab, vocab2id, id2vocab

    def get_dataset(self, mode):
        return Kp20DataSet(self.args, self.vocab, self.vocab2id, mode=mode)

    def get_dataloader(self, dataset, batch_size, shuffle, collate_fn):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return data_loader
