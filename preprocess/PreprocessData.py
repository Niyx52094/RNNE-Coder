import utils
from utils import (PAD_WORD, UNK_WORD, SEQ_WORD, BOS_WORD, EOS_WORD, DIGIT_WORD)
import os
import re
import argparse
import string
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pysenal import get_chunk, read_jsonline_lazy, append_jsonlines, write_lines
from tqdm import tqdm



class Kp20kPreprocessor(object):
    """
    kp20k data preprocessor, build the data and vocab for training.

    """
    num_and_punc_regex = re.compile(r'[_\-—<>{,(?\\.\'%]|\d+([.]\d+)?', re.IGNORECASE)
    num_regex = re.compile(r'\d+([.]\d+)?')

    def __init__(self, args):
        self.src_filename = args.src_filename
        self.dest_filename = args.dest_filename
        self.dest_vocab_path = args.dest_vocab_path
        self.vocab_size = args.vocab_size
        self.parallel_count = args.parallel_count
        self.is_src_lower = args.src_lower
        self.is_src_stem = args.src_stem
        self.is_target_lower = args.target_lower
        self.is_target_stem = args.target_stem
        self.is_test = args.is_test
        self.stemmer = PorterStemmer()
        if os.path.exists(self.dest_filename):
            print('destination file existed, will be deleted!!!')
            assert 0 == 1

    def process(self):
        pool = Pool(self.parallel_count)
        tokens = []
        chunk_size = 2
        # read_json = read_jsonline_lazy(self.src_filename)  # read_line_lazy 是读取一行
        # for index, line in enumerate(read_json):
        #     print(line)
        #     if index + 1 == 2:
        #         break
        for item_chunk in tqdm(get_chunk(read_jsonline_lazy(self.src_filename), chunk_size)):  #get_chunk 是封装的版本。直接读取chunk_size大小列表的行数
            preprocess_records = pool.map(self.tokenize_record, item_chunk)  # 把这几百行输入到tokenize_record里面做处理
            # print('type of preprccess_recored is:', type(preprocess_records))
            # 得到tokens 之后需要输出到vocab里面，做成词汇表
            if self.dest_vocab_path:
                for record in preprocess_records:
                    # 把所有token组合起来
                    tokens.extend(record['title_and_abstract_tokens'] + record['flatten_keyword_tokens'])
            for record in preprocess_records:
                record.pop('flatten_keyword_tokens')
            if not self.is_test:
                preprocess_records_flatten = self.flatten_token_list(preprocess_records)
            else:
                preprocess_records_flatten = self.test_token_list(preprocess_records)
            # 输出flatten 的 tokens
            append_jsonlines(self.dest_filename, preprocess_records_flatten) #输出tokens
        if self.dest_vocab_path:
            vocab = self.build_vocab(tokens)
            write_lines(self.dest_vocab_path, vocab)

    def test_token_list(self, data_dict_list):
        test_token_records = []
        for data_dict in data_dict_list:
            test_token_records.append({'title_and_abstract_tokens': data_dict['title_and_abstract_tokens'],
                                        'keyphrase': data_dict.get('keyword_tokens')})
        return test_token_records
    def flatten_token_list(self, data_dict_list):
        flatten_records = []
        for data_dict in data_dict_list:
            for token_list in data_dict.get('keyword_tokens'):
                flatten_records.append({'title_and_abstract_tokens': data_dict['title_and_abstract_tokens'],
                                        'keyphrase': token_list})
        return flatten_records

    def build_vocab(self, tokens):
        # vocab 是一个 list，word2id 和id2word 才是对应词典
        vocab = [PAD_WORD, UNK_WORD, SEQ_WORD, BOS_WORD, EOS_WORD, DIGIT_WORD]
        vocab.extend(list(string.digits))

        # 因为vocab不能无限大，所以只能取最高频的词汇数目
        token_count = dict()
        tokens = [token.lower() for token in tokens]
        print('generate vocab start')
        for token in tqdm(tokens):
                token_count[token] = token_count.get(token, 0) + 1
        sorted_tokens = sorted(token_count.items(), key=lambda x: x[1], reverse=True)
        for token, count in sorted_tokens:
            vocab.append(token)
            if len(vocab) >= self.vocab_size:
                break
        return vocab


    def tokenize_record(self, record):
        abstract_tokens = self.tokenize(record['abstract'], self.is_src_lower, self.is_src_stem)
        title_tokens = self.tokenize(record['title'], self.is_src_lower, self.is_src_stem)
        keyword_token_list = []
        for keyword in record['keyword'].split(";"):
            # 一个keyword是一个短语，所以返回的list可能是由多个token组成的
            keyword_token_list.append(self.tokenize(keyword, self.is_target_lower, self.is_target_stem))
        # print("keyword_token_list", keyword_token_list)  # [['biomedical', 'text'], ['machine', 'learning'], ['information', 'extraction']]
        results = {
            'title_and_abstract_tokens': title_tokens + abstract_tokens,
            'keyword_tokens': keyword_token_list,
            # 将内部list去掉,* 先去掉外部[]，chain把内部的list穿起来，list把他们做成[]
            'flatten_keyword_tokens': list(chain(*keyword_token_list))
        }
        return results

    def tokenize(self, text, is_lower, is_stem):
        text = self.num_and_punc_regex.sub(r' \g<0> ', text)  # 将标点等字符前后都拉开一个空格，\g<0>表示那个字符
        tokens = word_tokenize(text)  # 提取tokens 并生成list
        # print(tokens)
        # print(type(tokens))
        if is_lower:
            tokens =[token.lower() for token in tokens]
        if is_stem:
            # 生成主干，比如会去掉过去式的ed等。constrained -> constrain,但也会多去掉,比如 created ->creat等
            tokens = [self.stemmer.stem(token) for token in tokens]
            # print(tokens)
        for idx, token in enumerate(tokens):  # 将数字中转化成<digit> 符号
            token = tokens[idx]
            if self.num_regex.fullmatch(token):
                tokens[idx] = DIGIT_WORD
        return tokens



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_filename', type=str,
                        help='input source kp20k file path')
    parser.add_argument('-dest_filename', type=str,
                        help='destination of processed file path')
    parser.add_argument('-dest_vocab_path', type=str,
                        help='')
    parser.add_argument('-vocab_size', type=int, default=50000,
                        help='')
    parser.add_argument('-parallel_count', type=int, default=10)
    parser.add_argument('-src_lower', action='store_true')
    parser.add_argument('-src_stem', action='store_true')
    parser.add_argument('-target_lower', action='store_true')
    parser.add_argument('-target_stem', action='store_true')
    parser.add_argument('-is_test', action='store_true')



    args = parser.parse_args()

    args.src_filename = utils.SRC_FILE_TRAIN_PATH
    args.dest_filename = utils.DEST_TRAIN_PATH
    if args.src_filename == utils.SRC_FILE_TRAIN_PATH:
        args.dest_vocab_path = utils.DEST_VOCAB_PATH
    processor = Kp20kPreprocessor(args)
    processor.process()
    #
    # # 处理valid
    # args.src_filename = utils.SRC_FILE_VALID_PATH
    # args.dest_filename = utils.DEST_VALID_PATH
    #
    #
    # processor = Kp20kPreprocessor(args)
    # processor.process()

    # # 处理test
    # args.src_filename = utils.SRC_FILE_TEST_PATH
    # args.dest_filename = utils.DEST_TEST_PATH


    # processor = Kp20kPreprocessor(args)
    # processor.process()




if __name__ == '__main__':
    main()