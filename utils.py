
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
SEQ_WORD = '<seq>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT_WORD = '<digit>'

max_src_len = 0
max_target_len = 0
PAD_ID = 0

SRC_FILE_TRAIN_PATH = r'D:\CS\Train\deep-keyphrase-master\deep-keyphrase-master\data\raw\kp20k_training.json'
SRC_FILE_VALID_PATH = r'D:\CS\Train\deep-keyphrase-master\deep-keyphrase-master\data\raw\kp20k_validation.json'
SRC_FILE_TEST_PATH = r'D:\CS\Train\deep-keyphrase-master\deep-keyphrase-master\data\raw\kp20k_testing.json'
DEST_TRAIN_PATH = r'D:\CS\Train\deep-keyphrase-master\deep-keyphrase-master\data\kp20k.train.jsonl'
DEST_VALID_PATH = r'D:\CS\Train\deep-keyphrase-master\deep-keyphrase-master\data\kp20k.valid.jsonl'
DEST_TEST_PATH = r'D:\CS\Train\deep-keyphrase-master\deep-keyphrase-master\data\kp20k.test.jsonl'
DEST_VOCAB_PATH = r'D:\CS\Train\deep-keyphrase-master\deep-keyphrase-master\data\vocab_kp20k.txt'


TOKENS = 'tokens'
TOKEN_LENGTH = 'token_length'
TOKEN_WITH_OOV = 'token_with_oov'
OOV_LENGTH = 'oov_length'
OOV_LISTS = 'oov_lists'
TARGETS = 'targets'