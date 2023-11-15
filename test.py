import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from models.seq2seq_lstm import LSTMSeq2Seq
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
import pickle as pkl
from utils.seq2emo_metric import get_metrics, get_multi_metrics, jaccard_score, get_single_metrics
from utils.tokenizer import GloveTokenizer
from copy import deepcopy
from allennlp.modules.elmo import Elmo, batch_to_ids
import argparse
from data.data_loader import load_sem18_data, load_goemotions_data
from utils.scheduler import get_cosine_schedule_with_warmup
import utils.nn_utils as nn_utils
from utils.others import find_majority
from utils.file_logger import get_file_logger
from data.data_loader import load_sem18_data, load_goemotions_data, TextProcessor

import sys 
sys.path.append("/uusoc/exports/scratch/yyzhuang/emotion-reaction/model-scripts/")
import tuple_utils
# Argument parser
parser = argparse.ArgumentParser(description='Options')
args = parser.parse_args()
args.batch_size = 32
args.pad_len = 50
args.postname = ''
args.gamma=0.2
args.en_dim = 1200
args.de_dim = 400
args.glove_path='data/glove.840B.300d.txt'
args.attention ='dot'
args.dropout = 0.3
args.encoder_dropout =0.2
args.decoder_dropout = 0
args.attention_dropout=0.2
args.patience = 13 
args.download_elmo = True
args.warmup_epoch = 0
args.stop_epoch = 10 
args.max_epoch = 20
args.min_lr_ratio = 0.1
args.seed = 1
args.dev_split_seed = 0
args.eval_every = True
args.log_path = None
args.attention_heads = 1
args.output_path = None
args.attention_type = "luong"
args.shuffle_emo = None


if args.log_path is not None:
    dir_path = os.path.dirname(args.log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

logger = get_file_logger(args.log_path)  # Note: this is ugly, but I am lazy

SRC_EMB_DIM = 300
MAX_LEN_DATA = args.pad_len
PAD_LEN = MAX_LEN_DATA
MIN_LEN_DATA = 3
BATCH_SIZE = args.batch_size
CLIPS = 0.666
GAMMA = 0.5
SRC_HIDDEN_DIM = args.en_dim
TGT_HIDDEN_DIM = args.de_dim
VOCAB_SIZE = 60000
ATTENTION = args.attention
PATIENCE = args.patience
WARMUP_EPOCH = args.warmup_epoch
STOP_EPOCH = args.stop_epoch
MAX_EPOCH = args.max_epoch
RANDOM_SEED = args.seed
# Seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



# Init Elmo model
if args.download_elmo:
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
else:
    options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0).cuda()
elmo.eval()

GLOVE_EMB_PATH = args.glove_path
glove_tokenizer = GloveTokenizer(PAD_LEN)


# -------------------------
go_X_train_dev, go_y_train_dev, go_X_test, go_y_test, EMOS, EMOS_DIC, data_set_name = load_goemotions_data()
glove_tokenizer.build_tokenizer(go_X_train_dev + go_X_test, vocab_size=VOCAB_SIZE)
glove_tokenizer.build_embedding(GLOVE_EMB_PATH, dataset_name=data_set_name)


lemma_should_get_word = set([
    "i",
    "my",
    "me",
    "we",
    "our",
    "us",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their"
    ])

def get_lemmatized_ephrase_with_correct_pronoun(comp_idx, neg_idx, lemmas, words):
    ret = [] 
    for compid, nid in zip(comp_idx, neg_idx):
        if nid is not None: 
            ret.append('not')
        for cid in compid:
            l = lemmas[cid].lower()
            w = words[cid].lower()
            if l in lemma_should_get_word:
                ret.append(w)
            else:
                ret.append(l)
    return " ".join(" ".join(ret).split())

def get_tuples_with_bp(event_dicts, bp_idx, lemmas, words):
    event_phrases = [] 
    for edict in event_dicts:
        word_idx = [ind for comp_ind in edict['event_inds'] for ind in comp_ind]
        if bp_idx in word_idx:
            ephrase = get_lemmatized_ephrase_with_correct_pronoun(edict['event_inds'], edict['neg_inds'], lemmas, words)
            event_phrases.append(ephrase)

    return list(set(event_phrases))


def load_er_data(include_prev_sentence = False, kwords = 0, use_events = 0):

    with open("/uusoc/exports/scratch/yyzhuang/emotion-reaction/train_dev_test/test.json.final") as f:
        data = json.load(f)


    prev_sentences = [" ".join(t['prev_sentence']) for t in data]
    sentences = [t['sentence'] for t in data]

    x_train = []
    labels = [int(d['label']) - 1 for d in data]

    if include_prev_sentence:
        x_train = [f"{p} {s}" for p, s in zip(prev_sentences, sentences)]

    elif kwords > 0: 

        sentences = []
        for t in data: 
            center = t['word_bound'][0] + 1
            words = t['sentence'].split() 
            left = max(0, center - kwords)
            right = min(len(words), center + kwords + 1)
            sentences.append(" ".join(words[left:right]))  
        x_train = sentences

    elif use_events:
        for i, d in enumerate(data):
            # d['mod_head'] = {int(mod): d['mod_head'][mod] for mod in d['mod_head']}
            mod_head = d['enhancedplusplus_mod_head'] if 'enhancedplusplus_mod_head' in d else d['mod_head'] 
            mod_head = {int(mod): mod_head[mod] for mod in mod_head}

            event_dicts = tuple_utils.get_events_for_sentence(d['sentence'].split(), d['pos'], d['lemma'], mod_head, d['ner'])
            bp_idx = d['word_bound'][0] + 1 
            ephrases = get_tuples_with_bp(event_dicts, bp_idx, d['lemma'], d['sentence'].split())
            x_train.append(ephrases)
    else:
        x_train = [s for p, s in zip(prev_sentences, sentences)]
    
    return x_train, labels



test_para, labels  = load_er_data(include_prev_sentence = True)
test_window, _  = load_er_data(kwords = 6)
test_events, _ = load_er_data(use_events = True)


# ------------------------------------------------------------------
NUM_EMO = len(EMOS)

class TestDataReader(Dataset):
    def __init__(self, X, pad_len):
        self.glove_ids = []
        self.glove_ids_len = []
        self.pad_len = pad_len
        self.build_glove_ids(X)

    def build_glove_ids(self, X):
        for src in X:
            glove_id, glove_id_len = glove_tokenizer.encode_ids_pad(src)
            self.glove_ids.append(glove_id)
            self.glove_ids_len.append(glove_id_len)

    def __len__(self):
        return len(self.glove_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]])


def elmo_encode(ids):
    data_text = [glove_tokenizer.decode_ids(x) for x in ids]
    with torch.no_grad():
        character_ids = batch_to_ids(data_text).cuda()
        elmo_emb = elmo(character_ids)['elmo_representations']
        elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers
    return elmo_emb

test_set = TestDataReader(X_test, MAX_LEN_DATA)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE*3, shuffle=False)
    
model = torch.load('model.pth')
model.cuda()
model.eval()

preds = []
logger("Testing:")
for i, loader_input in tqdm(enumerate(test_loader), total=int(len(test_set) / BATCH_SIZE)):
    with torch.no_grad():
        src, src_len = loader_input
        elmo_src = elmo_encode(src)
        decoder_logit = model(src.cuda(), src_len.cuda(), elmo_src.cuda())
        preds.append(np.argmax(decoder_logit.data.cpu().numpy(), axis=-1))
        del decoder_logit

preds = np.concatenate(preds, axis=0)
gold = np.asarray(y_test)
binary_gold = gold
binary_preds = preds
logger("NOTE, this is on the test set")
metric = get_metrics(binary_gold, binary_preds)
logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
metric = get_multi_metrics(binary_gold, binary_preds)
logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
# show_classification_report(binary_gold, binary_preds)
logger('Jaccard:', jaccard_score(gold, preds))
return binary_gold, binary_preds


