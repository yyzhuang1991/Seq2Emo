from trainer_lstm_seq2emo import * 
from data.data_loader import load_sem18_data, load_goemotions_data, TextProcessor

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
ENCODER_LEARNING_RATE = args.en_lr
DECODER_LEARNING_RATE = args.de_lr
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

data_path_postfix = '_split'
data_pkl_path = <todo>

# -------------------------
go_X_train_dev, go_y_train_dev, go_X_test, go_y_test, EMOS, EMOS_DIC, data_set_name = load_goemotions_data()
glove_tokenizer.build_tokenizer(go_X_train_dev + go_X_test, vocab_size=VOCAB_SIZE)
glove_tokenizer.build_embedding(GLOVE_EMB_PATH, dataset_name=data_set_name)


def load_er_data():
    load test texts

    text_processor = TextProcessor()
    X_test = [ text_processor.processing_pipeline(x) for x in test_texts]
    y_test = [0] * len(X_test)
    EMOS = emo_list
    EMOS_DIC = {}
    for idx, emo in enumerate(EMOS):
        EMOS_DIC[emo] = idx
    data_set_name = ''
    return X_test, y_test

X_test, y_test  = load_er_data()   
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


