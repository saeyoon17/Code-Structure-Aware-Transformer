from dataset.fast_ast_data_set import FastASTDataSet
from module import CSATrans
from utils import PAD, LabelSmoothing

project_name = "final_exp"
# pe_dim sbm_enc_dim hidden_dim num_layers sbm_layers clusters batch size
task_name = "256_512_512_4_4_10_10_10_10_b64_tgt50_codescribe"
test_optimizer = True

seed = 2021
sw = 1e-2
use_pegen = "pegen"
pe_dim = 256
pegen_dim = 512
sbm_enc_dim = 512
num_layers = 4
sbm_layers = 4
clusters = [10, 10, 10, 10]
full_att = False
num_heads = 8
hidden_size = 512
dim_feed_forward = 2048
dropout = 0.2

# data
data_dir = f"./processed/compare_codescribe_python"
max_tgt_len = 50
max_src_len = 150
data_type = "pot"

# misc
is_test = False
testfile = ""
checkpoint = None

# train
batch_size = 64
num_epochs = 500
num_threads = 0
load_epoch_path = ""
val_interval = 5
save_interval = 50
data_set = FastASTDataSet
model = CSATrans
fast_mod = False
logger = ["tensorboard"]

# optimizer
learning_rate = 1e-4

# criterion
criterion = LabelSmoothing(padding_idx=PAD, smoothing=0.0)
g = "0"
