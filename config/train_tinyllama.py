# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-tinyllama'
eval_interval = 10 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare'
gradient_accumulation_steps = 2
batch_size = 64
block_size = 256 # context of up to 256 previous characters
flash_attn = False
# baby GPT model :)
dim: int = 384
n_layers: int = 2
n_heads: int = 4
vocab_size: int = 32000
max_seq_len: int = 256
hidden_dim: int = 11008

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 30
lr_decay_iters = 30 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
