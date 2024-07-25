# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-llama'


# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
flash_attn = False
gradient_accumulation_steps = 5 * 4

batch_size = 64
block_size = 4096 # context of up to 256 previous characters

#

n_layer = 32
n_head = 32
dim = 4096
hidden_dim = 11008
# optimizer

dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 30
lr_decay_iters = 30 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 0 # not super necessary potentially


# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
