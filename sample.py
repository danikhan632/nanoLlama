""" Sample from a trained model """
import os
import pickle
from contextlib import nullcontext
import torch
from transformers import LlamaTokenizer
from model import ModelArgs, Llama

# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a path to a checkpoint
out_dir = 'out'
start = "\n"  # or "<s>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float32'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster

exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model = Llama(model_args)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
else:
    # init from a given checkpoint
    print(f"Loading model from {init_from}")
    checkpoint = torch.load(init_from, map_location=device)
    model_args = checkpoint['model_args']
    model = Llama(model_args)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

model.eval()
model.to(device)

if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# initialize LLaMA tokenizer
tokenizer = LlamaTokenizer.from_pretrained("Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct")

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = tokenizer.encode(start, add_special_tokens=False)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = tokenizer.decode(y[0].tolist())
            print(generated_text)
            print('---------------')