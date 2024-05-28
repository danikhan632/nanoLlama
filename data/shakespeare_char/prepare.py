import os
import pickle
import requests
import numpy as np
from transformers import LlamaTokenizer

# Initialize the LLaMA tokenizer
tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Download the tiny Shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Tokenize the entire dataset using the LLaMA tokenizer
tokens = tokenizer(data, add_special_tokens=False)['input_ids']
vocab_size = len(tokenizer)
print(f"vocab size: {vocab_size:,}")

# Create train and validation splits
n = len(tokens)
train_data = tokens[:int(n*0.9)]
val_data = tokens[int(n*0.9):]

print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")

# Export to bin files
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'tokenizer': tokenizer,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Dataset preparation is complete.")
