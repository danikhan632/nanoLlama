from transformers import LlamaTokenizer
import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

# Number of workers for processing
num_proc = 12
num_proc_load_dataset = num_proc

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained("Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct")

def process(example):
    ids = tokenizer.encode(example['text'])
    ids.append(tokenizer.eos_token_id)
    return {'ids': ids, 'len': len(ids)}

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint32  # Updated to handle vocab size of 32000
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # To read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint32, mode='r')
