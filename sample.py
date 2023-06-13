"""
Sample from a trained model
"""
import os
import numpy as np
from contextlib import nullcontext
import torch
import argparse
import tiktoken
from models.model import GPTConfig, GPT
from tensorflow.keras.preprocessing.text import Tokenizer

num_samples = 1
max_new_tokens = 20 # max tokens per sample (~tokens per sample)
temperature = 1.0 # see https://medium.com/@imisri1/how-to-set-sampling-temperature-for-gpt-models-762887df1fac
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337 # for reproducibility
device = 'cuda' # set to cpu on mac
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'

# parse optional command line arguments
parser = argparse.ArgumentParser(description="out directory, start of text")
parser.add_argument('--out_dir', type=str, default="out", help='Out directory of model (ex. --out_dir="out-ipsum")')
parser.add_argument('--start', type=str, default="lorem" ,help='Start of text (ex. --start="Lorem Ipsum dolor...")')
args = parser.parse_args()
out_dir = args.out_dir
start = args.start

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# see https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# this is some pytorch prefix bug work around
# might be gone now? try uncommenting and see what happens
# --------------------------------------------------------
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
# --------------------------------------------------------

model.load_state_dict(state_dict)
model.eval()
model.to(device)
model = torch.compile(model)

meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.npz')
load_meta = os.path.exists(meta_path)
meta = np.load(meta_path, allow_pickle=True)

# load dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'data/ipsum/input.txt')
with open(dataset_path, 'r') as f:
    data = f.read()

# tokenization (filters no characters)
# for automatic punctuation filtering, remove filters argument
tokenizer = Tokenizer(char_level=False, filters='')
tokenizer.fit_on_texts([data])

# stoi (string to integer) and itos (integer to string) mappings
stoi = tokenizer.word_index
itos = {v: k for k, v in stoi.items()}

# encoder: take a string, output a list of integers
encode = lambda s: tokenizer.texts_to_sequences([s])[0]
# decoder: take a list of integers, output a string
decode = lambda l: ' '.join([itos[i] for i in l])

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # hack around model generating integers (strings) not in training data;
            # prevents KeyError when you make num_samples and max_new tokens ridiculously large
            # TODO: see if this error fixes iteself when using much more training data
            try:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
            except:
                pass
