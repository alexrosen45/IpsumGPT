"""
Sample from a trained model
"""
import os
import json
from contextlib import nullcontext
import torch
from arguments import model_args, generate_args
from model.model import GPT
from tensorflow.keras.preprocessing.text import Tokenizer

num_samples = 1
max_new_tokens = 40 # max tokens per sample (~tokens per sample)
temperature = 1.0 # see https://medium.com/@imisri1/how-to-set-sampling-temperature-for-gpt-models-762887df1fac
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337 # for reproducibilitymodel_args
device = 'cuda' # set to cpu on mac
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'

# TODO: implement this later
# # Get necessary metadata
# with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
#     metadata = json.load(f)
# vocab_size = metadata['vocab_size']
# use_uint16 = metadata['uint16'] # for encoding/decoding
# char_level = metadata['char_level_tokenization'] # for encoding/decoding

# Get command line and default arguments and perform checks
args = generate_args.get_args()
generate_args.arg_checks(args)

# Generation parameters
model_dir = 'models/' + args.model
start = args.start

# Get necessary metadata
with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
    metadata = json.load(f)
data_dir = metadata['data_dir']

# Load dataset
with open(data_dir + '/input.txt', 'r') as f:
    data = f.read()

# Tokenization (filters no characters)
# for automatic punctuation filtering, remove filters argument
tokenizer = Tokenizer(char_level=False, filters='')
tokenizer.fit_on_texts([data])

# stoi (string to integer) and itos (integer to string) mappings
stoi = tokenizer.word_index
itos = {v: k for k, v in stoi.items()}

# encoder: take a string, output a list of integers
encode = lambda s: tokenizer.texts_to_sequences([s])[0]
# decoder: take a list of integers, output a string
decode = lambda l: ' '.join([itos[i] for i in l])# Tokenization (filters no characters)
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

# PyTorch setup
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# see https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
ckpt_path = os.path.join(model_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model = GPT(
    **model_args.get_model_args(
        args=checkpoint['params'],
        vocab_size=checkpoint['params']['vocab_size']
    )
)
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
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(f'\n{decode(y[0].tolist())}\n')
            try:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(f'\n{decode(y[0].tolist())}\n')
            except:
                pass
