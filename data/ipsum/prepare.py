import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# train test split ratio for training data
SPLIT_RATIO = 0.8
# char level constant for Keras tokenizer
# set to True for 'character level' tokenization,
# or False for 'word level' tokenization
# see https://huggingface.co/docs/transformers/tokenizer_summary
CHAR_LEVEL = False

# parse optional command line arguments
parser = argparse.ArgumentParser(description="out directory")
parser.add_argument('--out_dir', type=str, help='Out directory of model (ex. --out_dir="out-ipsum")')
args = parser.parse_args()
if args.out_dir is not None:
    out_dir = args.out_dir

# load dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(dataset_path, 'r') as f:
    data = f.read()

# tokenization (filters no characters)
# for automatic punctuation filtering, remove filters argument
tokenizer = Tokenizer(char_level=CHAR_LEVEL, filters='')
tokenizer.fit_on_texts([data])

vocab_size = len(tokenizer.word_index) + 1  # +1 from reserved 0 index for padding
print(f'vocab size: {vocab_size}')

# stoi (string to integer) and itos (integer to string) mappings
stoi = tokenizer.word_index
itos = {v: k for k, v in stoi.items()}

# encoder: take a string, output a list of integers
encode = lambda s: tokenizer.texts_to_sequences([s])[0]
# decoder: take a list of integers, output a string
decode = lambda l: ' '.join([itos[i] for i in l])

# train test splits into bin files with encoding
# if vocab size exceeds 255, increase to uint16
# (or uint32 if you somehow need even more)
np.array(
    encode(data[:int(len(data) * SPLIT_RATIO)]),
    dtype=np.uint8
).tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
np.array(
    encode(data[int(len(data) * SPLIT_RATIO):]),
    dtype=np.uint8
).tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

# metadata for future encoding or decoding
# TODO: this is clunky, remove this later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
np.savez(os.path.join(os.path.dirname(__file__), 'meta.npz'), **meta)
