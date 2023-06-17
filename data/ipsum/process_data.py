import os
import glob
import sys
import argparse
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# train test split ratio for training data
SPLIT_RATIO = 0.8
# char level constant for Keras tokenizer
# set to True for 'character level' tokenization,
# or False for 'word level' tokenization
# see https://huggingface.co/docs/transformers/tokenizer_summary
CHAR_LEVEL = False

directory = 'data/ipsum/lorem-ipsum-dataset'
fetch = False
process = True

# parse optional command line arguments
parser = argparse.ArgumentParser(description="dataset directory, fetch, process")
parser.add_argument('--fetch_dir', type=str, help='Dataset directory (ex. --fetch_dir="data/ipsum/lorem-ipsum-dataset")')
parser.add_argument('--fetch', type=bool, help='Fetch dataset? (ex. --fetch=False)')
parser.add_argument('--process', type=bool, help='Process dataset? (ex. --process=True)')
args = parser.parse_args()
directory = args.fetch_dir
fetch = args.fetch

output_file = 'data/ipsum/input.txt'

# parse dataset
if fetch:
    # Get all files with no extension or .txt extension
    all_files = glob.glob(os.path.join(directory, '*'))
    lorem_ipsum_files = [f for f in all_files if '.' not in os.path.basename(f)]
    lorem_ipsum_files.extend(glob.glob(os.path.join(directory, '*.txt')))

    iter, percent_parsed = 1, 0 # Num of lorem_ipsum files and percent parsed files
    print(f'Parsing {len(lorem_ipsum_files)} files...')

    with open(output_file, 'w') as outfile:
        for file in lorem_ipsum_files:
            # Update progress in console
            if round(len(lorem_ipsum_files)/iter) > percent_parsed:
                print(f'{percent_parsed}% complete')
                percent_parsed += 1
            iter += 1

            # Write to output_file with space in-between
            with open(file, 'r') as infile:
                try:
                    outfile.write(infile.read())
                    outfile.write('\n')
                except Exception as e:
                    print(f"Could not read file {file}. Error: {str(e)}")

    print(f'parsing complete\nread {percent_parsed}% of all files')

# kill process if fetching only
if not process:
    print("Not processing data: --process=False")
    sys.exit(0)

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