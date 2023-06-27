import os
import glob
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime
from arguments import process_args
from tensorflow.keras.preprocessing.text import Tokenizer

# Parse optional command line arguments
args = process_args.get_args()
process_args.arg_checks(args)

# Processing parameters
home_dir = args.data_dir
data_dir = args.data_dir + "/" + args.data_folder
fetch = not args.no_fetch
split_ratio = args.split_ratio # default is 0.8
# char level constant for Keras tokenizer
# set to True for character level tokenization,
# or False for word level tokenization
# see https://huggingface.co/docs/transformers/tokenizer_summary
char_level = args.char_level
# use np.uint16 instead of np.uint8
use_uint16 = args.np_uint16

# Create dataset file if it does not exist
dataset = os.path.join(home_dir, 'input.txt')
if not os.path.exists(dataset):
    with open(dataset, 'a'):
        pass
    print("Created input.txt")

# Parse dataset
if fetch:
    # Get all files with no extension or .txt extension
    all_files = glob.glob(os.path.join(data_dir, '*'))
    lorem_ipsum_files = [f for f in all_files if '.' not in os.path.basename(f)]
    lorem_ipsum_files.extend(glob.glob(os.path.join(data_dir, '*.txt')))

    iter, percent_parsed = 0, 0 # Num of lorem_ipsum files and percent parsed files

    with open(home_dir + '/input.txt', 'w') as outfile:
        for file in tqdm(lorem_ipsum_files, desc="Reading files", unit="file"):
            # Write to input.txt
            with open(file, 'r') as infile:
                try:
                    outfile.write(infile.read())
                    outfile.write('\n')
                except Exception as e:
                    print(f"Could not read file {file}. Error: {str(e)}")

    print(f'Read 100% of all files in {data_dir}')

# Prepare dataset for training
dataset_path = os.path.join(os.path.dirname(__file__), home_dir + '/input.txt')
with open(home_dir + '/input.txt', 'r') as f:
    data = f.read()

# Tokenization (filters no characters)
# For automatic punctuation filtering, remove filters argument
tokenizer = Tokenizer(char_level=char_level, filters='')
tokenizer.fit_on_texts([data])

vocab_size = len(tokenizer.word_index) + 1  # +1 from reserved 0 index for padding
print(f'Vocab size: {vocab_size}')

# stoi (string to integer) and itos (integer to string) mappings
stoi = tokenizer.word_index
itos = {v: k for k, v in stoi.items()}

# encoder: take a string, output a list of integers
encode = lambda s: tokenizer.texts_to_sequences([s])[0]
# decoder: take a list of integers, output a string
decode = lambda l: ' '.join([itos[i] for i in l])

# Train test splits into bin files with encoding
# If vocab size exceeds 255, increase to uint16
if vocab_size > 255:
    use_uint16 = True
print(f'Split ratio {split_ratio}')
np.array(
    encode(data[:int(len(data) * split_ratio)]),
    dtype=np.uint16 if use_uint16 else np.uint8
).tofile(os.path.join(os.path.dirname(__file__), home_dir + '/train.bin'))
np.array(
    encode(data[int(len(data) * split_ratio):]),
    dtype=np.uint16 if use_uint16 else np.uint8
).tofile(os.path.join(os.path.dirname(__file__), home_dir + '/test.bin'))

metadata = {
    'vocab_size': vocab_size,
    'generation_date': datetime.now().isoformat(),
    'train_test_split': split_ratio,
    'data_dir': home_dir,
    'char_level_tokenization': char_level,
    'uint16': use_uint16, # uint8 otherwise
    'int_to_str': itos,
    'stf_to_int': stoi,
}

# Save process metadata to a JSON file
with open(os.path.join(os.path.dirname(__file__), home_dir + '/metadata.json'), 'w') as file:
    json.dump(metadata, file)
print("Processing complete")