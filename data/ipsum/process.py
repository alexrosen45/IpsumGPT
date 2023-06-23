import os
import glob
import json
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer

# Parse optional command line arguments
parser = argparse.ArgumentParser(description="Fetching and/or processing parameters")
parser.add_argument('--fetch_dir', type=str, help='Dataset directory (ex. --fetch_dir="data/ipsum/lorem-ipsum-dataset")')
parser.add_argument('--fetch', action='store_true', help='Fetch dataset with --fetch (requires non-empty --fetch_dir arg)')
parser.add_argument('--process', action='store_true', help='Prepare fetched dataset for training with --process')
parser.add_argument('--split_ratio', type=float, default = 0.8, help='Train test split ratio (ex. --split-ratio=0.8 trains on 80 percent of dataset)')
parser.add_argument('--char_level', action='store_true', help='Enables character level tokenization rather than word level tokenization')
parser.add_argument('--np_uint16', action='store_true', help='Forces use of np.int16 for encode/decode. \
                    This is automatic when vocab size exceeds 255 unique characters.')
args = parser.parse_args()
directory = args.fetch_dir
fetch = args.fetch
process = args.process
split_ratio = args.split_ratio # default is 0.8
# char level constant for Keras tokenizer
# set to True for character level tokenization,
# or False for word level tokenization
# see https://huggingface.co/docs/transformers/tokenizer_summary
char_level = args.char_level
output_file = 'data/ipsum/input.txt'

# Incompatible command line argument warnings
if fetch and directory is None:
    raise ValueError("Provide a dataset directory with --fetch_dir when fetching data with --fetch.")
if not (fetch or process):
    raise ValueError("No command given. Run with, at least, --fetch or --process.")
if not (0 < split_ratio < 1):
    raise ValueError("Invalid train test split ratio; must be between 0 and 1 exclusive.")

# Parse dataset
if fetch:
    # Check if directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' not found.")

    # Get all files with no extension or .txt extension
    all_files = glob.glob(os.path.join(directory, '*'))
    lorem_ipsum_files = [f for f in all_files if '.' not in os.path.basename(f)]
    lorem_ipsum_files.extend(glob.glob(os.path.join(directory, '*.txt')))

    iter, percent_parsed = 0, 0 # Num of lorem_ipsum files and percent parsed files

    with open(output_file, 'w') as outfile:
        for file in tqdm(lorem_ipsum_files, desc="Reading files", unit="file"):
            # Write to output_file with space in-between
            with open(file, 'r') as infile:
                try:
                    outfile.write(infile.read())
                    outfile.write('\n')
                except Exception as e:
                    print(f"Could not read file {file}. Error: {str(e)}")

    print(f'Read 100% of all files in {directory}')

# Prepare dataset for training
if process:
    # load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    with open(dataset_path, 'r') as f:
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
    ).tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    np.array(
        encode(data[int(len(data) * split_ratio):]),
        dtype=np.uint16 if use_uint16 else np.uint8
    ).tofile(os.path.join(os.path.dirname(__file__), 'test.bin'))

    metadata = {
        'vocab_size': vocab_size,
        'int_to_str': itos,
        'stf_to_int': stoi,
        'generation_date': datetime.now().isoformat(),
        'train_test_split': split_ratio,
        'output_file': output_file,
        'fetch_dir': directory,
        'char_level_tokenization': char_level,
        'uint16': use_uint16, # uint8 otherwise
    }

    # Save process metadata to a JSON file
    with open(os.path.join(os.path.dirname(__file__), 'metadata.json'), 'w') as file:
        json.dump(metadata, file)
    print("Processing complete")