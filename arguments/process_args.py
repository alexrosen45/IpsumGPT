import argparse
import os


# Parse optional command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Fetching and/or processing parameters")
    parser.add_argument('--data_dir', type=str, default='datasets/ipsum', help='Dataset home directory (ex. --data_dir="datasets/ipsum")')
    parser.add_argument('--data_folder', type=str, default='ipsum-dataset', help='Data folder in dataset home directory (ex. --data_folder="ipsum-dataset")')
    parser.add_argument('--no_fetch', action='store_true', help="Don't fetch data into input.txt and process the contents of input.txt only.")
    parser.add_argument('--split_ratio', type=float, default = 0.8, help='Train test split ratio (ex. --split-ratio=0.8 trains on 80 percent of dataset)')
    parser.add_argument('--char_level', action='store_true', help='Enables character level tokenization rather than word level tokenization')
    parser.add_argument('--np_uint16', action='store_true', help='Forces use of np.int16 for encode/decode. \
                        This is automatic when vocab size exceeds 255 unique characters.')
    return parser.parse_args()

# Perform argument checks to raise helpful errors
def arg_checks(args):
    # Check if directories exist
    if not os.path.exists(args.data_dir):
        raise NotADirectoryError(f"Directory {args.data_dir} does not exist.")
    if not os.path.exists(args.data_dir + '/' + args.data_folder) and not args.no_fetch:
        raise NotADirectoryError(f"Data folder directory {args.data_dir + '/' + args.data_folder} does not exist.")
    # Split ratio range
    if not (0.5 <= args.split_ratio <= 0.95):
        raise ValueError("Split ratio must be between 0.5 and 0.95 inclusive. You can bypass this, but it is not recommended.")
