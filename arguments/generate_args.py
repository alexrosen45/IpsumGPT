import argparse
import os


# Parse optional command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Fetching and/or processing parameters")
    parser.add_argument('--model', type=str, default="out", help='Model to generate from (ex. --model="out")')
    parser.add_argument('--start', type=str, default="Lorem" ,help='Start of text (ex. --start="Lorem Ipsum dolor...")')
    return parser.parse_args()

# Perform argument checks to raise helpful errors
def arg_checks(args):
    # Check if directories exist
    if not os.path.exists('models/' + args.model):
        raise NotADirectoryError(f"Directory {'models/' + args.model} does not exist.")
