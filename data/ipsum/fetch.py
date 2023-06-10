"""fetch data from arxiv-2002 dataset and
parse it into a single input.txt file
"""

import os
import glob

directory = 'data/ipsum/lorem-ipsum-dataset'

# If changed, makes changes in prepare.py too
output_file = 'data/ipsum/input.txt'

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
