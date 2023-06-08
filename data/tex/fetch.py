"""fetch data from arxiv-2002 dataset and
parse it into a single input.txt file
"""

import os
import glob

directory = 'data/tex/arxiv-2002'

# If changed, makes changes in prepare.py too
output_file = 'data/tex/input.txt'

# Get all files with no extension or .tex extension
all_files = glob.glob(os.path.join(directory, '*'))
latex_files = [f for f in all_files if '.' not in os.path.basename(f)]
latex_files.extend(glob.glob(os.path.join(directory, '*.tex')))

iter, percent_parsed = 1, 0 # Num of latex files and percent parsed files
print(f'Parsing {len(latex_files)} files...')

with open(output_file, 'w') as outfile:
    for latex_file in latex_files:
        # Update progress in console
        if round(len(latex_files)/iter) > percent_parsed:
            print(f'{percent_parsed}% complete')
            percent_parsed += 1
        iter += 1

        # Write to output_file with space in-between
        with open(latex_file, 'r') as infile:
            try:
                outfile.write(infile.read())
                outfile.write('\n')
            except Exception as e:
                print(f"Could not read file {latex_file}. Error: {str(e)}")

print(f'parsing complete\nread {percent_parsed}% of all files')
