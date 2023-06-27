# IpsumGPT
A modular implementation of a causal transformer with scaled dot-product attention, layer normalization, and residual connections.

Fetching dataset
python3 data/ipsum/fetch.py

Setup dataset
python3 process.py

Setup own:
python3 process.py --data_dir="datasets/ipsum" --data_folder="ipsum-dataset"

Training
python3 train.py --out_dir="out-ipsum"

Using
python3 generate.py --model="out-ipsum"

Check tensorflow installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
