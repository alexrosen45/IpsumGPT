# TeXGPT

Fetching dataset
python3 data/ipsum/fetch.py

Setup dataset
python3 data/ipsum/prepare.py

Training
python3 train.py --out_dir="out-ipsum"

Using
python3 sample.py --out_dir=out-ipsum --start="Lorem"

Check tensorflow installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"