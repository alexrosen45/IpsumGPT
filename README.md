# TeXGPT

Fetching dataset
python3 data/tex/fetch.py

Setup dataset
python3 data/tex/prepare.py

Training
python3 train.py config/train_tex.py

Using
python3 sample.py --out_dir=out-tex --start="matrix"

Check tensorflow installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"