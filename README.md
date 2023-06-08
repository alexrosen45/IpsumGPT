# TeXGPT

Fetching dataset
python3 data/tex/fetch.py

Setup dataset
python3 data/tex/prepare.py

Training
python3 train.py config/train_tex.py

Using
python3 sample.py --out_dir=out-tex --start="matrix"