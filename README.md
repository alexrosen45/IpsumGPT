# IpsumGPT
A modular implementation of a causal transformer with scaled dot-product attention, layer normalization, and residual connections.

# Prerequisites
- Text dataset
- Nvidia GPU with CUDA installed
- Anaconda (or another virtual environment for package management)

# Setup
> It should be straightforward to modify this setup yourself for use without Anaconda

#### Clone this repository
```sh
git clone https://github.com/alexrosen45/IpsumGPT
```
#### Create and activate conda environment
```sh
cd IpsumGPT
conda env create -f environment.yml
conda activate IpsumGPT
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
#### Install PyTorch in your conda environment
Follow the setup instructions [here](https://pytorch.org/). Install version 2.0 or above with conda and CUDA 11.7 or above. If your are unsure of which CUDA version to pick, run `nvidia-smi` in your conda environment and pick the version closest to 'CUDA Version' (it shouldn't need to be exactly the same).

# Usage
#### Add a dataset
From the project's home directory, create a folder named `datasets` with a subdirectory `your-dataset-name-here`. Metadata, training, and testing files will be stored in this subdirectory, so it's important to maintain the recommended file structure. Inside the subdirectory `your-dataset-name-here`, create a new folder `your-raw-data-name-here` and upload your dataset as any assortment of .txt files within this new folder.

#### Data Processing
Suppose your raw data is located in the directory `datasets/ipsum/ipsum-dataset`. Fetch, tokenize, and process your raw data for training with
```sh
python3 process.py
  --data_dir="datasets/ipsum"
  --data_folder="ipsum-dataset"
  --split_ratio=0.8
```
#### Model training
Pick a name for your model and train it with
```sh
python3 train.py
  --out_dir="your-model-name-here"
  --data_dir="datasets/ipsum"
```
#### Text generation
After training, generate text using your model with
```sh
python3 generate.py
  --model="your-model-name-here"
```
