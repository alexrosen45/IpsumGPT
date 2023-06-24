import os
import argparse
import time
import math
from contextlib import nullcontext
import sys
import json
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import train_args
from models.model import GPTConfig, GPT


# Get command line and default arguments and perform checks
args = train_args.get_args()
train_args.arg_checks(args)
train_args.arg_warnings(args)

# Directory management
out_dir = args.out_dir # default is 'out'
override_dir = args.override_dir
data_dir = args.data_dir # default is 'data/ipsum'

# Model evaluation parameters
# Evaluate the model every eval_interval iterations
eval_interval = 250 # keep frequent because we'll overfit
# How many iterations to preform during each evaluation
eval_iters = 200
eval_only = False # if True, script exits right after the first eval

# Training parameters
# Gradient accumulation helps simulate a larger batch size by accumulating gradients
# from multiple small batches weight update. Increase when facing memory issues.
# See https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa
gradient_accumulation_steps = args.gradient_accumulation_steps # Default is 1
batch_size = args.batch_size # Training examples per iteration; default is 64
block_size = args.block_size # Max previous tokens of context; default is 256
force_save_checkpoint = args.force_save # default is False
min_loss = 1e10 # for optional checkpoint saving

# Model parameters
# Number of transformer layers, each with self-attention
# mechanism and a feed forward neural network
n_layer = args.n_layer
# Number of attention heads in each self-attention mechanism
n_head = args.n_head
# Dimensionality of input/output of each transformer layer
# Higher dimensionality can increase ability to recognize more complex patterns
n_embd = args.n_embd
# Dropout rate (for regularization)
dropout = args.dropout_rate
# Use bias terms in linear transformations inside transfomer
# layers and layer normalization.
bias = args.bias

# AdamW optimizer parameters
# See https://paperswithcode.com/method/adamw
learning_rate = args.learning_rate
max_iters = args.max_iters # Total number of training iterations
weight_decay = args.weight_decay # Weight decay for L2 regularization
beta1 = args.beta1 # AdamW beta_1
beta2 = args.beta2 # AdamW beta_2

# Other parameters
log_interval = 10 # for printing training progress to console
create_new_model = not args.resume # default is True (don't resume)
device = 'cpu' if args.cpu else 'cuda'
iter_num = 0 # Number of training iterations that have passed

# Create out directory if it does not exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def init_ddp_environment():
    # initalize DDP environment
    init_process_group(backend='nccl')
    try:
        # rank: unique identifier assigned to each process in 
        # distributed setup. N processes -> 0 to N-1 unique ranks
        ddp_rank = int(os.environ['RANK'])
        # local rank: similar to rank, but for multiple processes
        # running on a single "local" GPU
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        # world size: total number of processes across whole DDP setup
        ddp_world_size = int(os.environ['WORLD_SIZE'])
    except KeyError as e:
        raise RuntimeError(f"Environment variable {e.args[0]} not set")

    # set device to GPU corresponding to local rank
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    return ddp_rank, ddp_local_rank, ddp_world_size, device

# check if DDP is enabled
try:
    ddp = int(os.environ.get('RANK', -1)) != -1
except KeyError as e:
    raise RuntimeError(f"Environment variable {e.args[0]} not set")

if ddp:
    ddp_rank, ddp_local_rank, ddp_world_size, device = init_ddp_environment()

    # generate unique random seed for each process
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # single GPU setup; only one process
    # rank is always 0 and no seed offset
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Calculate tokens per iteration
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per training iteration: {tokens_per_iter:,}")

# Set PyTorch seed
torch.manual_seed(1337 + seed_offset) # for reproducible results

# Enable TensorFloat-32 (TF32) for matrix multiplication and cuDNN operations
# TF32: data type introduced by NVIDIA in their Ampere architecture GPUs to
# speed up computations while maintaining enough precision for DL.
# Try FP16 or FP32 if not using an Ampere GPU
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
# Enable automatic mixed precision if GPU is being used
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# uses DataLoader from torch for automatic batching, shuffling, and multi-threaded loading
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        return x, y

# Get necessary metadata
with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
    metadata = json.load(f)
vocab_size = metadata['vocab_size']
use_uint16 = metadata['uint16'] # for encoding/decoding

train_data = np.memmap(os.path.join(data_dir, 'train.bin'),
    dtype=np.uint16 if use_uint16 else np.uint8,
    mode='r'
)
test_data = np.memmap(os.path.join(data_dir, 'test.bin'),
    dtype=np.uint16 if use_uint16 else np.uint8,
    mode='r'
)

# create datasets and data loaders
# iterate over a data loader to get a batch of data
train_dataset = TextDataset(train_data, block_size)
test_dataset = TextDataset(test_data, block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if create_new_model:
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
else:
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    min_loss = checkpoint['min_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint = None # free up memory

# compile model
print("compiling the model... (takes a ~minute)")
unoptimized_model = model
model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('test', test_loader)]:
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(loader):
            if k >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
train_iter = iter(train_loader)  # create an iterator for the training data loader
X, Y = next(train_iter)  # fetch the first batch
X, Y = X.to(device), Y.to(device)  # move the batch data to the correct device
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # evaluate loss on train tests sets and write to checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
        if losses['test'] < min_loss or force_save_checkpoint:
            min_loss = losses['test']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': min_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        try:
            X, Y = next(train_iter)
        except StopIteration:
            # if StopIteration is raised, reinitialize the data loader
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
        X, Y = X.to(device), Y.to(device)  # move the batch data to the correct device
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()