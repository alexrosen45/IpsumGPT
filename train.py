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

from models.model import GPTConfig, GPT

out_dir = 'out'
override_dir = True # allow replacement of current out_dir file
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # just for printing to console

eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # save checkpoint regardless of improvement
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # override via command line if you like
wandb_project = 'ipsum'
wandb_run_name = 'mini-gpt'
# data
dataset = 'ipsum'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2 # lots of finetuning
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 5000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# # -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# # -----------------------------------------------------------------------------

# parse optional command line arguments
parser = argparse.ArgumentParser(description="out directory")
parser.add_argument('--out_dir', type=str, help='Out directory of model (ex. --out_dir="out-ipsum")')
args = parser.parse_args()
if args.out_dir is not None:
    out_dir = args.out_dir

def init_ddp_environment():
    # initalize DDP environment
    init_process_group(backend=backend)
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

if master_process:
    if override_dir or not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    else:
        print(f'The directory {out_dir} does not exist.' if override_dir else f'Cannot override {out_dir}. Set override_dir=True.')
        sys.exit(1)
if not torch.cuda.is_available():
    print("WARNING: CUDA not available with torch, using CPU instead")
torch.manual_seed(1337 + seed_offset) # for reproducible results
# enable TensorFloat-32 (TF32) for matrix multiplication and cuDNN operations
# TF32: data type introduced by NVIDIA in their Ampere architecture GPUs to
# speed up computations while maintaining enough precision for deep learning.
# try FP16 or FP32 if not using an Ampere GPU
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# enable automatic mixed precision if GPU is being used
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

# Load data
data_dir = os.path.join('data', dataset)

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

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
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
    best_loss = checkpoint['best_loss']
else:
    print("ERROR: Could not load model from scratch or resuming training.")
    sys.exit(1)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
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


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

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
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate loss on train tests sets and write to checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "test/loss": losses['test'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['test'] < best_loss or always_save_checkpoint:
            best_loss = losses['test']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_loss,
                    'config': config,
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

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
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
