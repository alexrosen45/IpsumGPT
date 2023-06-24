import argparse
import os
import torch


# Parse optional command line arguments
def get_args():
    # We use the same argument name as global variable name for saving checkpoints (except for resume)
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--out_dir', type=str, default='out', help='Out directory of model (ex. --out_dir="out-ipsum")')
    parser.add_argument('--data_dir', type=str, default='data/ipsum', help='Data directory (ex. --out_dir="data/ipsum")')
    parser.add_argument('--override_dir', action='store_true', help='Override current training folder.')
    parser.add_argument('--force_save', action='store_true', help='Saves every training checkpoint regardless of loss improvement.')
    parser.add_argument('--resume', action='store_true', help='Resumes training from last checkpoint.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps\
        (ex. --gradient_accumulation_steps=1). Increase this value gradually when facing memory issues.')
    parser.add_argument('--block_size', type=int, default=256, help='Max previous tokens of context (ex. --block_size=256)')
    parser.add_argument('--batch_size', type=int, default=64, help='Training examples per iteration (ex. --batch_size=64)')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers (ex. --n_layer=8)')
    parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads per self-attention mechanism (ex. --n_head=8)')
    parser.add_argument('--n_embd', type=int, default=384, help='Dimensionality of input and output of each transformer\
        layer (ex. --n_embd=400). Increasing dimensionality can improve recognition of complex patterns.')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularization (ex. --dropout_rate=0.25)')
    parser.add_argument('--bias', action='store_true', help='Use bias in transformer layers and layer normalization.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (ex. --learning_rate=1e-6)')
    parser.add_argument('--max_iters', type=int, default=5000, help='Training iterations (ex. --max_iters=5500)')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay for L2 regularization (ex. --weight_decay=1e-2). \
        Set to 0 for no regularization.')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW optimizer beta_1 hyperparameter (ex. --beta1=0.9)')
    parser.add_argument('--beta2', type=float, default=0.99, help='AdamW optimizer beta_2 hyperparameter (ex. --beta2=0.999)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU with CUDA.')
    return parser.parse_args()

# Perform argument checks to raise helpful errors
def arg_checks(args):
    # Overriding existing directory without --resume or --override_dir
    if os.path.exists(args.out_dir) and not args.override_dir and not args.resume:
        raise FileExistsError(f"Directory {args.out_dir} exists and overriding is not permitted. Use --override_dir to override.")
    # Resuming from non-existent directory
    if args.resume and not os.path.exists(args.out_dir):
        raise FileExistsError(f"Cannot resume from training directory {args.out_dir} that does not exist. Try specifying\
             this directory with --out_dir.")
    # Data directory doesn't exist
    if not os.path.exists(args.data_dir):
        raise NotADirectoryError(f"Directory {args.data_dir} does not exist.")
    # Gradient accumulation steps range
    if not (1 <= args.gradient_accumulation_steps <= 100):
        raise ValueError("Gradient accumulation steps should be an integer between 1 and 100 inclusive.")
    # Positive arguments
    for arg in [args.block_size, args.batch_size, args.n_layer, args.n_head, args.n_embd, args.max_iters]:
        if not arg > 0:
            raise ValueError(f"{arg} should be a positive integer.")
    # Dropout rate range
    if not (0 <= args.dropout_rate <= 0.99):
        raise ValueError("Dropout rate should be between 0 and 0.99 inclusive.")
    # Learning rate range
    if not (1e-6 <= args.learning_rate <= 0.1):
        raise ValueError("Learning rate should be between 1e-6 and 0.1 inclusive.")
    # Weight decay range
    if not (0 <= args.weight_decay <= 1):
        raise ValueError("Weight decay should be between 0 and 1 inclusive.")
    # AdamW beta_1 range
    if not (0.8 <= args.beta1 <= 0.999):
        raise ValueError("AdamW beta1 should be between 0.8 and 0.999 inclusive in practice. In rare circumstances, it's ok\
         to modify this range in train_args.py.")
    # AdamW beta_2 range
    if not (0.9 <= args.beta2 <= 0.9999):
        raise ValueError("AdamW beta1 should be between 0.9 and 0.9999 inclusive in practice. In rare circumstances, it's ok\
             to modify this range in train_args.py.")
    # CUDA availability check
    if not args.cpu and not torch.cuda.is_available():
        raise Exception("CUDA is not available but the program is set to use a GPU. You can run the program on your \
            CPU instead with --cpu.")  

# Print argument warnings to console
def arg_warnings(args):
    # CPU over GPU with CUDA warning
    if args.cpu:
        print("WARNING: Compiling and training the model on your CPU may take a considerable amount of time.")