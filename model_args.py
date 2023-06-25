
# Get only model args from args
# args is usually command line arguments or old parameters from checkpoint
def get_model_args(args, vocab_size=-1):
    # 0 vocab size corresponds to reading it from args instead of metadata
    # Read as a dictionary or another data structure with dot notation
    # TODO: if isinstance(args, dict):
    return {
        'block_size': args.block_size,
        'vocab_size': args.vocab_size if vocab_size == -1 else vocab_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'dropout_rate': args.dropout_rate
    }
