
# Get only model args from args
def get_model_args(args, vocab_size=None):
    return {
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'block_size': args.block_size,
        'bias': args.bias,
        'vocab_size': vocab_size, # Set using data processing metadata
        'dropout': args.dropout_rate
    }
