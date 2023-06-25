"""
https://github.com/openai/gpt-2/blob/master/src/model.py
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from model.activation import swish
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, dropout_rate):
        super().__init__()
        # Transform input embeddings into (key, query, value) vectors for the attention mechanism
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Project the output of attention mechanism back to the original embedding dimension
        # This allows the model to mix information across different attention heads
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Dropout layers for regularization
        # They randomly set a fraction of inputs to zero during training to help prevent overfitting
        # attn_dropout is applied to the outputs of the attention mechanism
        # resid_dropout is applied to the residual connections in the transformer
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout_rate = dropout_rate
        # Layer normalization
        self.layer_norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Forward pass of multi-head attention mechanism."""
        # B Batch size: number of training examples per forward/backward pass
        # T Time steps (sequence length): sequence length of each training example
        # C Channel dimensions (embedding/feature dimensionality): number of features in input
        B, T, C = x.size()

        # Pass input tensor through linear layer then split parts
        query, key, value  = self.c_attn(x).split(self.n_embd, dim=2)

        # Use scaled dot product attention rather than additive attention
        # Both are similar, but the latter can be much slower
        y = torch.nn.functional.scaled_dot_product_attention(
            # Reshape tensors for multi-head attention with self.n_head heads
            # Each head will have reduced dimensionality C // self.n_head
            query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2),
            key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2),
            value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2),
            attn_mask=None,
            dropout_p=self.dropout_rate if self.training else 0,
            is_causal=True
        )

        # Gather all heads outputs in parallel
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Apply layer normalization
        y = self.layer_norm(y)

        # Return output projection
        return self.resid_dropout(self.c_proj(y))
    
class Block(nn.Module):
    """Layer of transformer's encoder with:
        - Self-attention mechanism
        - FFN
        - Layer normalization
        - Residual connections
    """
    def __init__(self, n_head, n_embd, dropout_rate, ffn_expand_factor=4, activation=swish):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, dropout_rate)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.c_fc = nn.Linear(n_embd, ffn_expand_factor * n_embd)
        self.c_proj = nn.Linear(ffn_expand_factor * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Feed forward pass"""
        x = x + self.attn(self.ln_1(x))
        ffn_output = self.c_fc(self.ln_2(x))
        ffn_output = self.activation(ffn_output)
        ffn_output = self.c_proj(ffn_output)
        ffn_output = self.dropout(ffn_output)
        return x + ffn_output

class GPT(nn.Module):
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, dropout_rate):
        super().__init__()
        # self.block_size = block_size
        # # Embedding layers
        # self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding = nn.Embedding(block_size, n_embd)
        # # Dropout layer
        # self.dropout_layer = nn.Dropout(dropout_rate)
        # # Transformer blocks
        # self.transformer_blocks = nn.ModuleList([Block(n_head, n_embd, dropout_rate) for _ in range(n_layer)])
        # # LayerNorm layer
        # self.final_layer_norm = nn.LayerNorm(n_embd)
        # # Linear layer
        # self.language_model_head = nn.Linear(n_embd, vocab_size)
        # # Weight tying between language model head and token embedding
        # self.token_embedding.weight = self.language_model_head.weight 
        # # Initialization of weights
        # self._initialize_weights()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout_rate),
            h = nn.ModuleList([Block(n_head, n_embd, dropout_rate) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        pos = torch.arange(0, idx.size()[1], dtype=torch.long, device=device).unsqueeze(0) # shape (1, idx.size()[1])

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
