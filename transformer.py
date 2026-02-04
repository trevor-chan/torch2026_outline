import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

# ——— RoPE positional embedding helpers —————————————————————————————
def precompute_rope_factors(max_seq_len, head_dim, device):
    """Precompute cos/sin for rotary embeddings once."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    # Shape: (max_seq_len, head_dim)
    return cos, sin

def apply_rope(x, cos, sin):
    """
    x: (B, H, L, D)
    cos, sin: (L, D)
    """
    return (x * cos[None, None, :, :]) + (rotate_half(x) * sin[None, None, :, :])

def rotate_half(x):
    """Helper for RoPE: rotate the last dim by chunking in half."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

# ——— SwiGLU feed-forward ———————————————————————————————————————
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # gate * up projection
        return self.fc_out(F.silu(self.fc1(x)) * self.fc2(x))

# ——— Single decoder block ————————————————————————————————————————
class FlexDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # combine QKV projection for simplicity
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ff = SwiGLU(dim, dim * 4)

        # precompute RoPE tables
        cos, sin = precompute_rope_factors(max_seq_len, self.head_dim, device='cpu')
        # register as buffers so they move with model.to(device)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        B, L, D = x.shape

        # 1) QKV and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each is (B, L, H, D_h)
        # bring seq first for RoPE: (B, H, L, D_h)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 2) apply RoPE
        cos = self.rope_cos[:L]
        sin = self.rope_sin[:L]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 3) causal via score_mod
        def causal_score_mod(score, b, h, qi, ki):
            return torch.where(qi >= ki, score, -float("inf"))

        # 4) FlexAttention call (returns (B, H, L, D_h))
        attn_out = flex_attention(q, k, v, score_mod=causal_score_mod)  # :contentReference[oaicite:0]{index=0}

        # 5) merge heads and project
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D_h)
        attn_out = attn_out.view(B, L, D)
        x = self.out_proj(attn_out)

        # 6) feed-forward
        x = x + self.ff(x)
        return x

# ——— The full decoder-only Transformer ——————————————————————————————
class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            FlexDecoderBlock(dim, num_heads, max_seq_len)
            for _ in range(num_layers)
        ])
        self.unembed = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        """
        tokens: LongTensor of shape (B, L)
        returns: logits (B, L, V)
        """
        x = self.token_emb(tokens)  # (B, L, D)
        for block in self.blocks:
            x = block(x)
        logits = self.unembed(x)
        return logits

class TransformerDiffusionModel(Transformer):
    """
    Adapts the Transformer model to work with diffusion model inputs
    (data, conditioning, and time), similar to the DiffusionModel class.
    
    This model embeds the conditioning vector and time step, prepends them
    to the data sequence, and processes the entire sequence through a transformer.
    Only the output corresponding to the data tokens is returned.
    """
    def __init__(
        self, 
        dim=256,               # Model embedding dimension  
        num_heads=4,           # Number of attention heads
        num_layers=4,          # Number of transformer layers
        max_seq_len=1024,      # Maximum sequence length
        output_dim=784         # Output dimension (784 for MNIST)
    ):
        # Initialize the parent Transformer with a minimal vocab size of 2
        # (we won't use token embeddings directly, but need to initialize properly)
        super(TransformerDiffusionModel, self).__init__(
            vocab_size=2,
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )
        
        # Replace token embedding with custom projections
        delattr(self, 'token_emb')
        delattr(self, 'unembed')
        
        # Create data embedding layer (linear projection)
        self.data_proj = nn.Linear(1, dim)  # Project each scalar to embedding dim
        
        # Create embeddings for conditioning and time
        self.cond_embedding = nn.Linear(10, dim)  # Embed one-hot conditioning vector
        self.time_embedding = nn.Linear(1, dim)   # Embed time scalar
        
        # Output projection to get back to target data dimension
        self.output_proj = nn.Linear(dim, output_dim)
        
    def forward(self, x, conditioning, t):
        """
        Forward pass with diffusion-style inputs.
        
        Args:
            x: Input data tensor [batch_size, data_dim]
            conditioning: Conditioning labels [batch_size, 10] (one-hot encoded)
            t: Timestep tensor [batch_size]
            
        Returns:
            Model prediction [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Flatten the input data if needed
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
        
        # Reshape data as batch_size x seq_len x 1 for projection
        x_reshaped = x.unsqueeze(-1)  # [batch_size, data_dim, 1]
        
        # Project each scalar in the data to embedding dimension
        x_embedded = self.data_proj(x_reshaped)  # [batch_size, data_dim, dim]
        
        # Embed conditioning and time
        t_embed = self.time_embedding(t.view(-1, 1))  # [batch_size, dim]
        cond_embed = self.cond_embedding(conditioning)  # [batch_size, dim]
        
        # Reshape conditioning and time to add sequence dimension
        t_embed = t_embed.unsqueeze(1)      # [batch_size, 1, dim]
        cond_embed = cond_embed.unsqueeze(1)  # [batch_size, 1, dim]
        
        # Concatenate along sequence dimension: [time, conditioning, data]
        # Result: [batch_size, 2+data_dim, dim]
        sequence = torch.cat([t_embed, cond_embed, x_embedded], dim=1)
        
        # Pass through transformer blocks (reusing parent class blocks)
        for block in self.blocks:
            sequence = block(sequence)
        
        # Extract only the data portion (exclude first 2 tokens for time and conditioning)
        data_output = sequence[:, 2:, :]  # [batch_size, data_dim, dim]
        
        # Project back to output dimension using the mean of the sequence representations
        data_mean = data_output.mean(dim=1)  # [batch_size, dim]
        output = self.output_proj(data_mean)  # [batch_size, output_dim]
        
        return output
    
    def shard(self, mp_policy: bool = True):
        """
        Shard the model using FSDP for distributed training.
        """
        if mp_policy:
            from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)
            
        fully_shard(self.output_proj, mp_policy=mp_policy, reshard_after_forward=True)
        fully_shard(self, mp_policy=mp_policy, reshard_after_forward=True)
