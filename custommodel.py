import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledRoPE(nn.Module):
    def __init__(self, dim, theta=10000.0, use_scaled_rope=True):
        super().__init__()
        self.theta = theta
        self.use_scaled_rope = use_scaled_rope

    def forward(self, x, seq_len):
        """Apply RoPE (Rotary Positional Encoding) to input embeddings."""
        half_dim = x.shape[-1] // 2
        freqs = torch.arange(half_dim, dtype=torch.float32, device=x.device)
        freqs = self.theta ** (-freqs / half_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device)

        # Compute sine and cosine positions
        sin, cos = torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)
        if self.use_scaled_rope:
            sin, cos = sin * math.sqrt(half_dim), cos * math.sqrt(half_dim)

        # Apply RoPE
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_weights, dim=-1)
        context = torch.einsum('bhts,bshd->bthd', attn_probs, v)

        # Concatenate heads and project output
        context = context.contiguous().view(batch_size, seq_len, dim)
        return self.out_proj(context)


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class TransformerLayer(nn.Module):
    def __init__(self, dim, ffn_dim, n_heads, n_kv_heads, norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = MultiHeadAttention(dim, n_heads, n_kv_heads)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn = FeedForwardNetwork(dim, ffn_dim)

    def forward(self, x):
        # Apply attention and residual connection
        x = x + self.attn(self.norm1(x))
        # Apply feed-forward network and residual connection
        x = x + self.ffn(self.norm2(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["dim"])
        self.layers = nn.ModuleList([
            TransformerLayer(
                dim=config["dim"],
                ffn_dim=int(config["dim"] * config["ffn_dim_multiplier"]),
                n_heads=config["n_heads"],
                n_kv_heads=config["n_kv_heads"],
                norm_eps=config["norm_eps"]
            )
            for _ in range(config["n_layers"])
        ])
        self.norm = nn.LayerNorm(config["dim"], eps=config["norm_eps"])
        self.rope = ScaledRoPE(config["dim"], config["rope_theta"], config["use_scaled_rope"])

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        x = self.token_embedding(input_ids)

        # Apply RoPE to input embeddings
        x = self.rope(x, seq_len)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        return self.norm(x)


'''# Example usage
config = {
    "dim": 2048,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 256,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 16,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "vocab_size": 128256
}

model = LlamaModel(config)
input_ids = torch.randint(0, config["vocab_size"], (2, 64))  # Example batch of input IDs
output = model(input_ids)
print(output.shape)  # Expected: (2, 64, 2048)'''
