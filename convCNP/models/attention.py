import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadCrossAttentionWrapper(nn.Module):
    """Wrapper for multi-head cross attention supporting separate key/value inputs."""
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
    
    def forward(self, query, key, value):
        """
        Args:
            query: (B, M, embed_dim) - queries from target
            key: (B, N, embed_dim) - keys from context
            value: (B, N, embed_dim) - values from context (can differ from key!)
        """
        attn_output, attn_weights = self.attention(
            query, key, value,
            need_weights=True, average_attn_weights=True
        )
        self.last_weights = attn_weights.detach()
        return attn_output


class LaplaceAttention(nn.Module):
    """Laplace kernel attention for spatial relationships."""
    def __init__(self, init_temp=1.0, min_temp=1e-3, max_temp=10.0, p_norm=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.p_norm = float(p_norm)
        if self.p_norm <= 0:
            raise ValueError("p_norm must be greater than 0 for LaplaceAttention.")

    def forward(self, query, key, value):
        """
        Paper-faithful Laplace attention.
        Args:
            query: s* (target spatial coords)
            key: s (context spatial coords)  
            value: w = MLP(s, y) (encoded context)
        """
        dists = torch.cdist(query, key, p=self.p_norm)
        temp = torch.exp(self.log_temp).clamp(min=self.min_temp, max=self.max_temp)
        weights = F.softmax(-dists / (temp + 1e-8), dim=-1)
        self.last_weights = weights.detach()   # (B, M_target, N_context) — ADD THIS
        return torch.bmm(weights, value)