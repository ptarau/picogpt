# model.py
# Core model: tiny GPT with RoPE (auto-growing cache)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    RoPE cos/sin cache that grows on demand.
    head_dim must be even (we rotate pairs).
    """
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        half_dim = head_dim // 2
        inv_freq = base ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        freqs = t * self.inv_freq.unsqueeze(0)                           # (T,D/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    @torch.no_grad()
    def _extend_to(self, new_max_seq_len: int, device, dtype):
        if new_max_seq_len <= self.max_seq_len:
            return
        t = torch.arange(self.max_seq_len, new_max_seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        freqs = t * self.inv_freq.to(device).unsqueeze(0)
        cos_new = freqs.cos().to(dtype=dtype)
        sin_new = freqs.sin().to(dtype=dtype)
        self.cos_cached = torch.cat([self.cos_cached.to(device=device, dtype=dtype), cos_new], dim=0)
        self.sin_cached = torch.cat([self.sin_cached.to(device=device, dtype=dtype), sin_new], dim=0)
        self.max_seq_len = new_max_seq_len

    def get_cos_sin(self, T: int, device, pos_offset: int = 0, dtype=None):
        end = pos_offset + T
        use_dtype = dtype or torch.float32
        if end > self.max_seq_len:
            self._extend_to(end, device=device, dtype=use_dtype)
        cos = self.cos_cached[pos_offset:end].to(device=device, dtype=use_dtype)
        sin = self.sin_cached[pos_offset:end].to(device=device, dtype=use_dtype)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotate last-dim pairs by RoPE. Shapes:
      x:   (B,T,D) even D
      cos: (T,D/2), sin: (T,D/2)
    """
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    cos = cos.unsqueeze(0)  # (1,T,D/2)
    sin = sin.unsqueeze(0)  # (1,T,D/2)
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos
    x_out = torch.empty_like(x)
    x_out[..., ::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd
    return x_out


class Head(nn.Module):
    """Single causal self-attention head with RoPE on Q/K."""
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float, rope: RotaryEmbedding):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v  # (B,T,hs)


class MultiHead(nn.Module):
    """Multi-head self-attention; shares a RoPE cache across heads."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, rope_base: float):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        assert head_size % 2 == 0, "head_size must be even for RoPE"
        self.rope = RotaryEmbedding(head_dim=head_size, max_seq_len=block_size, base=rope_base)
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout, self.rope) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        T = x.size(1)
        cos, sin = self.rope.get_cos_sin(T, x.device, pos_offset=pos_offset, dtype=x.dtype)
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedFwd(nn.Module):
    """Position-wise MLP"""
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(),
            nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer block with RoPE-aware attention."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, rope_base: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHead(n_embd, n_head, block_size, dropout, rope_base)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff  = FeedFwd(n_embd, dropout)
    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        x = x + self.sa(self.ln1(x), pos_offset=pos_offset)
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Minimal GPT-style LM with RoPE (no additive position embeddings)."""
    def __init__(self, V: int, n_embd: int, n_head: int, n_layer: int, block_size: int, dropout: float, rope_base: float):
        super().__init__()
        self.token_emb = nn.Embedding(V, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout, rope_base) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V)
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, pos_offset: int = 0):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.token_emb(idx)  # (B,T,C)
        for blk in self.blocks:
            x = blk(x, pos_offset=pos_offset)
        x = self.ln_f(x)
        logits = self.head(x)    # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0, top_k: int | None = None,
                 device: str = "cpu", amp_dtype=None):
        for _ in range(max_new_tokens):
            total_len = idx.size(1)
            idx_cond = idx[:, -self.block_size:]
            T = idx_cond.size(1)
            pos_offset = total_len - T
            if device == "cuda" and amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = self(idx_cond, pos_offset=pos_offset)
            else:
                logits, _ = self(idx_cond, pos_offset=pos_offset)

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
