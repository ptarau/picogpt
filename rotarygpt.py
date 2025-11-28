"""
mini_gpt_bpe_tiktoken_cuda_rope.py
Tiny GPT with tiktoken BPE. Auto-selects CUDA (AMP), MPS, or CPU.
Positional encoding: **Rotary Positional Embeddings (RoPE)** applied to Q/K.

Why RoPE?
---------
- Injects position via a rotation in query/key space (no additive pos vectors).
- Naturally supports sliding windows: just increase the position offset.
- Works head-wise; requires even head_size (we use 96 embd / 3 heads => 32).

Main changes vs. additive positions:
- Removed nn.Embedding pos_emb.
- Added a small RotaryEmbedding helper that prepares cos/sin caches.
- Heads receive (cos, sin) for the current T and offset; apply rotation to Q/K.
- During generation we pass a **pos_offset** reflecting how many tokens preceded
  the current (possibly truncated) context window.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 0) Device & AMP configuration
# ============================================================
if torch.cuda.is_available():
    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16  # GradScaler only for fp16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    amp_dtype = None
    use_scaler = False
else:
    device = "cpu"
    amp_dtype = None
    use_scaler = False

torch.manual_seed(0)
print(
    f"[info] using device={device}" + (f", amp_dtype={amp_dtype}" if amp_dtype else "")
)

# ============================================================
# 1) Tokenizer (tiktoken BPE)
# ============================================================
try:
    import tiktoken
except ImportError as e:
    raise SystemExit("tiktoken is required. Install with: pip install tiktoken") from e

enc = tiktoken.get_encoding("cl100k_base")
V = enc.n_vocab


def encode(s: str) -> torch.Tensor:
    return torch.tensor(enc.encode(s), dtype=torch.long)


def decode(ids) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return enc.decode(ids)


# ============================================================
# 2) Corpus (single stream)
# ============================================================
data_text = """
Reinforcement learning has quietly become the glue that helps large language models get better at code.
Supervised pretraining teaches syntax and common idioms, but RL shapes behavior with feedback that reflects what we actually want:
correctness, robustness, speed, and adherence to tool conventions.

One simple loop illustrates the idea. The model proposes a function. A sandbox runs unit tests, type checks, and linters.
The results are distilled into a scalar reward: passing tests earns positive feedback, failing tests are negative, flaky tests get reduced credit.
The model updates its policy to increase the probability of solutions that pass while reducing the probability of brittle or hallucinated code paths.
Over time, the distribution shifts toward code that actually works.

Reward models help when ground-truth signals are sparse. Instead of a raw pass/fail, we train a learned scorer that grades partial progress:
Did the code import the right modules? Did it honor the signature? Are edge cases handled? Is complexity reasonable?
These fine-grained judgments stabilize training and guide exploration toward promising regions of the solution space.

Self-play and curriculum learning matter too. Early on, the environment samples easy tasks—string transforms, simple math, basic file IO.
As competence grows, the tasks escalate: concurrency, stateful services, numerical stability, GPU kernels.
The model learns to use tools—package installers, REPLs, debuggers—and receives reward for shorter iteration cycles and fewer dependency conflicts.
Tool use is itself a policy: knowing when to run tests, when to search docs, and when to refactor is part of the optimization problem.

Critically, RL does not replace reasoning; it amplifies it. Search operators—beam search, MCTS-like reranking, or speculative execution—generate diverse candidate programs.
Executors provide ground truth by running those programs, and RL tunes the policy to prefer candidates that generalize beyond the seen tests.
With iterative evaluation, the system discovers reusable patterns—property-based tests, input validation layers, timeout guards—that raise success rates on novel tasks.

RLAIF (reinforcement learning from AI feedback) broadens coverage. A committee of models acts as reviewers, proposing tests, spotting undefined behavior,
and identifying invariants the code should satisfy. While noisy, this feedback is cheap and scalable, and reward modeling filters the signal.
When combined with occasional human audits and public benchmarks, the result is a curriculum that keeps getting harder exactly where the model is weakest.

The punchline is that coding becomes a real environment with actions, observations, and delayed rewards.
RL aligns the model with executable truth, not just textual plausibility.
That is why RL-tuned coders improve fastest where it matters most: passing new tests, integrating unfamiliar libraries, and fixing their own mistakes.
"""

X_all = encode(data_text)
N = X_all.numel()
assert N > 200, f"Corpus too short after BPE: {N} tokens. Add more text."

# ============================================================
# 3) Hyperparameters (small for speed)
# ============================================================
block_size = 64  # context length (RoPE caches are created up to this T)
n_embd = 96  # divisible by n_head
n_head = 3  # => head_size = 32 (must be even for RoPE pairwise rotation)
n_layer = 2
dropout = 0.1
batch_size = 48
train_steps = 600
learning_rate = 3e-3
rope_base = 10000.0  # RoPE frequency base (typical value)


# ============================================================
# 4) Batching
# ============================================================
def get_batch(B=batch_size):
    """
    Return B random (x, y) pairs of length T=block_size from the token stream.
    Note: during training, we start every subsequence at position offset 0
    (i.e., we don't carry absolute positions across batches).
    """
    ix = torch.randint(0, N - block_size - 1, (B,))
    x = torch.stack([X_all[i : i + block_size] for i in ix])
    y = torch.stack([X_all[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 5) Rotary Positional Embeddings (helper)
# ============================================================
class RotaryEmbedding(nn.Module):
    """
    Precomputes cos/sin caches for RoPE and returns slices for the current
    sequence length T and a given position offset.

    head_dim must be even. We use pairwise rotations on the last dimension.

    References: Su et al., "RoFormer", and GPT-NeoX/LLAMA implementations.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Frequencies: base^{-2i/d} for i in [0..d/2-1]
        half_dim = head_dim // 2
        inv_freq = base ** (-torch.arange(0, half_dim, dtype=torch.float32) / half_dim)

        # Precompute angles for positions 0..max_seq_len-1
        t = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        freqs = t * inv_freq.unsqueeze(0)  # (T, half_dim)
        # Register as buffers to move with the module across devices
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def get_cos_sin(self, T: int, device, pos_offset: int = 0):
        """
        Get cos/sin for positions [pos_offset .. pos_offset+T-1].

        Returns:
          cos, sin: (T, half_dim) on the requested device.
        """
        start = pos_offset
        end = pos_offset + T
        # Extend if needed (rare for demos; can add growth if you wish)
        assert (
            end <= self.max_seq_len
        ), "Requested positions exceed RoPE cache; raise max_seq_len"
        return (
            self.cos_cached[start:end].to(device),
            self.sin_cached[start:end].to(device),
        )


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding to last-dim pairs of x.

    Args
    ----
    x   : (B, T, D) where D is head_size (even)
    cos : (T, D/2)
    sin : (T, D/2)

    Returns
    -------
    x_rot : (B, T, D)
    """
    # Split into even/odd parts
    x_even = x[..., ::2]  # (B, T, D/2)
    x_odd = x[..., 1::2]  # (B, T, D/2)

    # Broadcast cos/sin over batch
    cos = cos.unsqueeze(0)  # (1, T, D/2)
    sin = sin.unsqueeze(0)  # (1, T, D/2)

    # Complex-like rotation: [a, b] -> [a*cos - b*sin, a*sin + b*cos]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    # Re-interleave even/odd back into the last dimension
    x_out = torch.empty_like(x)
    x_out[..., ::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd
    return x_out


# ============================================================
# 6) Transformer components with RoPE in attention
# ============================================================
class Head(nn.Module):
    """
    Single **causal self-attention** head with **RoPE** applied to Q/K.
    """

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float,
        rope: RotaryEmbedding,
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.rope = rope  # shared rotary cache (per head size)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        cos, sin: (T, head_size/2) — specific to this head's size.
        """
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Apply RoPE to Q and K
        q = apply_rope(q, cos, sin)  # (B, T, hs)
        k = apply_rope(k, cos, sin)  # (B, T, hs)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, hs)
        return out


class MultiHead(nn.Module):
    """
    Multi-head self-attention. Shares a single RotaryEmbedding (per head size)
    across heads, and slices the correct (cos, sin) for current T and offset.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        assert head_size % 2 == 0, "head_size must be even for RoPE"

        # One RoPE cache shared by all heads of this size
        self.rope = RotaryEmbedding(
            head_dim=head_size, max_seq_len=block_size, base=rope_base
        )

        self.heads = nn.ModuleList(
            [
                Head(n_embd, head_size, block_size, dropout, self.rope)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.block_size = block_size

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """
        x: (B, T, C)
        pos_offset: absolute position of x[:,0] in the (conceptual) full stream.
                    - Training: 0 (we use local subsequences).
                    - Generation: len(already_generated) - T (sliding window).
        """
        T = x.size(1)
        cos, sin = self.rope.get_cos_sin(
            T, x.device, pos_offset=pos_offset
        )  # (T, hs/2)
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFwd(nn.Module):
    """Position-wise MLP"""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer block with RoPE-aware attention"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHead(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedFwd(n_embd, dropout)

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        x = x + self.sa(self.ln1(x), pos_offset=pos_offset)
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Minimal GPT-style LM with **RoPE** (no additive position embeddings).

    Components
    ----------
    token_emb : nn.Embedding  -> (V, n_embd)
    blocks    : stack of RoPE-aware Transformer blocks
    ln_f      : final LayerNorm
    head      : projection to vocabulary logits

    Methods
    -------
    forward(idx, targets=None, pos_offset=0)
    generate(idx, max_new_tokens, temperature, top_k)
    """

    def __init__(
        self,
        V: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(V, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V)
        self.block_size = block_size

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        pos_offset: int = 0,
    ):
        """
        idx: (B, T) tokens
        targets: (B, T) next tokens (optional)
        pos_offset: absolute position of idx[:,0] in the conceptual stream.
                    - Training: 0 (fresh subsequences)
                    - Generation: total_len_so_far - T (for sliding window)
        """
        B, T = idx.shape
        assert T <= self.block_size

        x = self.token_emb(idx)  # (B, T, C)

        # Pass the same pos_offset through each block (attention will use it)
        for blk in self.blocks:
            x = blk(x, pos_offset=pos_offset)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample with a sliding window.
        We compute pos_offset = total_len_so_far - T for the current window.
        """
        for _ in range(max_new_tokens):
            # Use only the last block_size tokens as context
            total_len = idx.size(1)
            idx_cond = idx[:, -self.block_size :]
            T = idx_cond.size(1)
            pos_offset = total_len - T  # how many tokens precede the window

            # Forward pass (AMP on CUDA where available)
            if device == "cuda" and amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = self(idx_cond, pos_offset=pos_offset)
            else:
                logits, _ = self(idx_cond, pos_offset=pos_offset)

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < thresh, torch.full_like(logits, -float("inf")), logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ============================================================
# 7) Model / optimizer / scaler
# ============================================================
model = TinyGPT(
    V=V,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

# ============================================================
# 8) Training loop
# ============================================================
model.train()
for it in range(train_steps):
    xb, yb = get_batch(batch_size)
    optimizer.zero_grad(set_to_none=True)

    # Training uses pos_offset=0 (each batch is a fresh local chunk)
    if device == "cuda" and amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(xb, yb, pos_offset=0)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    else:
        _, loss = model(xb, yb, pos_offset=0)
        loss.backward()
        optimizer.step()

    if (it + 1) % 100 == 0:
        print(f"[train] iter {it+1:4d}  loss={loss.item():.3f}")

# ============================================================
# 9) Sampling
# ============================================================
model.eval()
prompt = "RL aligns the model"
start_ids = encode(prompt).unsqueeze(0).to(device)

with torch.no_grad():
    out = model.generate(start_ids, max_new_tokens=200, temperature=0.9, top_k=50)[
        0
    ].to("cpu")

print("\n--- SAMPLE ---\n")
print(decode(out))
