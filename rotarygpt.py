"""
mini_gpt_bpe_tiktoken_cuda_rope_autogrow.py
Tiny GPT with tiktoken BPE + Rotary Positional Embeddings (RoPE) that auto-extend
their cache on demand. Uses CUDA (bf16/fp16 AMP) if available, else MPS, else CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================
# Device + AMP configuration
# ================================
if torch.cuda.is_available():
    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16  # GradScaler only needed for fp16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    amp_dtype = None
    use_scaler = False
else:
    device = "cpu"
    amp_dtype = None
    use_scaler = False

torch.manual_seed(0)
print(f"[info] device={device}" + (f", amp_dtype={amp_dtype}" if amp_dtype else ""))

# ================================
# Tokenizer (tiktoken BPE)
# ================================
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


# ================================
# Corpus (single stream)
# ================================
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

# ================================
# Hyperparameters (tiny)
# ================================
block_size = 64  # context window (attention ~ O(T^2))
n_embd = 96  # model width (must be divisible by n_head)
n_head = 3  # => head_size = 32 (even -> required by RoPE)
n_layer = 2
dropout = 0.1
batch_size = 48
train_steps = 600
learning_rate = 3e-3
rope_base = 10000.0  # standard RoPE base


# ================================
# Batching over a single stream
# ================================
def get_batch(B=batch_size):
    ix = torch.randint(0, N - block_size - 1, (B,))
    x = torch.stack([X_all[i : i + block_size] for i in ix])
    y = torch.stack([X_all[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ================================
# Rotary Positional Embeddings
# (auto-extending cache)
# ================================
class RotaryEmbedding(nn.Module):
    """
    RoPE cos/sin cache that **grows on demand**.
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
        freqs = t * self.inv_freq.unsqueeze(0)  # (T, D/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)  # (T, D/2)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)  # (T, D/2)

    @torch.no_grad()
    def _extend_to(self, new_max_seq_len: int, device, dtype):
        """Extend cos/sin caches up to new_max_seq_len (inclusive end-1)."""
        if new_max_seq_len <= self.max_seq_len:
            return
        t = torch.arange(
            self.max_seq_len, new_max_seq_len, dtype=torch.float32, device=device
        ).unsqueeze(1)
        freqs = t * self.inv_freq.to(device).unsqueeze(0)
        cos_new = freqs.cos().to(dtype=dtype)
        sin_new = freqs.sin().to(dtype=dtype)
        self.cos_cached = torch.cat(
            [self.cos_cached.to(device=device, dtype=dtype), cos_new], dim=0
        )
        self.sin_cached = torch.cat(
            [self.sin_cached.to(device=device, dtype=dtype), sin_new], dim=0
        )
        self.max_seq_len = new_max_seq_len

    def get_cos_sin(self, T: int, device, pos_offset: int = 0, dtype=None):
        """
        Return cos/sin for positions [pos_offset .. pos_offset+T-1].
        Auto-extends cache if needed. Dtype matches current compute.
        """
        end = pos_offset + T
        use_dtype = dtype or torch.float32
        if end > self.max_seq_len:
            self._extend_to(end, device=device, dtype=use_dtype)
        cos = self.cos_cached[pos_offset:end].to(device=device, dtype=use_dtype)
        sin = self.sin_cached[pos_offset:end].to(device=device, dtype=use_dtype)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to last-dim pairs of x.
    x:   (B, T, D) with even D
    cos: (T, D/2), sin: (T, D/2)
    """
    x_even = x[..., ::2]  # (B, T, D/2)
    x_odd = x[..., 1::2]  # (B, T, D/2)
    cos = cos.unsqueeze(0)  # (1, T, D/2)
    sin = sin.unsqueeze(0)  # (1, T, D/2)
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    x_out = torch.empty_like(x)
    x_out[..., ::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd
    return x_out


# ================================
# Transformer (RoPE in attention)
# ================================
class Head(nn.Module):
    """Single causal self-attention head with RoPE on Q/K."""

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
        self.rope = rope

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # rotate queries/keys
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, hs)
        return out


class MultiHead(nn.Module):
    """Multi-head self-attention; shares a RoPE cache across heads."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        assert head_size % 2 == 0, "head_size must be even for RoPE"
        self.rope = RotaryEmbedding(
            head_dim=head_size, max_seq_len=block_size, base=10000.0
        )
        self.heads = nn.ModuleList(
            [
                Head(n_embd, head_size, block_size, dropout, self.rope)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        T = x.size(1)
        # Match dtype to current compute (important under autocast)
        cos, sin = self.rope.get_cos_sin(
            T, x.device, pos_offset=pos_offset, dtype=x.dtype
        )
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)
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
    """Pre-LN Transformer block with RoPE-aware attention."""

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
    """Minimal GPT-style LM with RoPE (no additive position embeddings)."""

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
        B, T = idx.shape
        assert T <= self.block_size
        x = self.token_emb(idx)  # (B,T,C)
        for blk in self.blocks:
            x = blk(x, pos_offset=pos_offset)
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)
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
    ):
        for _ in range(max_new_tokens):
            total_len = idx.size(1)
            idx_cond = idx[:, -self.block_size :]
            T = idx_cond.size(1)
            pos_offset = total_len - T  # absolute position of first token in window

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
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ================================
# Init model/optimizer/scaler
# ================================
model = TinyGPT(V, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

# ================================
# Train
# ================================
model.train()
for it in range(train_steps):
    xb, yb = get_batch(batch_size)
    optimizer.zero_grad(set_to_none=True)

    if device == "cuda" and amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(
                xb, yb, pos_offset=0
            )  # training uses local chunks (offset 0)
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

# ================================
# Sample
# ================================
model.eval()
prompt = "RL aligns the model"
start_ids = encode(prompt).unsqueeze(0).to(device)
with torch.no_grad():
    out = model.generate(start_ids, max_new_tokens=200, temperature=0.9, top_k=50)[
        0
    ].to("cpu")
print("\n--- SAMPLE ---\n")
print(decode(out))
"""
mini_gpt_bpe_tiktoken_cuda_rope_autogrow.py
Tiny GPT with tiktoken BPE + Rotary Positional Embeddings (RoPE) that auto-extend
their cache on demand. Uses CUDA (bf16/fp16 AMP) if available, else MPS, else CPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================
# Device + AMP configuration
# ================================
if torch.cuda.is_available():
    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16  # GradScaler only needed for fp16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    amp_dtype = None
    use_scaler = False
else:
    device = "cpu"
    amp_dtype = None
    use_scaler = False

torch.manual_seed(0)
print(f"[info] device={device}" + (f", amp_dtype={amp_dtype}" if amp_dtype else ""))

# ================================
# Tokenizer (tiktoken BPE)
# ================================
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


# ================================
# Corpus (single stream)
# ================================
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

# ================================
# Hyperparameters (tiny)
# ================================
block_size = 64  # context window (attention ~ O(T^2))
n_embd = 96  # model width (must be divisible by n_head)
n_head = 3  # => head_size = 32 (even -> required by RoPE)
n_layer = 2
dropout = 0.1
batch_size = 48
train_steps = 600
learning_rate = 3e-3
rope_base = 10000.0  # standard RoPE base


# ================================
# Batching over a single stream
# ================================
def get_batch(B=batch_size):
    ix = torch.randint(0, N - block_size - 1, (B,))
    x = torch.stack([X_all[i : i + block_size] for i in ix])
    y = torch.stack([X_all[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ================================
# Rotary Positional Embeddings
# (auto-extending cache)
# ================================
class RotaryEmbedding(nn.Module):
    """
    RoPE cos/sin cache that **grows on demand**.
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
        freqs = t * self.inv_freq.unsqueeze(0)  # (T, D/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)  # (T, D/2)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)  # (T, D/2)

    @torch.no_grad()
    def _extend_to(self, new_max_seq_len: int, device, dtype):
        """Extend cos/sin caches up to new_max_seq_len (inclusive end-1)."""
        if new_max_seq_len <= self.max_seq_len:
            return
        t = torch.arange(
            self.max_seq_len, new_max_seq_len, dtype=torch.float32, device=device
        ).unsqueeze(1)
        freqs = t * self.inv_freq.to(device).unsqueeze(0)
        cos_new = freqs.cos().to(dtype=dtype)
        sin_new = freqs.sin().to(dtype=dtype)
        self.cos_cached = torch.cat(
            [self.cos_cached.to(device=device, dtype=dtype), cos_new], dim=0
        )
        self.sin_cached = torch.cat(
            [self.sin_cached.to(device=device, dtype=dtype), sin_new], dim=0
        )
        self.max_seq_len = new_max_seq_len

    def get_cos_sin(self, T: int, device, pos_offset: int = 0, dtype=None):
        """
        Return cos/sin for positions [pos_offset .. pos_offset+T-1].
        Auto-extends cache if needed. Dtype matches current compute.
        """
        end = pos_offset + T
        use_dtype = dtype or torch.float32
        if end > self.max_seq_len:
            self._extend_to(end, device=device, dtype=use_dtype)
        cos = self.cos_cached[pos_offset:end].to(device=device, dtype=use_dtype)
        sin = self.sin_cached[pos_offset:end].to(device=device, dtype=use_dtype)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to last-dim pairs of x.
    x:   (B, T, D) with even D
    cos: (T, D/2), sin: (T, D/2)
    """
    x_even = x[..., ::2]  # (B, T, D/2)
    x_odd = x[..., 1::2]  # (B, T, D/2)
    cos = cos.unsqueeze(0)  # (1, T, D/2)
    sin = sin.unsqueeze(0)  # (1, T, D/2)
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    x_out = torch.empty_like(x)
    x_out[..., ::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd
    return x_out


# ================================
# Transformer (RoPE in attention)
# ================================
class Head(nn.Module):
    """Single causal self-attention head with RoPE on Q/K."""

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
        self.rope = rope

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # rotate queries/keys
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, hs)
        return out


class MultiHead(nn.Module):
    """Multi-head self-attention; shares a RoPE cache across heads."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        assert head_size % 2 == 0, "head_size must be even for RoPE"
        self.rope = RotaryEmbedding(
            head_dim=head_size, max_seq_len=block_size, base=10000.0
        )
        self.heads = nn.ModuleList(
            [
                Head(n_embd, head_size, block_size, dropout, self.rope)
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        T = x.size(1)
        # Match dtype to current compute (important under autocast)
        cos, sin = self.rope.get_cos_sin(
            T, x.device, pos_offset=pos_offset, dtype=x.dtype
        )
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)
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
    """Pre-LN Transformer block with RoPE-aware attention."""

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
    """Minimal GPT-style LM with RoPE (no additive position embeddings)."""

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
        B, T = idx.shape
        assert T <= self.block_size
        x = self.token_emb(idx)  # (B,T,C)
        for blk in self.blocks:
            x = blk(x, pos_offset=pos_offset)
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)
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
    ):
        for _ in range(max_new_tokens):
            total_len = idx.size(1)
            idx_cond = idx[:, -self.block_size :]
            T = idx_cond.size(1)
            pos_offset = total_len - T  # absolute position of first token in window

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
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ================================
# Init model/optimizer/scaler
# ================================
model = TinyGPT(V, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

# ================================
# Train
# ================================
model.train()
for it in range(train_steps):
    xb, yb = get_batch(batch_size)
    optimizer.zero_grad(set_to_none=True)

    if device == "cuda" and amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(
                xb, yb, pos_offset=0
            )  # training uses local chunks (offset 0)
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

# ================================
# Sample
# ================================
model.eval()
prompt = "RL aligns the model"
start_ids = encode(prompt).unsqueeze(0).to(device)
with torch.no_grad():
    out = model.generate(start_ids, max_new_tokens=200, temperature=0.9, top_k=50)[
        0
    ].to("cpu")
print("\n--- SAMPLE ---\n")
print(decode(out))
