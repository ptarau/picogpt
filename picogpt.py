"""
mini_gpt_bpe_tiktoken_cuda_commented.py
Tiny GPT with tiktoken BPE. Auto-selects CUDA (AMP), MPS, or CPU.

What this file demonstrates
---------------------------
- A minimal, readable GPT-style Transformer for next-token prediction.
- Tokenization with a production-grade BPE (tiktoken `cl100k_base`).
- Device selection: prefers CUDA (with mixed precision), else MPS, else CPU.
- A single-stream language-modeling dataset with random (x, y) slices.
- Short training run and autoregressive sampling from a prompt.

Design choices (why these constants?)
-------------------------------------
- block_size=64: short context keeps attention's O(T^2) cost tiny.
- n_embd=96, n_head=3 -> head_size=32: small matrices; n_embd % n_head == 0.
- n_layer=2: deeper than 1 for noticeable gains, still fast on CPU.
- dropout=0.1: light regularization; can set to 0.0 for raw speed.
- learning_rate=3e-3 with AdamW: simple, robust default.

Shapes (always handy)
---------------------
- Tokens: (B, T) int64
- Embeddings / hidden: (B, T, C) with C == n_embd
- Attention weights: (B, T, T)
- Logits: (B, T, V) with V == vocab size
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 0) Device & AMP (automatic mixed precision) configuration
# ============================================================
if torch.cuda.is_available():
    # Prefer CUDA. If bf16 tensor cores exist, use bf16 autocast (no GradScaler needed).
    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)  # GradScaler only for fp16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    # Apple Silicon (Metal). AMP not used here—float32 is fine for this toy model.
    device = "mps"
    amp_dtype = None
    use_scaler = False
else:
    # CPU fallback. Float32 math.
    device = "cpu"
    amp_dtype = None
    use_scaler = False

torch.manual_seed(0)
print(f"[info] using device={device}" + (f", amp_dtype={amp_dtype}" if amp_dtype else ""))

# ============================================================
# 1) Tokenizer (tiktoken BPE)
# ============================================================
try:
    import tiktoken
except ImportError as e:
    raise SystemExit("tiktoken is required. Install with: pip install tiktoken") from e

# Use a fast, common BPE encoding (same family as used in GPT-4/4o stacks).
enc = tiktoken.get_encoding("cl100k_base")
V = enc.n_vocab

def encode(s: str) -> torch.Tensor:
    """Encode a Python string to a 1D LongTensor of token ids."""
    return torch.tensor(enc.encode(s), dtype=torch.long)

def decode(ids) -> str:
    """Decode token ids (tensor or list[int]) back into a Python string."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return enc.decode(ids)

# ============================================================
# 2) Corpus (single stream)
#    Tip: append your own text here to specialize the tiny model.
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

with open("triples.tsv","r") as f:
    data_text=f.read()

X_all = encode(data_text)  # a single long stream of token ids, shape (N,)
N = X_all.numel()
assert N > 200, f"Corpus too short after BPE: {N} tokens. Add more text."

# ============================================================
# 3) Hyperparameters (small for speed)
# ============================================================
block_size    = 64   # context length; attention scales as O(T^2) -> keep modest
n_embd        = 96   # model width; must be divisible by n_head
n_head        = 3    # multi-head count -> head_size = n_embd // n_head == 32
n_layer       = 2    # transformer depth
dropout       = 0.1  # light regularization
batch_size    = 48
train_steps   = 600
learning_rate = 3e-3

# ============================================================
# 4) Batching
# ============================================================
def get_batch(B=batch_size):
    """
    Sample B random subsequences of length `block_size` from the single token stream.

    Returns
    -------
    x : LongTensor of shape (B, T)
        Input token indices.
    y : LongTensor of shape (B, T)
        Target token indices (next-token labels), i.e., x shifted by 1.
    """
    # Random start indices chosen so i+block_size+1 is valid.
    ix = torch.randint(0, N - block_size - 1, (B,))
    x = torch.stack([X_all[i:i+block_size] for i in ix])
    y = torch.stack([X_all[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# 5) Transformer components
# ============================================================
class Head(nn.Module):
    """
    A single **causal self-attention** head.

    What it does (per forward pass):
    - Projects the input (B, T, C) into query, key, value tensors of width `head_size`.
    - Computes scaled dot-product attention scores: (q @ k^T) / sqrt(head_size).
    - Applies a lower-triangular mask so position t cannot attend to positions > t (causality).
    - Softmax over source positions, dropout, then weighted sum of values to get (B, T, head_size).

    Key ideas:
    - The scale factor sqrt(head_size) stabilizes gradients.
    - The mask enforces left-to-right autoregressive behavior.
    """
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Precomputed causal mask (T x T); sliced to actual T at runtime.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, hs)
        return out


class MultiHead(nn.Module):
    """
    **Multi-head self-attention**: run several attention heads in parallel and concatenate.

    Why multiple heads?
    - Each head can specialize to different positional patterns or token relations.
    - Concatenating heads followed by a linear projection mixes their information.

    Constraints:
    - `n_embd % n_head == 0` so heads split evenly into size `head_size`.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)  # mixes concatenated heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_head*hs) == (B, T, n_embd)
        x = self.proj(x)                                   # (B, T, n_embd)
        x = self.dropout(x)
        return x


class FeedFwd(nn.Module):
    """
    **Position-wise feed-forward network** (applied independently at each time step).

    Typical Transformer FFN structure:
      Linear(n_embd -> 4*n_embd) -> GELU -> Linear(4*n_embd -> n_embd) -> Dropout

    Why the 4x expansion?
    - It provides capacity for non-linear mixing per position (token).
    - For ultra-tiny models you could try 2x to speed up further.
    """
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    A standard **Transformer block** using **Pre-LayerNorm**:

      x = x + SelfAttention(LayerNorm(x))
      x = x + FeedForward(LayerNorm(x))

    Why Pre-LN?
    - Normalizing before each sublayer tends to stabilize training, especially deeper stacks.
    - Residual connections (x + ...) are critical for gradient flow.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHead(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff  = FeedFwd(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    A minimal **GPT-style language model**.

    Components
    ----------
    token_emb : nn.Embedding
        Maps token ids -> vectors (V, n_embd).
    pos_emb : nn.Embedding
        Learned absolute positional embeddings for indices [0..block_size-1].
    blocks : nn.Sequential[Block, ...]
        A stack of Transformer blocks.
    ln_f : nn.LayerNorm
        Final normalization before projecting to logits.
    head : nn.Linear
        Language modeling head: (n_embd -> V) logits for each position.

    Methods
    -------
    forward(idx, targets=None):
        - idx: (B, T) token indices
        - targets (optional): (B, T) next-token indices for CrossEntropy loss
        - returns: (logits, loss) with logits (B, T, V).
    generate(idx, max_new_tokens, temperature, top_k):
        - Autoregressively samples new tokens from a prompt.

    Less obvious facts
    ------------------
    - pos_emb has size (block_size, n_embd), so runtime T must be <= block_size.
    - We keep a `block_size` attribute so generation can slide a context window.
    """
    def __init__(self, V: int, n_embd: int, n_head: int, n_layer: int,
                 block_size: int, dropout: float):
        super().__init__()
        self.token_emb = nn.Embedding(V, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)
        self.blocks    = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f      = nn.LayerNorm(n_embd)
        self.head      = nn.Linear(n_embd, V)
        self.block_size = block_size

        # Initialize weights with small std (common GPT-2 style init).
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Small-N init: helps start training stable (esp. logits)."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Compute logits (and optional loss) for a batch of token sequences.

        Parameters
        ----------
        idx : LongTensor (B, T)
            Token ids.
        targets : LongTensor (B, T) or None
            Next-token labels; if provided, returns CE loss.

        Returns
        -------
        logits : FloatTensor (B, T, V)
            Unnormalized scores for each vocabulary token at each position.
        loss : torch.Tensor or None
            Cross-entropy loss if `targets` is given, otherwise None.
        """
        B, T = idx.shape
        assert T <= self.block_size, "sequence length exceeds block_size"

        # Token + positional embeddings
        tok = self.token_emb(idx)                                # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))   # (T, C)
        x = tok + pos                                            # broadcast add -> (B, T, C)

        # Transformer stack
        x = self.blocks(x)                                       # (B, T, C)

        # Projection to logits
        x = self.ln_f(x)                                         # (B, T, C)
        logits = self.head(x)                                    # (B, T, V)

        loss = None
        if targets is not None:
            # Flatten to (B*T, V) vs (B*T,) as CE expects
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """
        Autoregressively sample new tokens from a prompt.

        Args
        ----
        idx : LongTensor (B, T0)
            Prompt token ids.
        max_new_tokens : int
            Number of tokens to append.
        temperature : float
            >1.0 -> more random; <1.0 -> more conservative (0.7-1.0 is common).
        top_k : int or None
            If set, restrict sampling to top_k tokens at each step (truncation).

        Returns
        -------
        LongTensor (B, T0 + max_new_tokens)
            The extended sequence of token ids.
        """
        for _ in range(max_new_tokens):
            # Keep only the last `block_size` tokens as context (sliding window)
            idx_cond = idx[:, -self.block_size:]

            # On CUDA, autocast helps both speed and memory.
            if device == "cuda" and amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, _ = self(idx_cond)
            else:
                logits, _ = self(idx_cond)

            # Take logits at the final time step (B, V) and apply temperature
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            # Optional top-k filtering: zero all logits below the kth highest
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)            # (B, V)
            next_id = torch.multinomial(probs, 1)        # (B, 1)

            # Append to running sequence
            idx = torch.cat([idx, next_id], dim=1)       # (B, T+1)
        return idx

# ============================================================
# 6) Model / optimizer / scaler
# ============================================================
model = TinyGPT(
    V=V,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
).to(device)

# AdamW commonly works well for Transformers.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# GradScaler improves numerical stability only for fp16; not needed for bf16.
scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

# ============================================================
# 7) Training loop (tiny)
# ============================================================
model.train()
for it in range(train_steps):
    xb, yb = get_batch(batch_size)
    optimizer.zero_grad(set_to_none=True)

    if device == "cuda" and amp_dtype is not None:
        # Mixed precision on CUDA: bf16 (no scaler) or fp16 (with scaler)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(xb, yb)

        if use_scaler:  # fp16 path
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:           # bf16 path
            loss.backward()
            optimizer.step()
    else:
        # Full precision path (CPU or MPS)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    if (it + 1) % 100 == 0:
        print(f"[train] iter {it+1:4d}  loss={loss.item():.3f}")

# ============================================================
# 8) Sampling from a prompt
# ============================================================
model.eval()
prompt = "non_monotonic_logic"
start_ids = encode(prompt).unsqueeze(0).to(device)  # (1, T0)

with torch.no_grad():
    out = model.generate(
        start_ids,
        max_new_tokens=200,
        temperature=0.9,  # a bit of creativity
        top_k=50          # truncate tail for cleaner text
    )[0].to("cpu")

print("\n--- SAMPLE ---\n")
print(decode(out))
