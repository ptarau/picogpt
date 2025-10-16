# picogpt

Small scale GPT-like transformer stack, runnable on any PC or Mac.

Run it with:

```
python3 picogpt.py
```

Features:

- A minimal, readable GPT-style Transformer for next-token prediction.
- Tokenization with a production-grade BPE (tiktoken `cl100k_base`).
- Device selection: prefers CUDA (with mixed precision), else MPS, else CPU.
- A single-stream language-modeling dataset with random (x, y) slices.
- Short training run and autoregressive sampling from a prompt.

Dependencies: *numpy, torch,tiktoken*

Edit picochat.py to adjust these parameters:

- block_size=64: short context keeps attention's O(T^2) cost tiny.
- n_embd=96, n_head=3 -> head_size=32: small matrices; n_embd % n_head == 0.
- n_layer=2: deeper than 1 for noticeable gains, still fast on CPU.
- dropout=0.1: light regularization; can set to 0.0 for raw speed.
- learning_rate=3e-3 with AdamW: simple, robust default.

Enjoy,
Paul Tarau (and GPT5-thinking)