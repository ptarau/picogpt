# picogpt

A tiny GPT with tiktoken BPE and RoPE (auto-extend), with a simple class API.

- Minimal, readable GPT-style Transformer for next-token prediction.
- Tokenization with a production-grade BPE (tiktoken `cl100k_base`).
- Device selection: prefers CUDA (with mixed precision), else MPS, else CPU.
- A single-stream language-modeling dataset with random (x, y) slices.
- Short training run and autoregressive sampling from a prompt.

Dependencies: *numpy, torch, tiktoken*

## Install

```bash
pip install picogpt

pip install build
python -m build
pip install dist/picogpt-*.whl

from picogpt import PicoGPT

picoGPT = PicoGPT()  # defaults: tiny model, CUDA/MPS/CPU auto-detect

# Train on your text file
picoGPT.train_with_file("corpus.txt")

# Save / load
picoGPT.save_model("out/pico.pt")
picoGPT.load_model("out/pico.pt")

# Inspect
picoGPT.info()

# Inference
answer = picoGPT.ask("RL aligns the model")
print(answer)

# defaults

PicoGPT(
  block_size=64, n_embd=96, n_head=3, n_layer=2, dropout=0.1, rope_base=10000.0,
  batch_size=48, train_steps=600, learning_rate=3e-3, seed=0,
  tokenizer_name="cl100k_base", max_new_tokens=200, temperature=0.9, top_k=50,
  device="auto", amp=True
)


---

## Publish to PyPI (quick guide)

```bash
# from the project root (where pyproject.toml lives)
python -m pip install --upgrade build twine
python -m build
twine upload dist/*
