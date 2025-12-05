# picogpt.py
# High-level PicoGPT class: training, saving/loading, inference, info.
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch
import tiktoken

from .model import TinyGPT


def _select_device_and_amp(device: str = "auto"):
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", amp_dtype, (amp_dtype == torch.float16)
    if device == "mps" or (device == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return "mps", None, False
    return "cpu", None, False


@dataclass
class PicoConfig:
    # model
    block_size: int = 64
    n_embd: int = 96
    n_head: int = 3
    n_layer: int = 2
    dropout: float = 0.1
    rope_base: float = 10000.0
    # training
    batch_size: int = 48
    train_steps: int = 600
    learning_rate: float = 3e-3
    seed: int = 0
    # tokenizer & generation
    tokenizer_name: str = "cl100k_base"
    max_new_tokens: int = 200
    temperature: float = 0.9
    top_k: int | None = 50
    # runtime
    device: str = "auto"   # "auto" | "cuda" | "mps" | "cpu"
    amp: bool = True       # enable AMP if device supports it


class PicoGPT:
    """
    High-level wrapper around TinyGPT (RoPE/BPE).
    API:
        pico = PicoGPT()
        pico.train_with_file("corpus.txt")
        pico.save_model("model.pt")
        pico.load_model("model.pt")
        pico.info()
        answer = pico.ask("Your prompt...")
    """
    def __init__(self, **kwargs):
        self.cfg = PicoConfig(**kwargs)
        torch.manual_seed(self.cfg.seed)

        # device / AMP
        self.device, self.amp_dtype, self.use_scaler = _select_device_and_amp(self.cfg.device)
        if not self.cfg.amp:
            self.amp_dtype, self.use_scaler = None, False

        # tokenizer
        self.enc = tiktoken.get_encoding(self.cfg.tokenizer_name)
        self.V = self.enc.n_vocab

        # model & optimizer (created lazily at first train or on load)
        self.model: TinyGPT | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.trained_token_stream: torch.Tensor | None = None  # filled by train_with_file

    # ------------- public API -------------

    def train_with_file(self, file_name: str):
        """
        Train on the contents of a text file as a single token stream.
        """
        text = Path(file_name).read_text(encoding="utf-8")
        X_all = torch.tensor(self.enc.encode(text), dtype=torch.long)
        if X_all.numel() < self.cfg.block_size + 2:
            raise ValueError(f"Corpus too short after BPE: {X_all.numel()} tokens")

        self.trained_token_stream = X_all
        self._ensure_model()

        model = self.model
        assert model is not None
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda" and self.amp_dtype is not None and self.use_scaler))

        N = X_all.numel()
        B, T = self.cfg.batch_size, self.cfg.block_size

        def get_batch():
            ix = torch.randint(0, N - T - 1, (B,))
            x = torch.stack([X_all[i:i+T] for i in ix])
            y = torch.stack([X_all[i+1:i+T+1] for i in ix])
            return x.to(self.device), y.to(self.device)

        model.train()
        for it in range(self.cfg.train_steps):
            xb, yb = get_batch()
            optimizer.zero_grad(set_to_none=True)

            if self.device == "cuda" and self.amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    _, loss = model(xb, yb, pos_offset=0)
                if self.use_scaler:
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
                print(f"[train] iter {it+1:4d} loss={loss.item():.3f}")

        self.optimizer = optimizer  # keep a reference for potential fine-tuning

    def save_model(self, model_file: str):
        """
        Save model weights + config in a single torch file.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")
        payload = {
            "config": asdict(self.cfg),
            "state_dict": self.model.state_dict(),
            "tokenizer_name": self.cfg.tokenizer_name,
        }
        torch.save(payload, model_file)
        print(f"[save] wrote {model_file}")

    def load_model(self, model_file: str):
        """
        Load model weights + config from a torch file.
        """
        payload = torch.load(model_file, map_location="cpu")
        cfg_dict = payload.get("config", {})
        # Allow overriding runtime fields (device/amp) from current init
        runtime_overrides = {"device": self.cfg.device, "amp": self.cfg.amp}
        cfg_dict.update(runtime_overrides)
        self.cfg = PicoConfig(**cfg_dict)

        # Rebuild tokenizer & model on new device
        self.enc = tiktoken.get_encoding(payload.get("tokenizer_name", self.cfg.tokenizer_name))
        self.V = self.enc.n_vocab
        self.device, self.amp_dtype, self.use_scaler = _select_device_and_amp(self.cfg.device)
        if not self.cfg.amp:
            self.amp_dtype, self.use_scaler = None, False

        self._ensure_model()
        assert self.model is not None
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"[load] loaded model from {model_file} to device={self.device}")

    def info(self):
        """Print human-friendly info about parameters."""
        print(json.dumps(asdict(self.cfg), indent=2))
        if self.model is not None:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"parameters: {n_params:,}")
            print(f"device: {self.device}, amp: {bool(self.amp_dtype)}")

    def ask(self, query: str) -> str:
        """
        Generate a completion continuing `query`.
        """
        if self.model is None:
            self._ensure_model()  # untrained but usable
        assert self.model is not None

        start_ids = torch.tensor(self.enc.encode(query), dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model.generate(
                start_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                device=self.device,
                amp_dtype=self.amp_dtype,
            )[0].to("cpu")
        return self.enc.decode(out.tolist())

    # ------------- internal helpers -------------

    def _ensure_model(self):
        if self.model is not None:
            return
        self.model = TinyGPT(
            V=self.V,
            n_embd=self.cfg.n_embd,
            n_head=self.cfg.n_head,
            n_layer=self.cfg.n_layer,
            block_size=self.cfg.block_size,
            dropout=self.cfg.dropout,
            rope_base=self.cfg.rope_base,
        ).to(self.device)
        self.model.eval()
