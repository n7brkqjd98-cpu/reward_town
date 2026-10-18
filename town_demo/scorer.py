"""Batched scoring with Skywork-Reward-V2-Qwen3-0.6B (the only model-touching code)."""

import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardScorer:
    def __init__(self, model_path: Path, device: str, batch_size: int = 16):
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "CUDA requested but torch.cuda.is_available() is False; "
                "run with .venv-gpu/bin/python or pass --device cpu"
            )
        self.device = device
        self.batch_size = batch_size
        # float32 on both devices: bfloat16 shifts scores by up to ~0.27 between batched and single
        # scoring (float32: ~1e-5), enough to flip close argmax decisions.
        self.dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # .to(device) instead of device_map=...: device_map requires `accelerate`, which .venv lacks.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=self.dtype)
        self.model.to(device).eval()
        self.stats = {"calls": 0, "sequences": 0, "seconds": 0.0}

    @property
    def gpu_name(self) -> str:
        return torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"

    def _format(self, prompt: str, response: str) -> str:
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(conv, tokenize=False)
        bos = self.tokenizer.bos_token
        if bos is not None and text.startswith(bos):
            text = text[len(bos):]
        return text

    def score(self, prompt: str, responses: list[str]) -> list[float]:
        if not responses:
            return []
        start = time.perf_counter()
        texts = [self._format(prompt, r) for r in responses]
        scores: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i:i + self.batch_size], return_tensors="pt", padding=True
                ).to(self.device)
                logits = self.model(**batch).logits
                scores.extend(logits[:, 0].float().tolist())
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.stats["calls"] += 1
        self.stats["sequences"] += len(responses)
        self.stats["seconds"] += time.perf_counter() - start
        return scores
