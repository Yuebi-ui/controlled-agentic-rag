"""Bridge existing fine-tuned BGE-style reranker checkpoints to the new Scorer."""

from __future__ import annotations

from typing import Sequence

from ..contracts import Evidence


class HFSequenceClassificationScorer:
    """Loads user-supplied local checkpoint. No bundled model weights.

    Caution: output logits must match the LR router's training distribution.
    A model tuned on Child-level text should be validated on Parent/Image surrogates.
    """

    def __init__(self, model_dir: str, device: str = "cpu", max_length: int = 2048):
        if not model_dir:
            raise ValueError("path to fine-tuned BGE reranker is required")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.device = device
        self.model.to(device).eval()
        self.max_length = max_length

    def score_many(self, query: str, evidences: Sequence[Evidence]) -> list[float]:
        if not evidences:
            return []
        pairs = [(query, e.text_surrogate) for e in evidences]
        inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim != 1:
            raise ValueError("expected one reranker logit per evidence")
        values = logits.detach().cpu().tolist()
        if len(values) != len(evidences):
            raise ValueError("checkpoint score count mismatch")
        return [float(x) for x in values]
