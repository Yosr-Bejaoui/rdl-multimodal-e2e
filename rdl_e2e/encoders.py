from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Tokenizer:
    def __init__(self, model_name: str, corpus: list[str], max_len: int = 32):
        self.max_len = max_len
        self.model_name = model_name
        self.hf_tokenizer = None
        self.vocab: dict[str, int] | None = None
        try:
            from transformers import AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.vocab_size_ = self.hf_tokenizer.vocab_size
        except Exception as e:
            warnings.warn(
                f"Could not load pretrained tokenizer '{model_name}' ({e!r}). "
                "Falling back to a whitespace tokenizer fit on the local corpus. "
                "Real subword tokenization requires network access to huggingface.co.",
                stacklevel=2,
            )
            vocab = {"[PAD]": 0, "[UNK]": 1}
            for text in corpus:
                for w in str(text).split():
                    if w not in vocab:
                        vocab[w] = len(vocab)
            self.vocab = vocab
            self.vocab_size_ = len(vocab)

    @property
    def is_pretrained(self) -> bool:
        return self.hf_tokenizer is not None

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hf_tokenizer is not None:
            enc = self.hf_tokenizer(
                list(map(str, texts)), padding="max_length", truncation=True,
                max_length=self.max_len, return_tensors="pt",
            )
            return enc["input_ids"], enc["attention_mask"]

        def encode_one(text: str) -> torch.Tensor:
            ids = [self.vocab.get(w, 1) for w in str(text).split()][: self.max_len]
            ids = ids + [0] * (self.max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        ids = torch.stack([encode_one(t) for t in texts])
        mask = (ids != 0).long()
        return ids, mask


class TabularEncoder(nn.Module):
    def __init__(self, n_numerical: int, category_sizes: list[int], hid: int = 64, cat_emb_dim: int = 8):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(n, cat_emb_dim) for n in category_sizes])
        in_dim = n_numerical + cat_emb_dim * len(category_sizes)
        self.mlp = nn.Sequential(nn.Linear(max(in_dim, 1), hid), nn.ReLU(), nn.Linear(hid, hid))
        self.in_dim = in_dim

    def forward(self, num_feats: torch.Tensor | None, cat_feats: list[torch.Tensor] | None) -> torch.Tensor:
        parts = []
        if num_feats is not None and num_feats.numel() > 0:
            parts.append(num_feats)
        if cat_feats:
            for emb, idx in zip(self.cat_embeddings, cat_feats):
                parts.append(emb(idx))
        if not parts:
            raise ValueError("TabularEncoder received no numeric or categorical features.")
        x = torch.cat(parts, dim=1)
        return self.mlp(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, vocab_size: int, hid: int = 64, trainable: bool = True):
        super().__init__()
        self.trainable = trainable
        self.is_pretrained = False
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(model_name)
            out_dim = self.backbone.config.hidden_size
            self.is_pretrained = True
        except Exception as e:
            warnings.warn(
                f"Could not load pretrained encoder '{model_name}' ({e!r}). "
                "Falling back to the same architecture with random weights. "
                "Real pretrained weights require network access to huggingface.co; "
                "results with the fallback demonstrate the training mechanics only, "
                "not the benefit of fine-tuning genuine pretrained knowledge.",
                stacklevel=2,
            )
            self.backbone, out_dim = self._build_fallback_bert(vocab_size)

        self.proj = nn.Linear(out_dim, hid)
        if not trainable:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _build_fallback_bert(vocab_size: int, dim: int = 64, n_layers: int = 2):
        from transformers import DistilBertConfig, DistilBertModel
        config = DistilBertConfig(
            vocab_size=vocab_size, dim=dim, hidden_dim=dim * 2,
            n_layers=n_layers, n_heads=4, max_position_embeddings=64,
        )
        return DistilBertModel(config), dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        grad_ctx = torch.enable_grad() if self.trainable else torch.no_grad()
        with grad_ctx:
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = hidden.mean(dim=1)
        return self.proj(pooled)


def build_node_encoder(
    table_name: str, raw_features: dict, column_types: dict[str, str],
    tokenizer: Tokenizer | None, text_model_name: str, hid: int, trainable_text: bool,
) -> tuple[nn.Module, dict]:
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t == "categorical"]
    text_cols = [c for c, t in column_types.items() if t == "text"]

    def normalize(values: np.ndarray) -> torch.Tensor:
        values = values.astype(float)
        mask = np.isnan(values)
        if mask.all():
            values[:] = 0.0
        else:
            values[mask] = np.nanmean(values)
        std = values.std() + 1e-6
        return torch.tensor((values - values.mean()) / std, dtype=torch.float)

    num_tensor = None
    if numeric_cols:
        num_tensor = torch.stack([normalize(raw_features[c]) for c in numeric_cols], dim=1)

    cat_tensors, cat_sizes = [], []
    for c in categorical_cols:
        codes, uniques = pd.factorize(raw_features[c])
        codes[codes == -1] = len(uniques)
        cat_tensors.append(torch.tensor(codes, dtype=torch.long))
        cat_sizes.append(len(uniques) + 1)

    tabular_encoder = TabularEncoder(
        n_numerical=len(numeric_cols) if numeric_cols else 0, category_sizes=cat_sizes, hid=hid,
    ) if (numeric_cols or categorical_cols) else None

    text_encoder, text_ids, text_mask = None, None, None
    if text_cols:
        if tokenizer is None:
            raise ValueError(f"Table '{table_name}' has text column(s) {text_cols} but no tokenizer was provided.")
        combined_text = [" ".join(str(raw_features[c][i]) for c in text_cols) for i in range(len(raw_features[text_cols[0]]))]
        text_ids, text_mask = tokenizer.encode_batch(combined_text)
        text_encoder = TextEncoder(text_model_name, tokenizer.vocab_size, hid=hid, trainable=trainable_text)

    class NodeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tabular = tabular_encoder
            self.text = text_encoder

        def forward(self, num_feats, cat_feats, ids, mask):
            parts = []
            if self.tabular is not None:
                parts.append(self.tabular(num_feats, cat_feats))
            if self.text is not None:
                parts.append(self.text(ids, mask))
            return torch.stack(parts, dim=0).sum(dim=0) if len(parts) > 1 else parts[0]

    precomputed = {
        "num_tensor": num_tensor, "cat_tensors": cat_tensors,
        "text_ids": text_ids, "text_mask": text_mask,
    }
    return NodeEncoder(), precomputed
