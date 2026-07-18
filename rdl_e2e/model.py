from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from rdl_e2e.encoders import Tokenizer, build_node_encoder


def make_hetero_conv(edge_types: list[tuple[str, str, str]], hid: int) -> HeteroConv:
    return HeteroConv(
        {et: SAGEConv((hid, hid), hid) for et in edge_types}, aggr="mean"
    )


class RDLModel(nn.Module):
    def __init__(
        self, node_types: list[str], edge_types: list[tuple[str, str, str]], entity_table: str,
        node_column_types: dict[str, dict[str, str]], tokenizers: dict[str, Tokenizer],
        raw_features: dict[str, dict], text_model_name: str, trainable_text: bool,
        hid: int = 64, n_gnn_layers: int = 2, out_dim: int = 1,
    ):
        super().__init__()
        self.entity_table = entity_table
        self.node_types = node_types

        self.node_encoders = nn.ModuleDict()
        self.precomputed: dict[str, dict] = {}
        for nt in node_types:
            encoder, precomputed = build_node_encoder(
                table_name=nt, raw_features=raw_features[nt], column_types=node_column_types[nt],
                tokenizer=tokenizers.get(nt), text_model_name=text_model_name, hid=hid,
                trainable_text=trainable_text,
            )
            self.node_encoders[nt] = encoder
            self.precomputed[nt] = precomputed

        self.convs = nn.ModuleList([make_hetero_conv(edge_types, hid) for _ in range(n_gnn_layers)])
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, out_dim))

    def encode_inputs(self, device: torch.device) -> dict[str, torch.Tensor]:
        x_dict = {}
        for nt in self.node_types:
            p = self.precomputed[nt]
            num_feats = p["num_tensor"].to(device) if p["num_tensor"] is not None else None
            cat_feats = [t.to(device) for t in p["cat_tensors"]] if p["cat_tensors"] else None
            ids = p["text_ids"].to(device) if p["text_ids"] is not None else None
            mask = p["text_mask"].to(device) if p["text_mask"] is not None else None
            x_dict[nt] = self.node_encoders[nt](num_feats, cat_feats, ids, mask)
        return x_dict

    def forward(self, edge_index_dict: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        x_dict = self.encode_inputs(device)
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        entity_emb = x_dict[self.entity_table]
        logits = self.head(entity_emb).squeeze(-1)
        return logits, entity_emb
