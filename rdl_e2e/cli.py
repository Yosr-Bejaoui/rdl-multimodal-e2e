from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from rdl_e2e.data import RelationalDatabase, build_hetero_graph
from rdl_e2e.encoders import Tokenizer
from rdl_e2e.evaluate import evaluate
from rdl_e2e.model import RDLModel
from rdl_e2e.train import TrainConfig, train_rdl


def load_database(cfg: dict) -> RelationalDatabase:
    ds_cfg = cfg["dataset"]
    if ds_cfg["source"] == "relbench":
        return RelationalDatabase.from_relbench(ds_cfg["name"], ds_cfg["task"])
    raise ValueError(f"Unknown dataset source: {ds_cfg['source']!r} (expected 'relbench')")


def make_split(n: int, seed: int, train_frac: float = 0.7, val_frac: float = 0.15):
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    n_train, n_val = int(train_frac * n), int(val_frac * n)
    return (
        torch.tensor(idx[:n_train]), torch.tensor(idx[n_train:n_train + n_val]),
        torch.tensor(idx[n_train + n_val:]),
    )


def run_one(cfg: dict, trainable_text: bool, device: torch.device) -> dict:
    db = load_database(cfg)
    built = build_hetero_graph(db)

    tokenizers = {}
    for nt, col_types in built.column_types.items():
        if "text" in col_types.values():
            text_cols = [c for c, t in col_types.items() if t == "text"]
            corpus = [str(v) for c in text_cols for v in built.raw_features[nt][c]]
            tokenizers[nt] = Tokenizer(
                cfg["model"]["text_model_name"], corpus, max_len=cfg["model"].get("max_text_len", 32)
            )

    node_types = list(built.graph.node_types)
    edge_types = list(built.graph.edge_types)
    out_dim = 1 if db.task.task_type != "multiclass_classification" else int(built.labels.max().item()) + 1

    model = RDLModel(
        node_types=node_types, edge_types=edge_types, entity_table=db.task.entity_table,
        node_column_types=built.column_types, tokenizers=tokenizers, raw_features=built.raw_features,
        text_model_name=cfg["model"]["text_model_name"], trainable_text=trainable_text,
        hid=cfg["model"].get("hidden_dim", 64), n_gnn_layers=cfg["model"].get("n_gnn_layers", 2), out_dim=out_dim,
    )

    train_idx, val_idx, test_idx = make_split(len(built.entity_indices), seed=cfg.get("seed", 42))
    train_cfg = TrainConfig(**cfg.get("train", {}))

    result = train_rdl(
        model=model, edge_index_dict=built.graph.edge_index_dict, labels=built.labels,
        train_idx=train_idx, val_idx=val_idx, task_type=db.task.task_type, device=device, config=train_cfg,
    )
    metrics = evaluate(
        model=result.model, edge_index_dict=built.graph.edge_index_dict, labels=built.labels,
        test_idx=test_idx, task_type=db.task.task_type, device=device,
    )
    metrics.pop("embeddings", None)

    return {
        "mode": "e2e" if trainable_text else "frozen",
        "dataset": db.name,
        "metrics": metrics,
        "train_time_sec": result.train_time_sec,
        "trainable_params": result.n_trainable_params,
        "stopped_epoch": result.stopped_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["frozen", "e2e", "both"], default="both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write JSON results.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    print(f"Config: {args.config} | device: {device}")

    results = []
    if args.mode in ("frozen", "both"):
        results.append(run_one(cfg, trainable_text=False, device=device))
        print("[frozen]", json.dumps(results[-1], indent=2))
    if args.mode in ("e2e", "both"):
        results.append(run_one(cfg, trainable_text=True, device=device))
        print("[e2e]", json.dumps(results[-1], indent=2))

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
