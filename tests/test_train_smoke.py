import math

import torch
import pytest

from rdl_e2e.cli import make_split
from rdl_e2e.data import RelationalDatabase, build_hetero_graph
from rdl_e2e.encoders import Tokenizer
from rdl_e2e.evaluate import evaluate
from rdl_e2e.model import RDLModel
from rdl_e2e.train import TrainConfig, train_rdl

pytest.importorskip("relbench")


def _run(trainable_text: bool, epochs: int = 3):
    db = RelationalDatabase.from_relbench("rel-avito", "ad-ctr")
    built = build_hetero_graph(db)

    tokenizers = {}
    for nt, col_types in built.column_types.items():
        if "text" in col_types.values():
            text_cols = [c for c, t in col_types.items() if t == "text"]
            corpus = [str(v) for c in text_cols for v in built.raw_features[nt][c]]
            tokenizers[nt] = Tokenizer("distilbert-base-uncased", corpus, max_len=16)

    model = RDLModel(
        node_types=list(built.graph.node_types), edge_types=list(built.graph.edge_types),
        entity_table=db.task.entity_table, node_column_types=built.column_types,
        tokenizers=tokenizers, raw_features=built.raw_features,
        text_model_name="distilbert-base-uncased", trainable_text=trainable_text,
        hid=16, n_gnn_layers=2, out_dim=1,
    )

    train_idx, val_idx, test_idx = make_split(len(built.entity_indices), seed=1)
    config = TrainConfig(epochs=epochs, patience=epochs)
    device = torch.device("cpu")

    result = train_rdl(
        model=model, edge_index_dict=built.graph.edge_index_dict, labels=built.labels,
        train_idx=train_idx, val_idx=val_idx, task_type=db.task.task_type, device=device, config=config,
    )
    metrics = evaluate(
        model=result.model, edge_index_dict=built.graph.edge_index_dict, labels=built.labels,
        test_idx=test_idx, task_type=db.task.task_type, device=device,
    )
    return result, metrics


def test_smoke_frozen_runs_without_nan():
    result, metrics = _run(trainable_text=False)
    assert all(not math.isnan(v) for v in result.history["train_loss"])
    assert all(not math.isnan(v) for v in result.history["val_loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_smoke_trainable_runs_without_nan():
    result, metrics = _run(trainable_text=True)
    assert all(not math.isnan(v) for v in result.history["train_loss"])
    assert all(not math.isnan(v) for v in result.history["val_loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_smoke_trainable_uses_more_params_than_frozen():
    frozen_result, _ = _run(trainable_text=False)
    trainable_result, _ = _run(trainable_text=True)
    assert trainable_result.n_trainable_params > frozen_result.n_trainable_params
