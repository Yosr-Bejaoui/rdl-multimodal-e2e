import torch
import pytest

from rdl_e2e.data import RelationalDatabase, build_hetero_graph
from rdl_e2e.encoders import Tokenizer
from rdl_e2e.model import RDLModel

pytest.importorskip("relbench")


def _build_test_model(trainable_text: bool):
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
    return model, built


def test_forward_pass_shapes_frozen():
    model, built = _build_test_model(trainable_text=False)
    logits, emb = model(built.graph.edge_index_dict, torch.device("cpu"))
    assert logits.shape[0] == built.labels.shape[0]
    assert emb.shape == (built.labels.shape[0], 16)


def test_forward_pass_shapes_trainable():
    model, built = _build_test_model(trainable_text=True)
    logits, emb = model(built.graph.edge_index_dict, torch.device("cpu"))
    assert logits.shape[0] == built.labels.shape[0]
    assert emb.shape == (built.labels.shape[0], 16)


def test_frozen_text_encoder_has_no_grad_params():
    model, _ = _build_test_model(trainable_text=False)
    text_encoder = model.node_encoders["review"].text
    assert all(not p.requires_grad for p in text_encoder.backbone.parameters())


def test_trainable_text_encoder_has_grad_params():
    model, _ = _build_test_model(trainable_text=True)
    text_encoder = model.node_encoders["review"].text
    assert all(p.requires_grad for p in text_encoder.backbone.parameters())


def test_frozen_vs_trainable_param_count_differs():
    frozen_model, _ = _build_test_model(trainable_text=False)
    trainable_model, _ = _build_test_model(trainable_text=True)
    frozen_trainable_params = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    e2e_trainable_params = sum(p.numel() for p in trainable_model.parameters() if p.requires_grad)
    assert e2e_trainable_params > frozen_trainable_params
