import torch
import pytest

from rdl_e2e.data import RelationalDatabase, build_hetero_graph

pytest.importorskip("relbench")


def _load_real_db():
    return RelationalDatabase.from_relbench("rel-avito", "ad-ctr")


def test_real_database_loads():
    db = _load_real_db()
    assert db.tables
    assert db.foreign_keys
    assert db.task.entity_table in db.tables
    assert db.task.target_column in db.tables[db.task.entity_table].columns


def test_build_hetero_graph_node_and_edge_counts():
    db = _load_real_db()
    built = build_hetero_graph(db)
    g = built.graph

    assert len(g.node_types) > 0
    assert len(g.edge_types) > 0
    for et in g.edge_types:
        assert g[et].edge_index.shape[0] == 2

    assert built.labels.shape[0] == built.entity_indices.shape[0]
    assert built.labels.shape[0] == len(db.tables[db.task.entity_table])


def test_column_type_inference_excludes_keys_and_target():
    db = _load_real_db()
    built = build_hetero_graph(db)
    entity_types = built.column_types[db.task.entity_table]

    assert entity_types[db.primary_keys[db.task.entity_table]] == "id"
    assert entity_types[db.task.target_column] == "target"
