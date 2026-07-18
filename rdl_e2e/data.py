from __future__ import annotations
import importlib
import importlib.util
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

TaskType = Literal["binary_classification", "multiclass_classification", "regression"]


@dataclass
class ForeignKey:
    table: str
    column: str
    ref_table: str


@dataclass
class TaskSpec:
    entity_table: str
    target_column: str
    task_type: TaskType
    metric: str = "auto"


@dataclass
class RelationalDatabase:
    tables: dict[str, pd.DataFrame]
    primary_keys: dict[str, str]
    foreign_keys: list[ForeignKey]
    task: TaskSpec
    name: str = "unnamed"

    @staticmethod
    def from_relbench(dataset_name: str, task_name: str) -> "RelationalDatabase":
        try:
            get_dataset = importlib.import_module("relbench.datasets").get_dataset
            get_task = importlib.import_module("relbench.tasks").get_task
        except ImportError as e:
            if importlib.util.find_spec("relbench") is None:
                raise ImportError(
                    "relbench is not installed. Run `pip install relbench pytorch_frame` "
                    "on a machine with network access, then retry."
                ) from e
            if importlib.util.find_spec("torch_frame") is None:
                raise ImportError(
                    "pytorch_frame is not installed in this Python environment. "
                    "Run `pip install pytorch_frame` in the same interpreter/kernel, then retry."
                ) from e
            raise ImportError(
                "Failed to import relbench dependencies. "
                f"Original import error: {type(e).__name__}: {e}"
            ) from e

        dataset = get_dataset(dataset_name, download=True)
        task = get_task(dataset_name, task_name, download=True)
        db = dataset.get_db()

        tables, primary_keys, foreign_keys = {}, {}, []
        for table_name, table in db.table_dict.items():
            df = table.df.copy()
            tables[table_name] = df
            if table.pkey_col is not None:
                primary_keys[table_name] = table.pkey_col
            for fk_col, ref_table in table.fkey_col_to_pkey_table.items():
                foreign_keys.append(ForeignKey(table=table_name, column=fk_col, ref_table=ref_table))

        # Join labels into the entity table
        dfs = []
        for split in ["train", "val", "test"]:
            try:
                dfs.append(task.get_table(split).df)
            except Exception:
                pass
        if dfs:
            task_df = pd.concat(dfs).drop_duplicates(subset=[task.entity_col], keep='last')
            entity_table_name = task.entity_table
            pkey = primary_keys.get(entity_table_name, task.entity_col)
            tables[entity_table_name] = tables[entity_table_name].merge(
                task_df[[task.entity_col, task.target_col]], 
                left_on=pkey, right_on=task.entity_col, how='left'
            )
            # Fill missing labels with 0 just so the tensor conversion doesn't fail
            tables[entity_table_name][task.target_col] = tables[entity_table_name][task.target_col].fillna(0)

        task_type_name = (
            task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type)
        )
        if "binary" in task_type_name:
            mapped_type: TaskType = "binary_classification"
        elif "multiclass" in task_type_name or "multilabel" in task_type_name:
            mapped_type = "multiclass_classification"
        else:
            mapped_type = "regression"

        metric_name = task.metrics[0].__name__ if getattr(task, "metrics", None) else "auto"

        return RelationalDatabase(
            tables=tables,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            task=TaskSpec(entity_table=task.entity_table, target_column=task.target_col,
                          task_type=mapped_type, metric=metric_name),
            name=f"{dataset_name}/{task_name}",
        )


ColumnType = Literal["numeric", "categorical", "text", "id", "target"]


def infer_column_types(
    df: pd.DataFrame, primary_key: str | None, foreign_key_cols: set[str],
    target_column: str | None = None, text_avg_len_threshold: float = 20.0,
    categorical_cardinality_threshold: int = 50,
) -> dict[str, ColumnType]:
    excluded = {primary_key} | foreign_key_cols | ({target_column} if target_column else set())
    types: dict[str, ColumnType] = {}
    for col in df.columns:
        if col in excluded:
            types[col] = "target" if col == target_column else "id"
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "numeric"
        else:
            avg_len = series.astype(str).str.len().mean()
            n_unique = series.nunique()
            if avg_len is not None and avg_len > text_avg_len_threshold:
                types[col] = "text"
            elif n_unique <= categorical_cardinality_threshold:
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types


@dataclass
class GraphBuildResult:
    graph: HeteroData
    raw_features: dict[str, dict[str, object]]
    column_types: dict[str, dict[str, ColumnType]]
    entity_indices: np.ndarray
    labels: torch.Tensor


def build_hetero_graph(db: RelationalDatabase) -> GraphBuildResult:
    graph = HeteroData()
    row_id_maps: dict[str, dict] = {}

    for table_name, df in db.tables.items():
        graph[table_name].num_nodes = len(df)
        pk = db.primary_keys.get(table_name)
        if pk is not None:
            row_id_maps[table_name] = {v: i for i, v in enumerate(df[pk].values)}
        else:
            row_id_maps[table_name] = {i: i for i in range(len(df))}

    for fk in db.foreign_keys:
        src_df = db.tables[fk.table]
        src_idx = torch.arange(len(src_df))
        ref_map = row_id_maps[fk.ref_table]
        dst_idx = torch.tensor([ref_map[v] for v in src_df[fk.column].values], dtype=torch.long)

        fwd_rel = f"fk_{fk.column}"
        bwd_rel = f"rev_fk_{fk.column}"
        graph[fk.table, fwd_rel, fk.ref_table].edge_index = torch.stack([src_idx, dst_idx])
        graph[fk.ref_table, bwd_rel, fk.table].edge_index = torch.stack([dst_idx, src_idx])

    raw_features: dict[str, dict[str, object]] = {}
    column_types: dict[str, dict[str, ColumnType]] = {}
    fk_cols_by_table: dict[str, set[str]] = {}
    for fk in db.foreign_keys:
        fk_cols_by_table.setdefault(fk.table, set()).add(fk.column)

    for table_name, df in db.tables.items():
        is_entity_table = table_name == db.task.entity_table
        target_col = db.task.target_column if is_entity_table else None
        col_types = infer_column_types(
            df, db.primary_keys.get(table_name), fk_cols_by_table.get(table_name, set()), target_col
        )
        column_types[table_name] = col_types
        raw_features[table_name] = {
            col: df[col].values
            for col, t in col_types.items()
            if t in ("numeric", "categorical", "text")
        }

    entity_df = db.tables[db.task.entity_table]
    entity_indices = np.arange(len(entity_df))
    labels_raw = entity_df[db.task.target_column].values
    if db.task.task_type == "regression":
        labels = torch.tensor(labels_raw, dtype=torch.float)
    else:
        labels = torch.tensor(
            labels_raw,
            dtype=torch.float if db.task.task_type == "binary_classification" else torch.long,
        )

    return GraphBuildResult(
        graph=graph, raw_features=raw_features, column_types=column_types,
        entity_indices=entity_indices, labels=labels,
    )
