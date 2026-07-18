from __future__ import annotations

import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, roc_auc_score,
)

from rdl_e2e.data import TaskType
from rdl_e2e.model import RDLModel


def evaluate(
    model: RDLModel, edge_index_dict: dict, labels: torch.Tensor, test_idx: torch.Tensor,
    task_type: TaskType, device: torch.device,
) -> dict:
    model.eval()
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    labels = labels.to(device)

    with torch.no_grad():
        logits, entity_emb = model(edge_index_dict, device)
        logits_test = logits[test_idx].cpu()
        y_true = labels[test_idx].cpu().numpy()

    if task_type == "binary_classification":
        probs = torch.sigmoid(logits_test).numpy()
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probs),
            "avg_precision": average_precision_score(y_true, probs),
        }
    elif task_type == "multiclass_classification":
        preds = logits_test.argmax(dim=1).numpy()
        metrics = {
            "accuracy": accuracy_score(y_true, preds),
            "f1_macro": f1_score(y_true, preds, average="macro", zero_division=0),
        }
    else:
        preds = logits_test.numpy()
        metrics = {
            "mae": mean_absolute_error(y_true, preds),
            "rmse": mean_squared_error(y_true, preds) ** 0.5,
        }

    metrics["embeddings"] = entity_emb.detach().cpu()
    return metrics
