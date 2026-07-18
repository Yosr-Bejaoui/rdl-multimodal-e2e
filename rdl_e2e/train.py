from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from rdl_e2e.data import TaskType
from rdl_e2e.model import RDLModel


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 5
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    seed: int = 42


@dataclass
class TrainResult:
    model: RDLModel
    history: dict = field(default_factory=lambda: {"train_loss": [], "val_loss": []})
    train_time_sec: float = 0.0
    n_trainable_params: int = 0
    stopped_epoch: int = 0


def _loss_fn(task_type: TaskType):
    if task_type == "binary_classification":
        return F.binary_cross_entropy_with_logits
    if task_type == "multiclass_classification":
        return F.cross_entropy
    return F.mse_loss


def train_rdl(
    model: RDLModel, edge_index_dict: dict, labels: torch.Tensor,
    train_idx: torch.Tensor, val_idx: torch.Tensor, task_type: TaskType,
    device: torch.device, config: TrainConfig = TrainConfig(),
) -> TrainResult:
    torch.manual_seed(config.seed)
    model = model.to(device)
    labels = labels.to(device)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    loss_fn = _loss_fn(task_type)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor, patience=config.scheduler_patience
    )

    result = TrainResult(model=model, n_trainable_params=sum(p.numel() for p in trainable_params))
    best_val, best_state, epochs_no_improve = math.inf, None, 0
    t0 = time.time()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(edge_index_dict, device)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(edge_index_dict, device)
            val_loss = loss_fn(logits[val_idx], labels[val_idx]).item()
        scheduler.step(val_loss)

        result.history["train_loss"].append(loss.item())
        result.history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-4:
            best_val, best_state, epochs_no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= config.patience:
            result.stopped_epoch = epoch
            break
    else:
        result.stopped_epoch = config.epochs - 1

    if best_state is not None:
        model.load_state_dict(best_state)
    result.train_time_sec = time.time() - t0
    return result
