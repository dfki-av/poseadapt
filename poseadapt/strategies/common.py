# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

import warnings
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from poseadapt.third_party.avalanche import ParamData, zerolike_params_dict


def extract_features(model, data_batch):
    """Extract features from the model."""
    return model.extract_feat(data_batch["inputs"])


def extract_logits(model, features):
    """Extract logits from the model."""
    return model.head.forward(features)


def get_predictions(model, data_batch):
    """Get predictions from the model."""
    features = extract_features(model, data_batch)
    return extract_logits(model, features)


def get_features_and_logits(model, data_batch):
    """Get logits and features from the model."""
    features = extract_features(model, data_batch)
    logits = extract_logits(model, features)
    return features, logits


def convert_predictions(preds_curr, c):
    if c is None:
        return preds_curr

    def _convert(preds):
        if c is None:
            return preds
        preds = preds.clone()
        preds[:, c.target_index, :] = preds[:, c.source_index, :]
        return preds[:, : c.num_keypoints, :]

    if isinstance(preds_curr, (tuple, list)):
        return [_convert(p) for p in preds_curr]
    return _convert(preds_curr)


def distillation_loss(
    student_logits,
    teacher_logits,
    T=1.0,
    tau: Optional[float] = None,
) -> torch.Tensor:
    # logits: [B, K, D]
    _, _, D = student_logits.shape
    s = (student_logits / T).reshape(-1, D)  # [B*K, D]
    t = (teacher_logits / T).reshape(-1, D)

    if tau is None:
        return F.kl_div(
            F.log_softmax(s, dim=-1), F.softmax(t, dim=1), reduction="batchmean"
        ) * (T * T)

    # per-distribution KL (no reduction)
    kl = F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="none",
    ).sum(dim=-1)  # [B*K]

    # teacher confidence per distribution = max prob
    conf = F.softmax(t, dim=-1).max(dim=-1).values  # [B*K]
    mask = (conf >= tau).float()

    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom * (T * T)


def compute_importances(
    model,
    dataloader,
    optim_wrapper,
    device,
) -> Dict[str, ParamData]:
    """
    Computes the EWC importance matrix for each parameter of the model.

    This method approximates the diagonal of the Fisher information matrix by computing
    the squared gradients of the model's parameters with respect to the loss as follows:

    .. math::

        F_i = \frac{1}{N} \sum_{n=1}^{N} \left(\frac{\partial \mathcal{L}_n}{\partial \theta_i}\right)^2

    where \( F_i \) is the Fisher information for parameter \( \theta_i \), \( \mathcal{L}_n \)
    is the loss function for the nth batch, and \( N \) is the number of batches in the dataloader.

    This serves as an approximation of each parameter's importance in preserving the learned
    experiences. The importances are averaged over the entire dataset to stabilize the estimates.

    Args:
        model: The model (trained on current experience) to compute parameter importances for.
        dataloader: The DataLoader object providing the dataset for the current experience.
        optim_wrapper: The optimizer used for gradient backpropagation.
        device: The device (cpu or cuda) the model is running on.

    Returns:
        A dictionary mapping parameter names to their importance scores, encapsulated in
        `ParamData` objects. Each `ParamData` contains the parameter name, its importance
        values, and additional metadata like shape and device information.
    """
    # Set model to eval mode
    model.eval()

    # Set RNN-like modules on GPU to training mode to avoid CUDA error
    if device == torch.device("cuda"):
        for module in model.modules():
            if isinstance(module, torch.nn.RNNBase):
                warnings.warn(
                    "RNN-like modules do not support "
                    "backward calls while in `eval` mode on CUDA "
                    "devices. Setting all `RNNBase` modules to "
                    "`train` mode. May produce inconsistent "
                    "output if such modules have `dropout` > 0."
                )
                module.train()

    # Compute importances by running the model on the experience dataset and using
    # gradients after each backward pass to approximate the diagonal of the
    # Fisher information matrix
    importances = zerolike_params_dict(model)
    for data in tqdm(dataloader, desc="Computing parameter importances", unit="batch"):
        # Forward pass
        optim_wrapper.optimizer.zero_grad()
        with optim_wrapper.optim_context(model):
            data = model.data_preprocessor(data, True)
            losses = model._run_forward(data, mode="loss")  # type: ignore
        loss, _ = model.parse_losses(losses)  # type: ignore

        # Backward pass
        loss.backward()

        # Compute parameter importances as square of gradients
        param_importance_pairs = zip(model.named_parameters(), importances.items())
        for (k1, p), (k2, imp) in param_importance_pairs:
            assert k1 == k2
            if p.grad is not None:
                imp.data += p.grad.data.clone().pow(2)

    # Average over mini batch length
    num_batches = float(len(dataloader))
    for _, imp in importances.items():
        imp.data /= num_batches

    # Restore model state
    model.train()

    return importances


def normalize_importances(
    importances: Dict[str, ParamData],
    eps: float = 1e-8,
    norm_mode: str = "minmax",
) -> Dict[str, ParamData]:
    """
    Normalize the importances to the range [0, 1].

    Args:
        importances: A dictionary mapping parameter names to their importance scores.
        eps: A small value to avoid division by zero.
        norm_mode: The normalization mode to use. Options are "minmax", "zscore", or "log".

    Returns:
        A dictionary with normalized importances.
    """

    def is_name_ignored(name):
        """Ignore model, top-level modules and batch norm layers."""
        return name == "" or len(name.split(".")) == 1 or "bn" in name.lower()

    # Filter out ignored names and zero importances
    importances = {
        name: imp.data.sum()
        for name, imp in importances.items()
        if not is_name_ignored(name) and imp.data.sum() > 0
    }

    def minmax(x, lower=5, upper=95):
        lower_val = torch.quantile(x, lower / 100.0)
        upper_val = torch.quantile(x, upper / 100.0)
        scale = (upper_val - lower_val).clamp(min=eps)
        x_clipped = torch.clamp(x, lower_val, upper_val)
        return (x_clipped - lower_val) / scale

    def zscore(x):
        mean = x.mean()
        std = x.std().clamp(min=eps)
        return (x - mean) / std

    def log(x):
        return torch.log(x + eps)

    # Normalize importances based on the specified mode
    if norm_mode == "minmax":
        normalize_func = minmax
    elif norm_mode == "zscore":
        normalize_func = zscore
    elif norm_mode == "log":
        normalize_func = log
    else:
        raise ValueError(f"Unknown normalization mode: {norm_mode}")

    # Apply normalization
    importances_tensor = torch.tensor(list(importances.values()))
    normalized = normalize_func(importances_tensor)
    return dict(zip(importances.keys(), normalized.tolist()))


def compute_normalized_importances(
    model,
    dataloader,
    optim_wrapper,
    device,
    eps: float = 1e-8,
    norm_mode: str = "minmax",
) -> Dict[str, ParamData]:
    """
    Compute and normalize the importances of model parameters.

    Args:
        model: The model to compute importances for.
        dataloader: The DataLoader providing the dataset.
        optim_wrapper: The optimizer wrapper used for backpropagation.
        device: The device (cpu or cuda) the model is running on.
        eps: A small value to avoid division by zero in normalization.
        norm_mode: The normalization mode to use.

    Returns:
        A dictionary mapping parameter names to their normalized importance scores.
    """
    importances = compute_importances(model, dataloader, optim_wrapper, device)
    return normalize_importances(importances, eps, norm_mode)
