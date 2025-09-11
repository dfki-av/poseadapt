# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

from typing import Any, Dict

import torch
import torch.nn as nn
from mmpose.registry import HOOKS

from .base_evolution_plugin import BaseEvolutionPlugin


@torch.no_grad()
def load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
    logger: Any = None,
    fill_mode: str = "zeros",  # ["zeros","mean","std","uniform","kaiming","xavier"]
    copy_slice: str = "front",  # or "center" if you later want center-crop behavior
) -> None:
    """
    Load parameters/buffers from `state_dict` into `model`, with shape-adaptive init.

    - Exact-shape tensors are copied directly (dtype/device adapted).
    - Mismatched shapes: initialize a destination-sized tensor per `fill_mode`,
      then copy the overlapping slice of `state_dict[name]` into it.
      (Default slice policy = "front", i.e., start at 0 along each dim.)

    If `strict` is True:
      - Raises on missing keys, unexpected keys, and shape mismatches.

    Args:
        model: Target nn.Module.
        state_dict: Source state dict (tensor values).
        strict: Enforce exact key coverage & shapes.
        logger: Optional logger (must support .info/.warning).
        fill_mode: How to init extra elements when shapes differ.
        copy_slice: "front" (default) or "center" (future option).
    """
    own_state = model.state_dict()
    device = (
        next(model.parameters()).device
        if any(p.requires_grad for p in model.parameters())
        else torch.device("cpu")
    )

    def _make_filled_like(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        if fill_mode == "zeros":
            out = torch.zeros_like(dst, device=device)
        elif fill_mode == "mean":
            m = src.mean().item() if src.numel() > 0 else 0.0
            out = torch.full_like(dst, fill_value=m, device=device)
        elif fill_mode == "std":
            mean, std = (
                (
                    src.mean().item(),
                    src.std().item(),
                )
                if src.numel() > 1
                else (0.0, 1.0)
            )
            out = torch.normal(
                mean=mean, std=std, size=dst.shape, dtype=dst.dtype, device=device
            )
        elif fill_mode == "uniform":
            if src.numel() > 0:
                lo, hi = src.min().item(), src.max().item()
            else:
                lo, hi = -0.01, 0.01
            out = torch.empty_like(dst, device=device).uniform_(lo, hi)
        elif fill_mode == "kaiming":
            out = torch.empty_like(dst, device=device)
            torch.nn.init.kaiming_normal_(out)
        elif fill_mode == "xavier":
            out = torch.empty_like(dst, device=device)
            torch.nn.init.xavier_normal_(out)
        else:
            raise ValueError(
                f"Invalid fill_mode '{fill_mode}'. Choose from ['zeros','mean','std','uniform','kaiming','xavier']."
            )
        return out

    def _front_slices(dst_shape, src_shape):
        return tuple(slice(0, min(d, s)) for d, s in zip(dst_shape, src_shape))

    # Compute key diffs first (by name only)
    unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    # Track shape mismatches we couldn’t resolve under strict mode
    shape_mismatch: Dict[str, str] = {}

    # Attempt to load everything that exists by name
    for name, src in state_dict.items():
        if not isinstance(src, torch.Tensor):
            # Non-tensor entries are ignored (rare)
            continue
        if name not in own_state:
            continue  # will be reported as unexpected if strict

        dst = own_state[name]
        if dst.shape == src.shape:
            dst.copy_(src.to(dtype=dst.dtype, device=device))
        else:
            # Initialize destination-shaped tensor, then copy overlap
            resized = _make_filled_like(dst, src.to(dtype=dst.dtype))
            # Build slices
            if copy_slice == "front":
                slices = _front_slices(dst.shape, src.shape)
                resized[slices] = src.to(dtype=dst.dtype, device=device)[slices]
            else:
                # Future: implement "center" if needed
                slices = _front_slices(dst.shape, src.shape)
                resized[slices] = src.to(dtype=dst.dtype, device=device)[slices]
            dst.copy_(resized)

            if logger:
                logger.info(
                    f"{name}: shape {tuple(src.shape)} → {tuple(dst.shape)} via '{fill_mode}'."
                )

            if strict:
                shape_mismatch[name] = f"{tuple(src.shape)} → {tuple(dst.shape)}"

    # After copy, report issues
    # Only report remaining missing/unexpected (by name)
    if strict:
        errors = []
        if missing_keys:
            errors.append(f"Missing keys: {', '.join(sorted(missing_keys))}")
        if unexpected_keys:
            errors.append(f"Unexpected keys: {', '.join(sorted(unexpected_keys))}")
        if shape_mismatch:
            details = ", ".join([f"{k} ({v})" for k, v in shape_mismatch.items()])
            errors.append(f"Shape mismatches: {details}")
        if errors:
            raise RuntimeError("State dict load failed (strict): " + " | ".join(errors))
    else:
        if missing_keys and logger:
            logger.warning(
                f"Missing keys in model state_dict: {', '.join(sorted(missing_keys))}."
            )
        if unexpected_keys and logger:
            logger.warning(
                f"Unexpected keys in state_dict: {', '.join(sorted(unexpected_keys))}."
            )


@HOOKS.register_module()
class DefaultEvolutionPlugin(BaseEvolutionPlugin):
    """
    A plugin that enables model evolution by taking snapshots and dynamically
    adapting the model architecture after each experience.

    Supports continual learning by snapshotting models, transferring weights, and
    changing architectures across experiences.
    """

    def __init__(
        self,
        model_cfgs: dict = {},
        mode="last",
        fill_mode="std",
        copy_weights=True,
        freeze_backbone=False,
    ):
        """
        Initialize the DefaultEvolutionPlugin.

        Args:
            model_cfgs (dict): Mapping from experience indices to model configurations.
            mode (str): Snapshot retention mode ('last' or 'all').
            fill_mode (str): Strategy for initializing new model weights when switching.
            copy_weights (bool): If True, copies weights from the previous model.
            freeze_backbone (bool): If True, freezes the backbone layers of the new model.
        """
        super().__init__(mode=mode)
        assert isinstance(model_cfgs, dict), "model_cfgs must be a dictionary."

        self.model_cfgs = model_cfgs
        self.fill_mode = fill_mode
        self.copy_weights = copy_weights
        self.freeze_backbone = freeze_backbone

    def switch_model(self, runner, experience_index: int):
        """
        Updates the model architecture after an experience.

        Args:
            runner (Any): The object managing the training process.
            experience_index (int): The index of the current experience.
        """
        next_exp = experience_index + 1
        cfg = self.model_cfgs.get(next_exp, None)

        if cfg is None:
            runner.logger.info(
                f"[DefaultEvolutionPlugin] No change: M_{next_exp} == M_{experience_index}"
            )
            return

        runner.logger.info(f"[DefaultEvolutionPlugin] Switching to M_{next_exp}")

        # Change model architecture
        runner.switch_model(cfg)

        # Get state_dict of the last model
        last_state_dict = runner.memory.get("state_dict", None)
        if isinstance(last_state_dict, list) and len(last_state_dict) > 0:
            runner.logger.info(
                "[DefaultEvolutionPlugin] Multiple snapshots found. Only the last one will be used."
            )
            last_state_dict = last_state_dict[-1]

        # Load previous weights if available
        if self.copy_weights and last_state_dict is not None:
            runner.logger.info(
                f"[DefaultEvolutionPlugin] Initializing M_{next_exp} with M_{experience_index} weights"
            )
            load_state_dict(
                runner.module,
                last_state_dict,
                strict=False,
                logger=runner.logger,
                fill_mode=self.fill_mode,
            )

            # Freeze backbone layers if needed
            if self.freeze_backbone and hasattr(runner.module, "backbone"):
                for param in runner.module.backbone.parameters():
                    param.requires_grad = False
        elif self.copy_weights:
            runner.logger.warning(
                "[DefaultEvolutionPlugin] No previous state_dict found for weight transfer!"
            )

    def after_experience(self, runner, experience_index: int):
        """
        Hook to save snapshot and evolve model after an experience.

        Args:
            runner (Any): The object managing the model and training process.
            experience_index (int): The index of the current experience in a continual learning scenario.
        """
        super().after_experience(runner, experience_index)
        self.switch_model(runner, experience_index)
