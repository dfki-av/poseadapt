# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.registry import LOOPS
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.utils.data import DataLoader


@LOOPS.register_module()
class EpochBasedContinualTrainLoop(EpochBasedTrainLoop):
    """Loop for epoch-based training in continual learning.

    Extends EpochBasedTrainLoop with a 'before_backward' hook for modifying
    the loss function before backpropagation. This supports continual learning
    strategies like EWC and LWF.

    Note:
        Models requiring a custom train_step are not supported.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Union[DataLoader, Dict]): A dataloader object or dict.
        max_epochs (int): Total training epochs.
        val_begin (int): Epoch to begin validation. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (Optional[List[Tuple[int, int]]]): Milestones and
            intervals for dynamic validation. Defaults to None.
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """Initialize EpochBasedContinualTrainLoop."""
        super().__init__(
            runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals
        )

    def _train_step(
        self, runner, idx, data: Union[dict, tuple, list]
    ) -> Dict[str, torch.Tensor]:
        """Run a single training step with a before-backward hook.

        Performs forward pass, loss computation, applies before-backward hook,
        backward pass, and parameter update.

        Args:
            runner: The runner managing the training.
            idx: Index of the current iteration.
            data (Union[dict, tuple, list]): A batch of data.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of logging tensors.
        """
        model = runner.module

        # Enable automatic mixed precision training context.
        with runner.optim_wrapper.optim_context(self):
            data = model.data_preprocessor(data, True)
            losses = model._run_forward(data, mode="loss")  # type: ignore

        # Call before_backward hook
        # This hook is used by regularization-based plugins to add
        # a penalty to the loss for mitigating forgetting
        runner.call_hook(
            "before_backward",
            experience_index=runner.train_loop.experience_id,
            losses=losses,
            data_batch=data,
        )

        parsed_losses, log_vars = model.parse_losses(losses)  # type: ignore
        loss = runner.optim_wrapper.scale_loss(parsed_losses)
        runner.optim_wrapper.backward(loss)
        self.runner.call_hook("after_backward", batch_idx=idx, data_batch=data)

        if runner.optim_wrapper.should_update():
            runner.optim_wrapper.step()
            runner.optim_wrapper.zero_grad()

        return log_vars

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            idx: Index of the current iteration.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook("before_train_iter", batch_idx=idx, data_batch=data_batch)

        outputs = self._train_step(self.runner, idx, data_batch)

        self.runner.call_hook(
            "after_train_iter", batch_idx=idx, data_batch=data_batch, outputs=outputs
        )
        self._iter += 1
