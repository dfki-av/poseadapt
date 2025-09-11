# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

import torch
import torch.nn.functional as F
from mmpose.registry import HOOKS

from poseadapt.engine import BasePlugin


@HOOKS.register_module()
class LFLPlugin(BasePlugin):
    """Less-Forgetful Learning (LFL) Plugin.

    LFL satisfies two properties to mitigate catastrophic forgetting.
        1) To keep the decision boundaries unchanged
        2) The feature space should not change much on target (new) data

    LFL uses euclidean loss between features from current and previous version
    of model as regularization to maintain the feature space and avoid
    catastrophic forgetting.

    This plugin does not use task identities.

    Args:
        alpha (float or list[float]): Euclidean loss hyper parameter.
            If a list is provided, it should have the same length as the number
            of experiences. If a single value is provided, it will be used for
            all experiences.

    See Also:
        - Less-forgetting Learning in Deep Neural Networks
          https://arxiv.org/abs/1607.00122
    """

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def before_experience(self, runner, experience_index: int):
        super().before_experience(runner, experience_index)
        prev_model = runner.last_model
        if experience_index > 0:
            assert prev_model is not None, (
                "LFL needs a previous snapshot. Include a *EvolutionPlugin."
            )

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        """
        Add euclidean loss between prev and current features as penalty
        """
        current_model = runner.module
        prev_model = runner.last_model
        alpha = (
            self.alpha[experience_index]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )

        if prev_model is None:
            return

        # Extract features
        feats_curr = current_model.extract_feat(data_batch["inputs"])
        with torch.no_grad():
            feats_prev = prev_model.extract_feat(data_batch["inputs"])

        # Compute LFL penalty (euclidean distance between features)
        penalty = 0.0
        for f_curr, f_prev in zip(feats_curr, feats_prev):
            penalty += F.mse_loss(f_curr, f_prev)

        # Add penalty to losses dictionary
        losses["loss_lfl"] = alpha * penalty
        if "loss_kpt" in losses:
            losses["loss_kpt"] = (1 - alpha) * losses["loss_kpt"]
