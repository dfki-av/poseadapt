# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

import itertools
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import torch
from mmpose.registry import HOOKS

from poseadapt.engine import BasePlugin
from poseadapt.third_party.avalanche import ParamData, copy_params_dict

from ..common import compute_importances


@HOOKS.register_module()
class EWCPlugin(BasePlugin):
    """
    Elastic Weight Consolidation (EWC) plugin.

    EWC computes importance of each parameter at the end of training on current
    experience. During training on each mini batch, the loss is augmented
    with a penalty which keeps the value of the current weights close to the
    value they had on previous experiences in proportion to their importance
    on that experience. Importances are computed with an additional pass on the
    training set.

    This plugin does not use experience identities.

    Args:
        alpha (float): A hyperparameter that balances the base and regularization loss
                            components. Values range from 0 to 1, where higher values prioritize
                            knowledge retention over learning new information, enhancing the
                            model's stability.

        mode (str): Specifies the method of accumulating penalties across tasks.
                    - `separate`: Maintains a distinct penalty for each previous experience,
                      allowing unique regularization for each task's learned parameters.
                    - `online`: Aggregates all penalties into a single term using a decay factor,
                      which simplifies the regularization across all tasks into one evolving term.

        decay_factor (Optional[float]): A decay multiplier applied to the importance values in
                                        `online` mode. This factor reduces the influence of older
                                        tasks' importances over time, focusing more on recent tasks.
                                        This parameter is ignored if the mode is not set to `online`.

        keep_importance_data (bool): Determines whether to retain the importance values and
                                     parameter states for all tasks or just the most recent.
                                     - If True, stores importance values and parameters for every
                                       task, which increases memory usage but allows for detailed
                                       historical analysis and more precise reversions to past states.
                                     - If False, only the latest task's data is retained, minimizing
                                       memory footprint.
                                     - Note: When mode is `separate`, this is always set to True,
                                       as each task's unique data is necessary for separate penalties.
    """

    def __init__(
        self,
        alpha: float,
        mode: str = "separate",
        decay_factor: Optional[float] = 0.95,
        keep_importance_data: bool = False,
    ):
        super().__init__()
        assert alpha >= 0 and alpha <= 1, "Value of `alpha` must be in range 0 and 1"
        assert (decay_factor is not None) or (mode != "online"), (
            "You need to set `decay_factor` to use the `online` mode."
        )
        assert mode == "separate" or mode == "online", (
            "Mode must be separate or online."
        )

        self.alpha = alpha
        self.mode = mode
        self.decay_factor = decay_factor

        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params: Dict[int, Dict[str, ParamData]] = defaultdict(dict)
        self.importances: Dict[int, Dict[str, ParamData]] = defaultdict(dict)

    def before_backward(self, runner, experience_index, losses, data_batch=None):
        """
        Compute EWC penalty and add it to the loss.
        """
        super().before_backward(runner, experience_index, losses, data_batch)

        def compute_penalty(experiences):
            penalty = torch.zeros((), device=runner.device)
            for exp in experiences:
                for k, cur_param in runner.module.named_parameters():
                    # new parameters do not count
                    if k not in self.saved_params[exp]:
                        continue
                    saved_param = self.saved_params[exp][k]
                    imp = self.importances[exp][k]
                    new_shape = cur_param.shape
                    penalty += (
                        imp.expand(new_shape)
                        * (cur_param - saved_param.expand(new_shape)).pow(2)
                    ).sum()
            return penalty

        # Compute EWC penalty
        penalty = None
        if self.mode == "separate":
            penalty = compute_penalty(range(experience_index))
        elif self.mode == "online":
            if experience_index > 0:
                penalty = compute_penalty([experience_index - 1])
        else:
            raise ValueError("Incorrect EWC mode specified.")

        # Add penalty to losses dictionary
        if penalty is not None:
            losses["loss_ewc"] = self.alpha * penalty
            if "loss_kpt" in losses:
                losses["loss_kpt"] = (1 - self.alpha) * losses["loss_kpt"]

    def after_experience(self, runner, experience_index):
        """
        Compute importances of parameters after each experience.
        """
        super().after_experience(runner, experience_index)
        # Get current model
        model = runner.module
        assert model is not None

        # Compute parameter importances
        runner.logger.info(
            f"[EWCPlugin] Computing parameter importances for E_{experience_index}"
        )
        importances = compute_importances(
            model,
            runner.train_loop.dataloader,
            runner.optim_wrapper,
            runner.device,
        )
        self.update_importances(importances, experience_index)
        self.saved_params[experience_index] = copy_params_dict(model)

        # Clear previous parameter values
        if experience_index > 0 and (not self.keep_importance_data):
            del self.saved_params[experience_index - 1]

    @torch.no_grad()
    def update_importances(self, importances, t: int):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[t - 1].items(),
                importances.items(),
                fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    assert k2 is not None
                    assert curr_imp is not None
                    self.importances[t][k2] = curr_imp
                    continue

                assert k1 == k2, "Error in importance computation."
                assert curr_imp is not None
                assert old_imp is not None
                assert k2 is not None

                # Manage expansion of existing layers
                self.importances[t][k1] = ParamData(
                    f"imp_{k1}",
                    curr_imp.shape,
                    init_tensor=self.decay_factor * old_imp.expand(curr_imp.shape)
                    + curr_imp.data,
                    device=curr_imp.device,
                )

            # Clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


ParamDict = Dict[str, Union[ParamData]]
EwcDataType = Tuple[ParamDict, ParamDict]
