# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

import copy

from mmpose.registry import HOOKS

from poseadapt.third_party.avalanche import freeze_everything

from ..base import BasePlugin


@HOOKS.register_module()
class BaseEvolutionPlugin(BasePlugin):
    """
    A plugin that enables model snapshots after each experience.

    Stores a copy of the model after each experience, which can be used
    for distillation or other continual learning strategies. Designed to be
    extended by other evolution strategy plugins.
    """

    def __init__(self, mode="last"):
        """
        Initialize the BaseEvolutionPlugin.

        Args:
            mode (str): Snapshot retention mode ('last' for latest, 'all' for all snapshots).
        """
        super().__init__()
        assert mode in ["last", "all"], "mode must be either 'last' or 'all'"

        self.mode = mode

    def _copy_model_state(self, runner):
        """Creates a frozen deep copy of the model's state_dict."""
        prev_model = copy.deepcopy(runner.module)
        state_dict = prev_model.state_dict()
        freeze_everything(prev_model)
        return prev_model, state_dict

    def _save_snapshot(self, runner):
        """Saves a single model snapshot (mode='last')."""
        runner.memory = runner.memory or {}
        model, state_dict = self._copy_model_state(runner)
        runner.memory["model"] = model
        runner.memory["state_dict"] = state_dict

    def _append_snapshot(self, runner):
        runner.memory = runner.memory or {}

        if "model" not in runner.memory:
            runner.memory["model"] = []

        if "state_dict" not in runner.memory:
            runner.memory["state_dict"] = []

        model, state_dict = self._copy_model_state(runner)
        runner.memory["model"].append(model)
        runner.memory["state_dict"].append(state_dict)

    def save_snapshot(self, runner):
        """
        Manages snapshot saving based on mode ('last' or 'all').

        Args:
            runner (Any): The training runner containing the model and memory attributes.

        Raises:
            ValueError: If the mode is not one of the supported values.
        """
        if self.mode == "last":
            self._save_snapshot(runner)
        elif self.mode == "all":
            self._append_snapshot(runner)
        else:
            raise ValueError("Invalid mode for BaseEvolutionPlugin.")

    def before_experience(self, runner, experience_index: int):
        """
        Hook called before an experience begins.

        Args:
            runner (Any): The training runner instance.
            experience_index (int): Index of the current experience.
        """
        # if experience_index == 1 and runner._val_loop is not None:
        #     runner._val_loop.run()

    def after_experience(self, runner, experience_index):
        """
        Hook to save snapshot after an experience.

        Args:
            runner (Any): The training runner instance.
            experience_index (int): Index of the completed experience.
        """
        super().after_experience(runner, experience_index)
        self.save_snapshot(runner)
