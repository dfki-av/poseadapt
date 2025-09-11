# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

from typing import Optional, Union

from mmengine.hooks import Hook
from mmpose.registry import HOOKS

from ..core import ContinualLearningRunner

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class BasePlugin(Hook):
    """
    Base class for plugins in the continual learning pipeline.

    Provides hook methods that can be overridden by derived plugins to inject
    custom behavior at various stages of training and evaluation.
    """

    def __init__(self):
        """
        Initialize the BasePlugin.
        """
        super().__init__()

    def before_first_experience(self, runner: ContinualLearningRunner):
        """
        Hook called before the first experience begins.

        Args:
            runner (ContinualLearningRunner): The training runner instance.
        """
        runner.logger.info(f"Loading {self.__class__.__name__}")

    def before_experience(self, runner: ContinualLearningRunner, experience_index: int):
        """
        Hook called before an experience begins.

        Args:
            runner (ContinualLearningRunner): The training runner instance.
            experience_index (int): Index of the current experience.
        """

    def after_experience(self, runner: ContinualLearningRunner, experience_index: int):
        """
        Hook called after an experience completes.

        Args:
            runner (ContinualLearningRunner): The training runner instance.
            experience_index (int): Index of the completed experience.
        """

    def before_backward(
        self,
        runner: ContinualLearningRunner,
        experience_index: int,
        losses: dict,
        data_batch: DATA_BATCH = None,
    ):
        """
        Hook called before the backward pass.

        Args:
            runner (ContinualLearningRunner): The training runner instance.
            experience_index (int): Index of the current experience.
            losses (dict): The computed loss values.
            data_batch (Optional[Union[dict, tuple, list]]): The input data batch.
        """

    def after_backward(
        self,
        runner: ContinualLearningRunner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
    ):
        """
        Hook called after the backward pass.

        Args:
            runner (ContinualLearningRunner): The training runner instance.
            batch_idx (int): Index of the current batch.
            data_batch (Optional[Union[dict, tuple, list]]): The input data batch.
        """
