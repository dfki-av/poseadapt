# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 DFKI GmbH. All rights reserved.

import os
import os.path as osp
import shutil
from typing import Dict, List, Tuple, Union

from mmengine.registry import LOOPS
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.loops import TestLoop, ValLoop
from torch.utils.data import DataLoader

from ._train_loop import EpochBasedContinualTrainLoop


@LOOPS.register_module()
class ContinualTrainingLoop(BaseLoop):
    """Loop for continual learning, managing multiple datasets sequentially.

    Executes training, validation, and testing across multiple experiences.

    Args:
        runner (Runner): A reference of runner.
        num_experiences (int): Number of learning experiences.
        max_epochs_per_experience (Union[int, List, Tuple]): Training epochs per experience.
        val_interval (int): Validation interval. Defaults to 1.
    """

    def __init__(
        self,
        runner,
        num_experiences: int,
        max_epochs_per_experience: Union[int, List, Tuple],
        val_interval: int = 1,
    ) -> None:
        """Initialize ContinualTrainingLoop."""
        super().__init__(
            runner, None
        )  # No single dataloader applies to all experiences
        self.num_experiences = num_experiences
        if isinstance(max_epochs_per_experience, int):
            max_epochs_per_experience = [max_epochs_per_experience] * num_experiences
        self.max_epochs_per_experience = max_epochs_per_experience
        self.val_interval = val_interval

        self.experience_id = -1
        self.experience = None

    def get_info(self, index):
        """Get metadata for an experience.

        Args:
            index: Experience index.

        Returns:
            Dict[str, Union[int, str]]: Metadata for the experience.
        """
        return dict(
            id=index,
            name=f"Experience {index + 1}",
            max_epochs=self.max_epochs_per_experience[index],
            val_interval=self.val_interval,
        )

    def _build_dataloader(self, dataloader: Union[DataLoader, Dict]) -> DataLoader:
        """Construct a DataLoader from a config or instance.

        Args:
            dataloader (Union[DataLoader, Dict]): Dataloader or configuration dict.

        Returns:
            DataLoader: Constructed DataLoader.
        """
        if isinstance(dataloader, dict):
            return self.runner.build_dataloader(dataloader)
        return dataloader

    def build_experience_train_loop(
        self, dataloader, max_epochs, val_interval
    ) -> EpochBasedContinualTrainLoop:
        """Build training loop for a specific experience.

        Args:
            dataloader: DataLoader for the experience.
            max_epochs: Number of training epochs.
            val_interval: Validation interval.

        Returns:
            EpochBasedContinualTrainLoop: Experience training loop.
        """
        self.runner.logger.info(
            f"[ContinualTrainingLoop] Building training loop for {self.get_info(self.experience_id)['name']}"
        )

        num_batches = len(dataloader)
        if num_batches == 0:
            raise ValueError("The dataloader has no data.")

        self.runner.logger.info(
            f"[ContinualTrainingLoop] Training for {max_epochs} epochs, {num_batches} iterations per epoch, validating every {val_interval} epochs."
        )
        return EpochBasedContinualTrainLoop(
            runner=self.runner,
            dataloader=dataloader,
            max_epochs=max_epochs,
            val_interval=val_interval,
        )

    def build_experience_val_loop(self, dataloader, evaluator) -> ValLoop:
        """Build validation loop for a specific experience.

        Args:
            dataloader: Validation dataloader.
            evaluator: Evaluator instance.

        Returns:
            ValLoop: Validation loop.
        """
        # TODO: Consider using a specialized ContinualValLoop if needed
        return ValLoop(
            runner=self.runner,
            dataloader=dataloader,
            evaluator=evaluator,
        )

    def build_experience_test_loop(self, dataloader, evaluator) -> TestLoop:
        """Build test loop for a specific experience.

        Args:
            dataloader: Test dataloader.
            evaluator: Evaluator instance.

        Returns:
            TestLoop: Test loop.
        """
        # TODO: Consider using a specialized ContinualTestLoop if needed
        return TestLoop(
            runner=self.runner,
            dataloader=dataloader,
            evaluator=evaluator,
        )

    def run_experience(self, index):
        """Run a single continual learning experience.

        Args:
            index: Index of the experience.

        Returns:
            torch.nn.Module: Trained model after the experience.
        """
        info = self.get_info(index)
        info.update(self.runner.get_experience_data(index))

        if "train_dataloader" in info:
            train_loop = self.build_experience_train_loop(
                self._build_dataloader(info["train_dataloader"]),
                info["max_epochs"],
                info["val_interval"],
            )
        if "val_dataloader" in info:
            self.runner._val_loop = self.build_experience_val_loop(
                self._build_dataloader(info["val_dataloader"]),
                info["val_evaluator"],
            )
        if "test_dataloader" in info:
            self.runner._test_loop = self.build_experience_test_loop(
                self._build_dataloader(info["test_dataloader"]),
                info["test_evaluator"],
            )

        self.experience_id = index
        self.experience = train_loop
        self.dataloader = train_loop.dataloader

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.runner.optim_wrapper = self.runner.build_optim_wrapper(
            self.runner._optim_wrapper
        )
        # Automatically scaling lr by linear scaling rule
        self.runner.scale_lr(self.runner.optim_wrapper, self.runner._auto_scale_lr)

        if self.runner._param_schedulers is not None:
            self.runner.param_schedulers = self.runner.build_param_scheduler(  # type: ignore
                self.runner._param_schedulers
            )  # type: ignore

        # Initiate inner count of `optim_wrapper`.
        self.runner.optim_wrapper.initialize_count_status(
            self.runner.model,
            train_loop.iter,
            train_loop.max_iters,  # type: ignore
        )  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self.runner._maybe_compile("train_step")

        self.runner.call_hook("before_experience", experience_index=index)
        model = train_loop.run()
        self.runner._test_loop.run()
        self.runner.call_hook("after_experience", experience_index=index)

        return model

    def run(self) -> None:
        """Execute training across all experiences."""
        runner = self.runner

        self.experience_id = -1
        runner.call_hook("before_first_experience")
        base_work_dir = osp.join(runner.work_dir, "experiences")
        for index in range(self.num_experiences):
            # Create experience-specific work_dir
            experience_dir = osp.join(base_work_dir, f"{index}")
            os.makedirs(experience_dir, exist_ok=True)

            # Run the current experience
            self.run_experience(index)

            # Move all checkpoints to experience directory
            ckpt_files = [
                f
                for f in os.listdir(runner.work_dir)
                if f.endswith(".pth") and f.startswith("epoch")
            ]
            for ckpt_file in ckpt_files:
                src = osp.join(runner.work_dir, ckpt_file)
                dst = osp.join(experience_dir, ckpt_file)
                shutil.move(src, dst)

    @property
    def max_epochs(self) -> int:
        """Total training epochs for the current experience.

        Returns:
            int: Maximum epochs.
        """
        if self.experience is None:
            return self.max_epochs_per_experience[0]
        return self.experience.max_epochs

    @property
    def max_iters(self) -> int:
        """Total training iterations for the current experience.

        Returns:
            int: Maximum iterations.
        """
        if self.experience is None:
            return self.max_epochs
        return self.experience.max_iters

    @property
    def epoch(self) -> int:
        """Current epoch for the current experience.

        Returns:
            int: Current epoch.
        """
        if self.experience is None:
            return 0
        return self.experience.epoch

    @property
    def iter(self) -> int:
        """Current iteration for the current experience.

        Returns:
            int: Current iteration.
        """
        if self.experience is None:
            return 0
        return self.experience.iter
