# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Author: Saif Khan (c) 2025

import math
from typing import Iterator, Optional, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


@DATA_SAMPLERS.register_module()
class SubsetSampler(Sampler[int]):
    """Restrict a dataset to a subset of N samples.

    Works in both distributed and non-distributed environments.

    Args:
        dataset (Sized): The dataset.
        num_samples (int): Number of samples to use (N <= len(dataset)).
        shuffle (bool): Whether to shuffle indices. Defaults to True.
        seed (int, optional): Random seed used to shuffle. If None, uses
            `sync_random_seed()`. Defaults to None.
    """

    def __init__(
        self,
        dataset: Sized,
        num_samples: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        # clamp to dataset length
        self.num_samples_total = min(int(num_samples), len(self.dataset))
        # ensure even split across ranks
        self.num_samples = math.ceil(self.num_samples_total / world_size)
        self.total_size = self.num_samples * world_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices for this rank."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # restrict to subset size
        indices = indices[: self.num_samples_total]

        # pad up if needed for even division
        if len(indices) < self.total_size:
            indices = (indices * (self.total_size // len(indices) + 1))[
                : self.total_size
            ]

        # slice by rank
        indices = indices[self.rank : self.total_size : self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        """Number of samples for this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
