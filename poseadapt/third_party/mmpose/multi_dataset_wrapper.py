# SPDX-License-Identifier: Apache-2.0
# Derived from MMPose (Apache-2.0).
# Modifications: Based on the CombinedDataset class from MMPose,
# modified to use explicit dataset keys for routing,
# addressing https://github.com/open-mmlab/mmpose/issues/3061.
# Original project: https://github.com/open-mmlab/mmpose

# Copyright (c) 2025 DFKI GmbH. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.registry import build_from_cfg
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import DATASETS


@DATASETS.register_module()
class MultiDatasetWrapper(BaseDataset):
    """A wrapper for combining multiple datasets into a single dataset.

    Supports optional resampling to adjust the relative sampling ratios
    of the sub-datasets.

    Args:
        metainfo (dict): Meta information for the combined dataset.
        datasets (dict[str, list]): Mapping from dataset names to dataset configs.
        pipeline (List[Union[dict, Callable]], optional): Processing pipeline.
            Defaults to [].
        sample_ratio_factor (Optional[List[float]], optional): Sampling ratio factors
            for each dataset. If provided, controls the resampling rate of sub-datasets.
            Defaults to None.
        **kwargs: Additional arguments passed to the parent BaseDataset.
    """

    def __init__(
        self,
        metainfo: dict,
        datasets: dict[str, list],
        pipeline: List[Union[dict, Callable]] = [],
        sample_ratio_factor: Optional[List[float]] = None,
        **kwargs,
    ):
        """Initialize the MultiDatasetWrapper.

        Args:
            metainfo (dict): Combined dataset meta information.
            datasets (dict[str, list]): Mapping from dataset names to configs.
            pipeline (List[Union[dict, Callable]], optional): Data processing pipeline.
                Defaults to [].
            sample_ratio_factor (Optional[List[float]], optional): Sampling ratio factors
                for each dataset. Defaults to None.
            **kwargs (Any): Extra keyword arguments for BaseDataset.

        Raises:
            AssertionError: If the length of `sample_ratio_factor` does not
                match the number of datasets.
            AssertionError: If any sampling ratio factor is negative.
        """
        self.keys = list(datasets.keys())
        datasets = list(datasets.values())

        self.datasets = []
        self.resample = sample_ratio_factor is not None

        for cfg in datasets:
            dataset = build_from_cfg(cfg, DATASETS)
            self.datasets.append(dataset)

        self._lens = [len(dataset) for dataset in self.datasets]
        if self.resample:
            assert len(sample_ratio_factor) == len(datasets), (
                f"the length "
                f"of `sample_ratio_factor` {len(sample_ratio_factor)} does "
                f"not match the length of `datasets` {len(datasets)}"
            )
            assert min(sample_ratio_factor) >= 0.0, (
                "the ratio values in " "`sample_ratio_factor` should not be negative."
            )
            self._lens_ori = self._lens
            self._lens = [
                round(l * sample_ratio_factor[i]) for i, l in enumerate(self._lens_ori)
            ]

        self._len = sum(self._lens)

        super(MultiDatasetWrapper, self).__init__(pipeline=pipeline, **kwargs)
        self._metainfo = parse_pose_metainfo(metainfo)

    @property
    def metainfo(self):
        """Meta information of the combined dataset.

        Returns:
            dict: A deep copy of the combined dataset meta information.
        """
        return deepcopy(self._metainfo)

    @property
    def lens(self):
        """Lengths of each sub-dataset.

        Returns:
            List[int]: A deep copy of the list of dataset lengths.
        """
        return deepcopy(self._lens)

    def __len__(self):
        """Total number of samples across all sub-datasets.

        Returns:
            int: The combined length of the dataset.
        """
        return self._len

    def _get_subset_index(self, index: int) -> Tuple[int, int]:
        """Resolve a global index into a sub-dataset index and local index.

        Args:
            index (int): Global sample index.

        Returns:
            Tuple[int, int]:
                - subset_index: Index of the sub-dataset.
                - local_index: Local sample index within the sub-dataset.

        Raises:
            ValueError: If the index is out of bounds.
        """
        if index >= len(self) or index < -len(self):
            raise ValueError(
                f"index({index}) is out of bounds for dataset with "
                f"length({len(self)})."
            )

        if index < 0:
            index = index + len(self)

        subset_index = 0
        while index >= self._lens[subset_index]:
            index -= self._lens[subset_index]
            subset_index += 1

        if self.resample:
            gap = (self._lens_ori[subset_index] - 1e-4) / self._lens[subset_index]
            index = round(gap * index + np.random.rand() * gap - 0.5)

        return subset_index, index

    def prepare_data(self, idx: int) -> Any:
        """Prepare a data sample using the pipeline.

        Args:
            idx (int): Global sample index.

        Returns:
            Any: Processed data sample, depends on the pipeline.
        """

        data_info = self.get_data_info(idx)

        # the assignment of 'dataset' should not be performed within the
        # `get_data_info` function. Otherwise, it can lead to the mixed
        # data augmentation process getting stuck.
        data_info["dataset"] = self

        return self.pipeline(data_info)

    def get_data_info(self, idx: int) -> dict:
        """Get raw annotation information by global index.

        Includes dataset name and relevant meta information fields.

        Args:
            idx (int): Global sample index.

        Returns:
            dict: Annotation information for the specified index.
        """
        subset_idx, sample_idx = self._get_subset_index(idx)
        # Get data sample processed by ``subset.pipeline``
        data_info = self.datasets[subset_idx][sample_idx]

        if "dataset" in data_info:
            data_info.pop("dataset")

        # Set dataset name
        data_info["dataset_name"] = self.keys[subset_idx]

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            "upper_body_ids",
            "lower_body_ids",
            "flip_pairs",
            "dataset_keypoint_weights",
            "flip_indices",
        ]

        for key in metainfo_keys:
            data_info[key] = deepcopy(self._metainfo[key])

        return data_info

    def full_init(self):
        """Fully initialize all sub-datasets.

        Returns:
            None.
        """

        if self._fully_initialized:
            return

        for dataset in self.datasets:
            dataset.full_init()
        self._fully_initialized = True
