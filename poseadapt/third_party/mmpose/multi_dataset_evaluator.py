# SPDX-License-Identifier: Apache-2.0
# Derived from MMPose (Apache-2.0).
# Modifications: Adds a new `keys` parameter to the MultiDatasetEvaluator class
# to avoid relying on dataset names from metainfo,
# addressing https://github.com/open-mmlab/mmpose/issues/3061.
# Original project: https://github.com/open-mmlab/mmpose

# Copyright (c) 2025 DFKI GmbH. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, Sequence, Union

from mmengine.evaluator.evaluator import Evaluator
from mmengine.evaluator.metric import BaseMetric
from mmengine.structures import BaseDataElement
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import DATASETS, EVALUATORS


@EVALUATORS.register_module()
class MultiDatasetEvaluatorV2(Evaluator):
    """Wrapper class to compose multiple BaseMetric instances for different datasets.

    This evaluator allows routing evaluation data to the correct metric
    based on dataset keys, avoiding reliance on metainfo dataset names.

    Args:
        metrics (Union[dict, BaseMetric, Sequence]): Configurations or instances of metrics.
        datasets (Sequence[dict]): Sequence of dataset configs. Each entry must align
            with a metric in `metrics`.
        keys (Sequence[str]): Unique identifiers for each dataset, used to route
            data samples to the correct metric.
    """

    def __init__(
        self,
        metrics: Union[dict, BaseMetric, Sequence],
        datasets: Sequence[dict],
        keys: Sequence[str],
    ):
        """Initialize the evaluator with metrics, datasets, and dataset keys.

        Args:
            metrics (Union[dict, BaseMetric, Sequence]): Metric configurations or instances.
            datasets (Sequence[dict]): Dataset configurations.
            keys (Sequence[str]): Unique dataset identifiers.

        Raises:
            AssertionError: If the number of metrics does not match the number of datasets.
        """
        assert len(metrics) == len(
            datasets
        ), "The number of metrics must match the number of datasets"

        super().__init__(metrics)

        # Initialize mapping: dataset_name -> metric
        metrics_dict = {}
        for dataset_name, dataset, metric in zip(keys, datasets, self.metrics):
            metainfo_file = DATASETS.module_dict[dataset["type"]].METAINFO
            dataset_meta = parse_pose_metainfo(metainfo_file)
            metric.dataset_meta = dataset_meta
            metrics_dict[dataset_name] = metric

        self.metrics_dict = metrics_dict

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Meta information for the evaluator.

        This global dataset meta is not used per-dataset but may be set
        for general metadata.

        Returns:
            Optional[dict]: Dataset meta information if available.
        """
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set global dataset meta information.

        Args:
            dataset_meta (dict): Global dataset meta info.
        """
        self._dataset_meta = dataset_meta

    def process(
        self, data_samples: Sequence[BaseDataElement], data_batch: Optional[Any] = None
    ):
        """Process predictions and ground truths for each metric.

        Converts `BaseDataElement` instances to dictionaries and routes
        them to the corresponding dataset-specific metric.

        Args:
            data_samples (Sequence[BaseDataElement]): Model predictions and
                ground truths for evaluation.
            data_batch (Optional[Any], optional): Batch of inputs and samples
                from the dataloader.

        Raises:
            ValueError: If a data sample does not contain a 'dataset_name' key
                or if the dataset name is not recognized among registered metrics.

        Returns:
            None.
        """
        _data_samples = defaultdict(list)
        _data_batch = dict(
            inputs=defaultdict(list),
            data_samples=defaultdict(list),
        )

        for inputs, data_ds, data_sample in zip(
            data_batch["inputs"], data_batch["data_samples"], data_samples
        ):
            if isinstance(data_sample, BaseDataElement):
                data_sample = data_sample.to_dict()
            assert isinstance(data_sample, dict)

            dataset_name = data_sample.get("dataset_name", None)
            if dataset_name is None:
                raise ValueError(
                    "Each data_sample must include a 'dataset_name' key "
                    "matching one of the registered datasets."
                )
            else:
                if dataset_name not in self.metrics_dict:
                    raise ValueError(
                        f"Unrecognized dataset name '{dataset_name}'. "
                        f"Available datasets are: "
                        f"{list(self.metrics_dict.keys())}"
                    )

            _data_samples[dataset_name].append(data_sample)
            _data_batch["inputs"][dataset_name].append(inputs)
            _data_batch["data_samples"][dataset_name].append(data_ds)

        for dataset_name, metric in self.metrics_dict.items():
            if dataset_name in _data_samples:
                data_batch_for_ds = dict(
                    inputs=_data_batch["inputs"][dataset_name],
                    data_samples=_data_batch["data_samples"][dataset_name],
                )
                metric.process(data_batch_for_ds, _data_samples[dataset_name])
