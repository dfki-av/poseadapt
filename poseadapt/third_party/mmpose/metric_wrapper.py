# SPDX-License-Identifier: Apache-2.0
# Derived from MMPose (Apache-2.0).
# Modifications: This file is based on the CocoMetric class from MMPose,
# and abstracts the keypoint conversion process into a generic metric wrapper
# to work with any metric.
# Original project: https://github.com/open-mmlab/mmpose

# Copyright (c) 2025 DFKI GmbH. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Dict, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS, TRANSFORMS


@METRICS.register_module()
class MetricWrapper(BaseMetric):
    """Wrapper for evaluation metrics with optional converters.

    Allows wrapping a metric with preprocessing steps for predictions
    and ground truths.

    Args:
        metric (Dict): Configuration for the underlying metric.
        pred_converter (Dict, optional): Configuration for converting predictions.
            Follows the same parameters as 'KeypointConverter'. Defaults to None.
        gt_converter (Dict, optional): Configuration for converting ground truths.
            Follows the same parameters as 'KeypointConverter'. Defaults to None.
    """

    def __init__(
        self,
        metric: Dict,
        pred_converter: Dict = None,
        gt_converter: Dict = None,
    ) -> None:
        """Initialize the MetricWrapper.

        Args:
            metric (Dict): Metric configuration.
            pred_converter (Optional[Dict], optional): Prediction converter config.
                Defaults to None.
            gt_converter (Optional[Dict], optional): Ground truth converter config.
                Defaults to None.
        """
        metric = METRICS.build(metric)
        super().__init__(collect_device=metric.collect_device, prefix=metric.prefix)
        self._metric = metric

        # Build converters
        self.pred_converter = (
            TRANSFORMS.build(pred_converter) if pred_converter else None
        )
        self.gt_converter = TRANSFORMS.build(gt_converter) if gt_converter else None

    @property
    def dataset_meta(self) -> Optional[Dict]:
        """Dataset meta information.

        Returns:
            Optional[Dict]: Meta information of the dataset.
        """
        return self._metric._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: Dict) -> None:
        """Set dataset meta information.

        Args:
            dataset_meta (dict): Meta info to assign to the wrapped metric.
        """
        self._metric.dataset_meta = dataset_meta

    def process(self, data_batch: Sequence[Dict], data_samples: Sequence[Dict]) -> None:
        """Process a batch of predictions and ground truths.

        The results are stored in ``self.results`` for later metric computation.

        Args:
            data_batch (Sequence[dict]): A batch of input data from the dataloader.
            data_samples (Sequence[dict]): A batch of predictions and ground truths.

        Returns:
            None.
        """
        for sample in data_samples:
            if self.pred_converter is not None:
                sample["pred_instances"] = self.pred_converter(sample["pred_instances"])

            if self.gt_converter is not None:
                sample["gt_instances"] = self.gt_converter(sample["gt_instances"])

        self._metric.process(data_batch, data_samples)
        self.results = self._metric.results

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute evaluation metrics from processed results.

        Args:
            results (list): Processed results from each batch.

        Returns:
            Dict[str, float]: Computed metrics, where keys are metric names and
            values are the corresponding results.
        """
        return self._metric.compute_metrics(results)

    def evaluate(self, size: int) -> Dict:
        """Evaluate the overall performance after all batches are processed.

        Args:
            size (int): Size of the validation dataset. Used to handle padding
                when batch size > 1.

        Returns:
            dict: Evaluation results where keys are metric names and values are
            the computed scores.
        """
        # evaluate wrapped metric
        metrics = self._metric.evaluate(size)

        # reset the results list
        self.results.clear()

        # return the evaluation results
        return metrics
