import os.path as osp

########################################## Dataset Settings ##########################################
data_mode = "topdown"
coco_root = "data/coco"
posebench_root = "benchmark/data/domain_incremental"

pa_type = "mode"
pa_levels = ["coco", "depth"]


def make_dataset(split, level, test_mode=False):
    """Helper to create a single dataset dict."""
    if level == "coco":
        data_root = coco_root
        split = "val"
        ann_file = f"person_keypoints_{split}2017.json"
        data_prefix = dict(img=f"{split}2017/")
    else:
        data_root = posebench_root
        ann_file = f"person_keypoints_{split}2017-{pa_type}.json"
        data_prefix = dict(img=f"images/{split}2017-{pa_type}-{level}/")

    return dict(
        type="CocoDataset",
        data_root=data_root,
        data_mode=data_mode,
        ann_file=osp.join("annotations", ann_file),
        data_prefix=data_prefix,
        test_mode=test_mode,
    )


# Train and validation datasets for each level
train_datasets = {lvl: make_dataset("train", lvl) for lvl in pa_levels}
val_datasets = {lvl: make_dataset("val", lvl, test_mode=True) for lvl in pa_levels}


def make_multi_dataset(pa_levels, is_val=False):
    """Wrapper for MultiDatasetWrapper with given pa_levels."""
    return dict(
        type="MultiDatasetWrapper",
        metainfo=dict(from_file="configs/_base_/datasets/coco.py"),
        datasets={
            lvl: (val_datasets if is_val else train_datasets)[lvl] for lvl in pa_levels
        },
        test_mode=is_val,
    )


# Training datasets (one per experience)
dataset_train1 = make_multi_dataset(pa_levels[0:1])
dataset_train2 = make_multi_dataset(pa_levels[1:2])

# Validation datasets (cumulative)
dataset_val1 = make_multi_dataset(pa_levels[:1], is_val=True)
dataset_val2 = make_multi_dataset(pa_levels[:2], is_val=True)

######################################## Evaluation Settings #########################################
coco_metric = dict(
    type="CocoMetric",
    ann_file=osp.join(coco_root, "annotations", "person_keypoints_val2017.json"),
)
coco_metric_level = dict(
    type="CocoMetric",
    ann_file=osp.join(
        posebench_root, "annotations", f"person_keypoints_val2017-{pa_type}.json"
    ),
)

# Per-level metrics
metrics = {
    lvl: (coco_metric if lvl == "coco" else {**coco_metric_level, "prefix": lvl})
    for lvl in pa_levels
}


def make_evaluator(pa_levels):
    if len(pa_levels) == 1:
        return metrics[pa_levels[0]]
    return dict(
        type="MultiDatasetEvaluatorV2",
        metrics=[metrics[lvl] for lvl in pa_levels],
        datasets=[val_datasets[lvl] for lvl in pa_levels],
        keys=pa_levels,
    )


# Evaluators
evaluator_val1 = make_evaluator(pa_levels[:1])
evaluator_val2 = make_evaluator(pa_levels[:2])
