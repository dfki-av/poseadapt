_base_ = [
    "mmpose::_base_/default_runtime.py",
    "configs/_base_/models/rtmpose-t_256x192.py",
    "./poseadapt-lighting-ll.py",
]


############################################## Runtime ###############################################
randomness = dict(seed=21)

max_epochs = 10
train_cfg = dict(
    type="ContinualTrainingLoop",
    max_epochs_per_experience=[0, max_epochs],
    val_interval=2,
)


############################################ Optimizer & LR ##########################################
base_lr = 4e-3
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.0),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
]

auto_scale_lr = dict(base_batch_size=1024)


######################################### Model Components ############################################
codec = _base_.codec
backbone = _base_.backbone
head = _base_.head


########################################## Data  Pipelines ############################################
backend_args = dict(backend="local")
data_preprocessor = dict(
    type="PoseDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
)
train_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]


############################################ Model ####################################################
NUM_KPTS = 17
MODEL_PRETRAINED = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"
)
model17 = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=data_preprocessor,
    backbone=backbone,
    head=dict(**head, out_channels=NUM_KPTS),
    test_cfg=dict(flip_test=True),
    init_cfg=dict(type="Pretrained", checkpoint=MODEL_PRETRAINED),
)

########################################## Dataloaders ###############################################
NUM_WORKERS = 8
BATCH_SIZE = 32

# training sets
train_datasets = [
    dict(**_base_.dataset_train1, pipeline=train_pipeline),
    dict(**_base_.dataset_train2, pipeline=train_pipeline),
]
train_dataloaders = [
    dict(
        batch_size=NUM_WORKERS,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=True),
        dataset=d,
    )
    for d in train_datasets
]

# validation/test sets
val_datasets = [
    dict(**_base_.dataset_val1, pipeline=val_pipeline),
    dict(**_base_.dataset_val2, pipeline=val_pipeline),
]
val_dataloaders = [
    dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=d,
    )
    for d in val_datasets
]
test_dataloaders = val_dataloaders

############################################ Evaluators ##############################################
val_evaluators = [
    {{_base_.evaluator_val1}},
    {{_base_.evaluator_val2}},
]
test_evaluators = val_evaluators

############################################# Hooks ###################################################
default_hooks = dict(
    checkpoint=dict(save_best="coco/AP", rule="greater", max_keep_ckpts=1)
)

#################################### Continual Learning Plugins ######################################
model = model17  # Set initial model
custom_hooks = [
    # CL Strategy (optional)
    # If not set, the model will be finetuned naively on each experience
    # Evolution Plugin (required)
    dict(
        type="DefaultEvolutionPlugin",
        mode="last",
        model_cfgs={
            # model cfgs to be updated during evolution
            # in this case, model architecture remains unchanged
        },
    ),
]
