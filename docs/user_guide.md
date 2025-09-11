# PoseAdapt User Guide

This guide provides detailed instructions on how to use the PoseAdapt framework for adapting pretrained human pose estimation models to new skeletons and domains, as well as benchmarking continual learning strategies.

## Data Preparation

PoseAdapt works with all [MMPose-supported 2D keypoint datasets](https://mmpose.readthedocs.io/en/latest/user_guides/prepare_datasets.html). Please follow the instructions in the MMPose documentation to download and prepare your dataset of choice when adapting models to new domains or skeletons.

For strategy benchmarking, we provide preprocessed subsets of the COCO dataset that simulate the strict testing conditions described in our paper. Download ([Drive](https://drive.google.com/drive/folders/1aT7LpZqMEzXs_Knpz6kbgjCwQXeVdTDb?usp=sharing) | ([Zenodo](#))) and extract this data to the `benchmark/data/` directory.

## Adapting Pretrained Models

TBA.

## Benchmarking CL Strategies

TBA.

## Adding a New CL Strategy

TBA.
