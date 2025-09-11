# PoseAdapt: Toolkit for Continual Adaptation of Human Pose Estimation Models

<div align="center">

[![Home](https://img.shields.io/badge/Project-Home-311B92.svg?labelColor=311B92&style=plastic)](https://saifkhichi96.github.io/research/poseadapt/)
[![Conference](https://img.shields.io/badge/WACV-2026-003366.svg?labelColor=white&style=plastic)](#)
[![Dataset](https://img.shields.io/badge/Dataset-PoseAdaptBench-black.svg?labelColor=white&style=plastic)](https://drive.google.com/drive/folders/1aT7LpZqMEzXs_Knpz6kbgjCwQXeVdTDb?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2504.08110-B31B1B.svg?labelColor=white&style=plastic)](https://arxiv.org/abs/2409.20469)
![PoseAdapt Teaser](teaser.png)
</div>

> __Abstract__: Human pose estimators are typically retrained from scratch or naively fine-tuned whenever keypoint sets, sensing modalities, or deployment domains change--an inefficient, compute-intensive practice that rarely matches field constraints. We present PoseAdapt, an open-source framework and benchmark suite for continual pose model adaptation. PoseAdapt defines domain-incremental and class-incremental tracks that simulate realistic changes in density, lighting, and sensing modality, as well as skeleton growth. The toolkit supports two workflows: (i) Strategy Benchmarking, which lets researchers implement continual learning (CL) methods as plugins and evaluate them under standardized protocols; and (ii) Model Adaptation, which allows practitioners to adapt strong pretrained models to new tasks with minimal supervision. We evaluate representative regularization-based methods in single-step and sequential settings. Benchmarks enforce a fixed lightweight backbone, no access to past data, and tight per-step budgets. This isolates adaptation strategy effects, highlighting the difficulty of maintaining accuracy under strict resource limits. PoseAdapt connects modern CL techniques with practical pose estimation needs, enabling adaptable models that improve over time without repeated full retraining.

## Overview

PoseAdapt is an open-source framework for benchmarking continual learning strategies in human pose estimation and adapting pretrained models to new skeletons and domains with minimal supervision.

- [Installation](#installation)
- [Usage](#usage)

---

## Installation

PyTorch 2.1.0 and MMPose 1.3.0 are required to run this project. Please follow the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html) to set up the environment, and then install PoseAdapt:

```bash
pip install [-e] .
```

Use the `-e` flag to install the package in editable mode when contributing to the codebase (e.g., implementing new continual learning strategies). We recommend using a virtual environment such as `venv` or `conda` to avoid dependency conflicts.

## Usage

Please refer to the [User Guide](docs/user_guide.md) for detailed instructions on using PoseAdapt for both adapting pretrained models and benchmarking continual learning strategies.

---

## Citation
If you use PoseAdapt in your research, please cite the following paper:

```bibtex
@inproceedings{khan2026poseadapt,
  title={PoseAdapt: Toolkit for Continual Adaptation of Human Pose Estimation Models},
  author={Muhammad Saif Ullah Khan and Didier Stricker},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

## License

- **Code**: Licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).  
  Non-commercial scientific and educational use is permitted. Commercial use requires separate permission.  

- **Third-Party Code**:  
  - Portions derived from [MMPose](https://github.com/open-mmlab/mmpose), licensed under Apache License 2.0.  
  - Portions adapted from [Avalanche](https://avalanche.continualai.org/), licensed under the MIT License.  

- **Datasets**:  
  - Benchmark subsets are derived from [COCO](https://cocodataset.org), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).  
  - See `benchmarks/data/ATTRIBUTION.md` for details on modifications and attribution.  

See the `NOTICE` file for a summary of licensing and attribution.
