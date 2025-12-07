conda create -n poseadapt python=3.10
conda activate poseadapt

# Numpy 1.26.4 + OpenCV (headless) - pre-install to avoid conflicts
pip install numpy==1.26.4 opencv-python-headless

# PyTorch 2.1.2 + torchvision 0.16.2 + CUDA 12.1
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# MMCV 2.1.0
pip install openmim
mim install mmcv==2.1.0

# MMDetection 3.2.0
mim install "mmdet==3.2.0"

## MMPose 1.3.0
mim install "mmpose==1.3.0"
