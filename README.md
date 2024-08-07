# GraspAnything

## Hardware
* Realsense D435i

* Airbot v2.8,3

## Work Space
Hardware Set Up
```
-------------------
|        |        |
|       | |       |
|                 |
-------------------
```
Grasp Range
```
X : [ 20,  50]
Y : [-20,  20]
Z : [-10,   0]
```
Observation View
```
Translations : [0.20, 0.00, 0.30]
Rotations : [0.0, 1.0, 0.0, 0.15]
```

## Camera Calibration
- Intristic Matrix: ROS
- Extristic Matrix: Hand-Eye

## Install
We strongly recommend `Cuda 11.8` and `torch 2.3.0`.
```
# Environment
conda create -n grasp python=3.10
conda activate grasp

# pytorch
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# GraspNet
cd GraspNet

cd pointnet2
python setup.py install

cd knn
python setup.py install

cd graspnetAPI
pip install .

# ultralytics
pip install ultralytics

# SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# pyrealsense2
pip install pyrealsense2
```


## Runs
```
python demo.py
```

## To Do

* Interaction: 
    - One Click, SAM Debug