

# AirbotGrasp

AirbotGrasp is a customizable, automated grasping framework designed to be compatible with arbitrary vision-based detection and segmentation systems. Built upon the grasp pose generation methodology provided by GraspNet, AirbotGrasp offers a unified interface and an interactive graphical user interface (GUI) to facilitate rapid integration with diverse visual algorithms and application scenarios. 

Furthermore, we introduce several engineering optimizations tailored to tabletop environments, including filtering of invalid grasp poses, point cloud scaling for small objects, trajectory planning and smoothing, and an emergency stop algorithm for robotic arms. And modified the PointNet++ CUDA code to make it compatible with PyTorch 2.0 and later versions.

Additionally, the framework provides a subjective evaluation tool for assessing calibration accuracy.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/786739982/AirbotGrasp/">
    <img src="assets/logo.png" alt="Logo" width="146" height="64">
  </a>

  <h3 align="center">AirbotGrasp</h3>
  <p align="center">
    A Customizable, Automated Grasping Framework！
    <br />
    <a href="https://github.com/786739982/AirbotGrasp"><strong>Explore the documentation of this project »</strong></a>
    <br />
    <br />
    <a href="https://github.com/786739982/AirbotGrasp">Demo</a>
    ·
    <a href="https://github.com/786739982/AirbotGrasp/issues">Report Bug</a>
    ·
    <a href="https://github.com/786739982/AirbotGrasp/issues">Propose New Feature</a>
  </p>

</p>

<p align="center">
<img src="assets/airbotgrasp.gif", width="300"/>
</p>

## 目录

- [Getting-Started](#Getting-Started)
  - [Requirements](#Requirements)
  - [Installation](#Installation)
  - [Deployment](#Deployment)
- [Directory-Structure](#Directory-Structure)
- [Customization-Guide](#Customization-Guide)
  - Add a New Vision Algorithm
  - Integrate Your Own Robotic Arm
- [Author](#Author)
- [Acknowledgements](#Acknowledgements)




### Getting-Started

#### Requirements

1. Python 3
2. PyTorch 2.6
3. CUDA 12.8
4. Open3d >=0.8
5. TensorBoard 2.3

#### **Installation**

Get the code.
```bash
git clone https://github.com/786739982/AirbotGrasp
cd AirbotGrasp/GraspNet
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

#### Deployment

Hardware

* Realsense D435i
* AirbotPlay v2.8,3

通过手眼标定获得相机相对于机械臂末端的旋转矩阵，并替换 ```AirbotGrasper.py``` 中的代码：
```
  Tmat_end2cam = np.array([
            [0.01679974372021955, -0.3417827195107749, 0.9396288316429802, -0.1205800830169818],
            [-0.9988576340892403, -0.04778282400951217, 0.0004780703983183782, 0.003676703639963586],
            [0.04473472289580515, -0.9385634631571248, -0.3421950177807104, 0.1103012524056419], 
            [0,         0,        0,             1]
        ])
```

Run the Pipeline.
```
  python3 Pipeline.py
```




### Customization-Guide

#### Add a New Vision Algorithm
添加新的视觉算法类似于 ```AirbotSegment.py``` ，并修改 ```AirbotInterface.py``` 中的代码：
```
  def init_predictor(self):
        if self.type_predictor == 'SAM':
            Predictor = AirbotSegment()
            self.predictor = Predictor.get_model()
        elif self.type_predictor == 'Yolo-World': # You can use your own predictor model
            pass
```

#### Set the matrix of the camera relative to the robotic arm end-effector.
通过手眼标定获得相机相对于机械臂末端的旋转矩阵，并替换 ```AirbotGrasper.py``` 中的代码：
```
  Tmat_end2cam = np.array([
            [0.01679974372021955, -0.3417827195107749, 0.9396288316429802, -0.1205800830169818],
            [-0.9988576340892403, -0.04778282400951217, 0.0004780703983183782, 0.003676703639963586],
            [0.04473472289580515, -0.9385634631571248, -0.3421950177807104, 0.1103012524056419], 
            [0,         0,        0,             1]
        ])
```

#### Integrate Your Own Robotic Arm
添加新的机械臂，并修改 ```AirbotGrasper.py``` 中的代码：
```
  # You can use your own robot
  self.bot = airbot.create_agent("down", "can0", 1.0, "gripper", 'OD', 'DM') 
```

#### Integrate Your Own RGBD-Camera

```
  def get_image_and_depth(self):
      '''
          Read color and depth image from stream.
          Depth is align to color image.
      '''
      pass
```



### Directory-Structure
eg:

```
filetree 
├── AirbotAccuracy.py  # 主观法到位精度测试
├── AirbotCollect.py  # 重力补偿模式下的点云数据采集
├── AirbotGrasper.py  # 抓取全流程
├── AirbotGraspNet.py  # 测试GraspNet模型
├── AirbotInterface.py  # UI交互界面以及视觉框架接口
├── AirbotSegment.py  # 分割算法接口
├── AirbotTipVerify.py  # 末端到位精度验证
├── AirbotUtils  # 工具类
├── GraspNet  # 适配 PyTorch 2.0 and later versions.
├── images  # Logo Images
├── Pipeline.py  # main pipeline
├── README.md  # README.md
└── yolo_sam.py  # YoloWorld + SAM

```




### Author

Hongrui Zhu 

E-Mail：786739982@qq.com or hongrui0226@gmail.com

qq:786739982

vx：Hong_Rui_0226



  
### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/786739982/AirbotGrasp/blob/master/LICENSE.txt)





### Acknowledgements

- [DISCOVERSE](https://airbots.online/)




<!-- links -->
[contributors-shield]: https://img.shields.io/github/contributors/786739982/AirbotGrasp.svg?style=flat-square
[contributors-url]: https://github.com/786739982/AirbotGrasp/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/786739982/AirbotGrasp.svg?style=flat-square
[forks-url]: https://github.com/786739982/AirbotGrasp/network/members
[stars-shield]: https://img.shields.io/github/stars/786739982/AirbotGrasp.svg?style=flat-square
[stars-url]: https://github.com/786739982/AirbotGrasp/stargazers
[issues-shield]: https://img.shields.io/github/issues/786739982/AirbotGrasp.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/786739982/AirbotGrasp.svg
[license-shield]: https://img.shields.io/github/license/786739982/AirbotGrasp.svg?style=flat-square
[license-url]: https://github.com/786739982/AirbotGrasp/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555





