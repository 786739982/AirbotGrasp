import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation
from AirbotGrasper import AirbotGrasper
import json
import time 

steps = 10

grasper = AirbotGrasper(debug=True)
time.sleep(3)
grasper.bot.set_target_end(1)

with open('save_pose2.json', 'r') as file:
    Tmat_list = json.load(file)
    # print(Tmat_list)

grasper.bot.set_target_pose(Tmat_list[0][0], Tmat_list[0][1], blocking=True)
time.sleep(1)
euler = Rotation.from_quat(Tmat_list[0][1]).as_euler('XYZ', degrees=True)


for i in range(steps):
    euler[0] -= 10
    print('set euler: ', euler)
    quat = Rotation.from_euler('XYZ', euler, degrees=True).as_quat()
    print('set quat: ', quat)
    grasper.bot.set_target_pose(Tmat_list[0][0], quat)
    time.sleep(0.5)
    print('get euler: ', Rotation.from_quat(grasper.bot.get_current_pose()[1]).as_euler('XYZ', degrees=True))
    print('get quat: ', grasper.bot.get_current_pose()[1])
    print('get translation: ', grasper.bot.get_current_pose()[0])
    print('---------------------------------------------')

