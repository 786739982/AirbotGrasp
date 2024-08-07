import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation
from AirbotGrasper import AirbotGrasper
import json

grasper = AirbotGrasper(debug=True)

Tmat_end2cam = np.array([
            [0.01679974372021955, -0.3417827195107749, 0.9396288316429802, -0.1185800830169818],
            [-0.9988576340892403, -0.04778282400951217, 0.0004780703983183782, 0.003676703639963586],
            [0.04473472289580515, -0.9385634631571248, -0.3421950177807104, 0.1163012524056419], 
            [0,         0,        0,             1]
        ])

count = 0
Tmat_list = []

while (1):
    key = input('S: save cloud, Q: quit, P: print end pose \n')
    if key == 's':
        pose = grasper.bot.get_current_pose()
        depth_image, color_image = grasper.get_image_and_depth()
        print(pose)
        Tmat_base2end = np.eye(4)
        Tmat_base2end[:3,:3] = Rotation.from_quat(pose[1]).as_matrix()
        Tmat_base2end[:3, 3] = pose[0]
        depth = np.array(Image.open('depth_origin.png'), dtype=np.float32)
        workspace_mask = np.ones_like(depth, dtype=bool)
        print(workspace_mask.shape) 
        print(type(workspace_mask))
        workspace_mask[:, :] = True 
        end_points, cloud = grasper.grasp_net.process_data(color_image, depth_image, workspace_mask,
                                                        grasper.intrinsic_matrix, 720, 1280)
        cloud = cloud.transform(Tmat_end2cam)
        cloud = cloud.transform(Tmat_base2end)
        o3d.io.write_point_cloud(f"cloud_trans_{count}.ply", cloud)
        print(f'Save cloud_trans_{count}.ply successfully! ')
        count += 1
        Tmat_list.append(pose)
    elif key == 'p':
        print('translation: ', grasper.bot.get_current_pose()[0])
    elif key == 'q':
        with open('save_pose.json', 'w') as file:
            json.dump(Tmat_list, file)
        print('Save save_pose.json successfully! ')
        break