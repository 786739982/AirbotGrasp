import numpy as np
import open3d as o3d
from AirbotGrasper import AirbotGrasper
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation


intrinsic_matrix = np.array([[929.4484486435581, 0, 639.864561710594],
                                          [0, 930.086640276201, 363.8675974620355],
                                          [0.0, 0.0, 1.0]])


Tmat_end2cam = np.array([
            [0.01679974372021955, -0.3417827195107749, 0.9396288316429802, -0.1185800830169818],
            [-0.9988576340892403, -0.04778282400951217, 0.0004780703983183782, 0.003676703639963586],
            [0.04473472289580515, -0.9385634631571248, -0.3421950177807104, 0.1163012524056419], 
            [0,         0,        0,             1]
        ])


# Tmat_end2cam = np.array([
#             [0.04855, -0.30770,  0.95024, -0.12346],
#             [-0.99602, 0.05626,  0.06910, -0.01329],
#             [-0.07472, -0.94982, -0.30375, 0.09521], 
#             [0,         0,        0,             1]
#         ])


# addition_euler = [0., 6., 4.5]
# addition_matrix = Rotation.from_euler('xyz', addition_euler, degrees=True).as_matrix()
# Tmat_add = np.eye(4)
# Tmat_add[:3, :3] = addition_matrix
# print('Tmat_add: ', Tmat_add)


# Tmat_add_init = [[-0.64053597, -0.64594599,  0.41529872, -0.06120419],
#  [-0.50902638, -0.047802,   -0.85942356,  0.09287453],
#  [ 0.57498988, -0.76188773 ,-0.29818273, -0.14541181],
#  [ 0.     ,     0.     ,     0.  ,        1.        ]]

# # no.2
# Tmat_end2cam = np.eye(4)
# quat_end2cam = [0.39776377333694335, -0.6052033658379471, 0.5582691995613924, -0.40478187634855783] 
# Tmat_end2cam[:3, :3] = Rotation.from_quat(quat_end2cam).as_matrix()
# Tmat_end2cam[:3, 3] = [-0.134065, 0.00379009, 0.109446]

translations_list = [0.145, -0.042, 0.2054]
rotations_list = [-0.061, 0.497, -0.059, 0.863]

Tmat_base2end = np.eye(4)
Tmat_base2end[:3,:3] = Rotation.from_quat(rotations_list).as_matrix()
Tmat_base2end[:3, 3] = translations_list

depth = np.array(Image.open('depth_origin.png'), dtype=np.float32)

workspace_mask = np.ones_like(depth, dtype=bool)
print(workspace_mask.shape)
workspace_mask[:, :] = True 

grasper = AirbotGrasper()

depth_image, color_image = grasper.get_image_and_depth()
end_points, cloud = grasper.grasp_net.process_data(color_image, depth_image, workspace_mask,
                                                        intrinsic_matrix, 720, 1280)

# cloud = cloud.transform(np.linalg.inv(Tmat_add_init))
cloud = cloud.transform(Tmat_end2cam)
cloud = cloud.transform(Tmat_base2end)
# cloud = cloud.transform(Tmat_add)


# print( Tmat_add @ np.linalg.inv(Tmat_base2end) @ np.linalg.inv(Tmat_end2cam))

o3d.io.write_point_cloud("cloud_trans.ply", cloud)
print('save successfully!')