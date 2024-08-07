import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image

from GraspNet.models.graspnet import GraspNet as GraspModel, pred_decode
from GraspNet.utils.collision_detector import ModelFreeCollisionDetector
from GraspNet.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup


class AirbotGraspNet(object):
    def __init__(self) -> None:
        self.num_view = 300
        self.num_point = 20000
        self.collision_thresh = 0.01
        self.voxel_size = 0.01
        self.factor_depth = 1000
        self.model = self.get_net()
        self.is_scale = False

    def set_factor_depth(self, factor_depth):
        self.factor_depth = factor_depth

    def get_net(self):
        # Init the model
        net = GraspModel(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                         cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load('./checkpoint/checkpoint_dist.tar',
                                map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint (epoch: %d)" % start_epoch)
        net.eval()
        return net

    def get_grasps(self, end_points, cloud):
        # Forward pass
        with torch.no_grad():
            end_points = self.model(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        # Collision detection
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        # Grasp post process
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def process_data(self, color, depth, workspace_mask, intrinsic, h, w):
        color = color.astype(np.float32) / 255.0
        depth = depth.astype(np.float32)
        # generate cloud
        camera = CameraInfo(w, h, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], self.factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        # get valid points
        if workspace_mask is None:
            # mask = (depth < self.factor_depth)
            mask = (np.ones_like(depth).astype(np.bool_) & (depth > 0))
        else:
            # mask = (workspace_mask & (depth < self.factor_depth))
            mask = (workspace_mask & (depth > 0.2*self.factor_depth) & (depth < 0.5*self.factor_depth))
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # Scale Point If it is small
        width, length, height = self.compute_dimensions(cloud_masked)
        print("width,length, height: ", width,length, height)
        if width < 0.01 and length < 0.01 and height < 0.01:
            print('It is too small!')
            self.is_scale = True
            cloud_masked *= 1.2

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        o3d.io.write_point_cloud("cloud.ply", cloud)

        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        

        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled
        return end_points, cloud

    def vis_grasps(self, gg, cloud):
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def compute_dimensions(self, point_cloud):
        """
        :input:  point_cloud: Nx3
        :return: width, length, height
        """
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)

        width = max_coords[0] - min_coords[0]  
        length = max_coords[1] - min_coords[1] 
        height = max_coords[2] - min_coords[2]
        
        return width, length, height

    

if __name__ == '__main__':
    image = cv2.imread('./GraspNet/doc/example_data/color.png')
    depth = np.array(Image.open('./GraspNet/doc/example_data/depth.png'), dtype=np.float32)
    print(depth.max(), depth.min())
    h, w = 720, 1280
    grasp = AirbotGraspNet()
    intrinsic = np.array([[637.91, 0., 639.65],
                          [0., 637.91, 391.311],
                          [0., 0., 1.]])
    mask = np.ones_like(depth).astype(np.bool_)
    endpoints, cloud = grasp.process_data(image, depth, mask, intrinsic, h, w)

    gg = grasp.get_grasps(endpoints, cloud)
    
    grasp.vis_grasps(gg, cloud)

    np.savez('gripper.npz', translations=gg.translations, 
                            rotation_matrices=gg.rotation_matrices, 
                            heights=gg.heights, widths=gg.widths)
 