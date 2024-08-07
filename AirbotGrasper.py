import cv2
import time
import numpy as np
import open3d as o3d
from datetime import datetime
from scipy.spatial.transform import Rotation

import airbot
import pyrealsense2 as rs
from AirbotGraspNet import AirbotGraspNet
from AirbotInterface import AirbotInterface


class AirbotGrasper(object):
    def __init__(self, visuale_model='SAM', debug=False):
        self.debug = debug
        self.H, self.W = 720, 1280
        self.P = 30
        self.init()
        self.grasp_net = AirbotGraspNet()
        self.grasp_net.set_factor_depth(1. / self.depth_scale)
        self.event_connected = False
        self.fit = None
        self.now_trans = None
        self.now_rota = None
        self.interface = AirbotInterface(visuale_model)

    def init(self):
        # ====================== Initialize Camera =======================
        print("[%s]Init RGBD Camera......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, 6)

        cfg = self.pipeline.start(config)
        depth_sensor = cfg.get_device().query_sensors()[0]
        color_sensor = cfg.get_device().query_sensors()[1]
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        depth_sensor = cfg.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        profile = cfg.get_stream(rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
        # self.intrinsic_matrix = np.array([[intr.fx, 0.0, intr.ppx],
        #                                   [0.0, intr.fy, intr.ppy],
        #                                   [0.0, 0.0, 1.0]])
        # print('intrinsic_matrix camera: ', self.intrinsic_matrix, end='\n\n')  
        #    
        # self.intrinsic_matrix = np.array([[903.481457, 0.0, 647.869078],
        #                                   [0.0, 908.882758, 350.609108],
        #                                   [0.0, 0.0, 1.0]])

        self.intrinsic_matrix = np.array([[929.4484486435581, 0, 639.864561710594],
                                          [0, 930.086640276201, 363.8675974620355],
                                          [0.0, 0.0, 1.0]])
        print('intrinsic_matrix calibration: ', self.intrinsic_matrix, end='\n\n')

        # ====================== Initialize Robot =======================
        print("[%s]Init Gripper Robot......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.translations_list = [0.24795357914631247, -0.000333295809713752, 0.23000896578243882]
        self.rotations_list = [0.03907312406344713, 0.604864292787, -0.023462915064072824, 0.7950232511718768]
        self.bot = airbot.create_agent("down", "can0", 1.0, "gripper", 'OD', 'DM')

        if not self.debug:
            self.bot.set_target_pose(self.translations_list, self.rotations_list, blocking=True)
        

    def grasp_object(self, color_image, depth_image, workspace_mask):
        '''
            Grasp Object via GraspNet
        '''
        print("[%s]Generate Grippers......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        end_points, cloud = self.grasp_net.process_data(color_image, depth_image, workspace_mask,
                                                        self.intrinsic_matrix, self.H, self.W)        
        ggs = self.grasp_net.get_grasps(end_points, cloud)
        print("[%s]Filter Grippers......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        gg = self.filter_grippers(ggs)
    
        grippers = ggs.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud])
        o3d.visualization.draw_geometries([cloud, *grippers])

        o3d.visualization.draw_geometries([cloud, gg.to_open3d_geometry()])
        # Along x-axis forward offset
        offset = 0.035
        gg_translation, gg_rotation = gg.translation, gg.rotation_matrix
        gg_translation += offset * gg_rotation[:, 0]
        gg.translation = gg_translation
        # o3d.visualization.draw_geometries([cloud, gg.to_open3d_geometry()])

        trans_marix = self.cam2base(gg)
        pred_translation, pred_rotation = trans_marix[:3, 3], trans_marix[:3, :3]
        pred_translation[0] -= 0.02
        # pred_translation[1] -= 0.015

        print('pred_translation: ', pred_translation)
        # rot_euler = Rotation.from_matrix(pred_rotation).as_euler("xyz")

        # quat: wxyz
        pred_quat = Rotation.from_matrix(pred_rotation).as_quat()
        
        if not self.debug:
            self.bot.set_target_end(1)
            # pred_translation[2] -= 0.035
            
            self.now_trans, self.now_rota = self.bot.get_current_pose()
            self.smooth_grasp(pred_translation, self.now_trans, pred_quat, self.now_rota, 100)

            self.bot.set_target_end(0)
            time.sleep(0.5)
    
    def cam2base(self, gg, cloud=None):
        '''
            Tranform gripper: Camera --> Object --> 
            Object Frame: Grasped Object
                X: Forward, Y: Right, Z: Down
            Camera Frame: Color-Depth Camera
                X: Right, Y: Down, Z: Forward
            End Frame: 
                X: Down, Y: Left, Z: Forward
        '''
        Tmat_cam2obj = np.eye(4)
        Tmat_cam2obj[:3,:3] = gg.rotation_matrix # @ Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()
        Tmat_cam2obj[:3, 3] = gg.translation

        Tmat_base2end = np.eye(4)
        Tmat_base2end[:3,:3] = Rotation.from_quat(self.rotations_list).as_matrix()
        Tmat_base2end[:3, 3] = self.translations_list

        # Static Matrix
        # Tmat_end2cam = np.array([
        #     [0.04855, -0.30770,  0.95024, -0.12346],
        #     [-0.99602, 0.05626,  0.06910, -0.01329],
        #     [-0.07472, -0.94982, -0.30375, 0.09521], 
        #     [0,         0,        0,             1]
        # ])
        Tmat_end2cam = np.array([
            [0.01679974372021955, -0.3417827195107749, 0.9396288316429802, -0.1205800830169818],
            [-0.9988576340892403, -0.04778282400951217, 0.0004780703983183782, 0.003676703639963586],
            [0.04473472289580515, -0.9385634631571248, -0.3421950177807104, 0.1103012524056419], 
            [0,         0,        0,             1]
        ])

        Tmat_base2obj = Tmat_base2end @ Tmat_end2cam @ Tmat_cam2obj
        rotation = Tmat_base2obj[:3, :3]
        rot_euler = Rotation.from_matrix(rotation).as_euler("XYZ", degrees=True)
        if abs(rot_euler[0]) > np.pi / 2.:
            if rot_euler[0] > 0.0:
                rot_euler[0] -= np.pi
            else:
                rot_euler[0] += np.pi

        if abs(rot_euler[1]) > np.pi / 2.:
            if rot_euler[1] > 0.0:
                rot_euler[1] -= np.pi
            else:
                rot_euler[1] += np.pi

        if abs(rot_euler[2]) > np.pi / 2.:
            if rot_euler[2] > 0.0:
                rot_euler[2] -= np.pi
            else:
                rot_euler[2] += np.pi

        
        Tmat_base2obj[:3, :3] = Rotation.from_euler("XYZ", rot_euler, degrees=True).as_matrix()

        if Tmat_base2obj[2, 3] < -0.09:
            # print('Out range Z')
            Tmat_base2obj[2, 3] = max(Tmat_base2obj[2,3], -0.09)


        if cloud != None:
            cloud = cloud.transform(Tmat_end2cam)
            cloud = cloud.transform(Tmat_base2end)

            return Tmat_base2obj, cloud

        return Tmat_base2obj
    
    def filter_grippers(self, grippers):
        '''
            Select candidate gripper from cache.
            Select [0, 0, -1] direction grispper
        '''
        angles = []
        # Compute Z direction angle of grisppers
        for gg in grippers:
            trans_marix = self.cam2base(gg)
            trans, rotation = trans_marix[3, :3], trans_marix[:3, :3]
            z_direction = rotation[:, 0]
            direction = np.array([0, 0, -1])
            dot_product = np.dot(z_direction, direction)
            angle = np.arccos(dot_product)
            angle_degrees = np.degrees(angle) 
            angles.append(angle_degrees)
        angles = np.array(angles)
        # print(angles.shape)
        # Filter threshold
        masks = angles < 60
        angles = angles[masks]
        grippers = grippers[masks]
        # Select best gripper
        index = np.argsort(angles)

        return grippers[int(index[0])]

    def put_back(self):
        '''
            Put object to target position.
            Two step movement for safe.
        '''
        t_back = [-0.084, -0.296, 0.189]
        r_back = [0.283, 0.177, -0.761, 0.557]

        self.now_trans, self.now_rota = self.bot.get_current_pose()
        self.smooth_grasp(t_back, self.now_trans, r_back, self.now_rota, 100)
        self.bot.set_target_end(1)
        self.now_trans, self.now_rota = self.bot.get_current_pose()
        self.smooth_grasp(self.translations_list, self.now_trans, self.rotations_list, self.now_rota, 100)

    def get_image_and_depth(self):
        '''
            Read color and depth image from stream.
            Depth is align to color image.
        '''
        time.sleep(1)
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.array(depth_frame.get_data()).astype(np.float32)
        color_image = np.array(color_frame.get_data())
        depth_imagemap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imwrite('color.png', color_image)
        cv2.imwrite('depth.png', depth_imagemap)
        cv2.imwrite('depth_origin.png', depth_image)
        img_origin = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # BGR to RGB
        return depth_image, img_origin

    def smooth_grasp(self, pred_translation, now_translation, pred_quat, now_quat, step):
        '''
            Make grasp path smoothly via linspace
        '''
        trans_x_array = np.linspace(now_translation[0], pred_translation[0], step)
        trans_y_array = np.linspace(now_translation[1], pred_translation[1], step)
        trans_z_array = np.linspace(now_translation[2], pred_translation[2], step)
        rota_init_euler = Rotation.from_quat(now_quat).as_euler('XYZ', degrees=True)
        rota_pred_euler = Rotation.from_quat(pred_quat).as_euler('XYZ', degrees=True)

        rota_lins_x = np.linspace(rota_init_euler[0], rota_pred_euler[0], step-40)
        rota_lins_y = np.linspace(rota_init_euler[1], rota_pred_euler[1], step-40)
        rota_lins_z = np.linspace(rota_init_euler[2], rota_pred_euler[2], step-40)
        rota_lins_x = np.concatenate((rota_lins_x, np.full(40,rota_lins_x[-1])))
        rota_lins_y = np.concatenate((rota_lins_y, np.full(40,rota_lins_y[-1])))
        rota_lins_z = np.concatenate((rota_lins_z, np.full(40,rota_lins_z[-1])))

        for index in range(len(trans_z_array)-30):
            translation = [trans_x_array[index+30], trans_y_array[index+30], trans_z_array[index+30]]
            rotation = Rotation.from_euler( 'XYZ', [rota_lins_x[index+30], rota_lins_y[index+30], rota_lins_z[index+30]], degrees=True).as_quat()
            self.bot.set_target_pose(translation, rotation, blocking=False, use_planning=True)
            time.sleep(1/step)

        self.bot.set_target_pose(pred_translation, pred_quat, blocking=True, use_planning=True)


    def workspace_mask(self, bbox):

        workspace_mask = np.zeros(shape=[self.H, self.W], dtype=np.bool_)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1 = x1 - self.P if x1 - self.P > 0 else 0
        y1 = y1 - self.P if y1 - self.P > 0 else 0
        x2 = x2 + self.P if x2 + self.P < self.W else self.W
        y2 = y2 + self.P if y2 + self.P < self.H else self.H       
        print(x1, x2, y1, y2)
        if x2 - x1 < 220:
            x1 = x1 - self.P if x1 - self.P > 0 else 0
            x2 = x2 + self.P if x2 + self.P < self.W else self.W
        if y2 - y1 < 220:
            y1 = y1 - self.P if y1 - self.P > 0 else 0
            y2 = y2 + self.P if y2 + self.P < self.H else self.H  
        x1, x2 = int(x1), int(x2)  
        y1, y2 = int(y1), int(y2)

        workspace_mask[y1:y2, x1:x2] = True
        
        return workspace_mask

    def run(self):
        '''
            Run the grasp pipeline.
        '''
        try:
            while 1:
                print("[%s]Read RGBD Stream......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                depth_image, color_image = self.get_image_and_depth()
                print("[%s]Select Target Object......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                mask, bbox = self.interface.segment(color_image)
                print("[%s]Grasp Select Object......" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                if len(bbox) > 0:
                    workspace_mask = self.workspace_mask(bbox)
                    # grasp object
                    self.grasp_object(color_image, depth_image, workspace_mask)
                    if not self.debug:
                        self.put_back()
                    break
        except Exception as e:
            print(f"Error during run: {e}")
            self.pipeline.stop()
