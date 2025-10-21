from PIL import Image
import json
from easydict import EasyDict as edict
import numpy as np
import os, torch
from transforms import SE3

def load_cam(idx, image_path):
    image = np.asarray(Image.open(image_path))
    cam = edict(
        idx=idx,
        image=(torch.from_numpy(image.copy()).float()/255),
        width=image.shape[1],
        height=image.shape[0],
        image_name=os.path.split(image_path)[1]
    )
    return cam
    
class Loader():
    @staticmethod
    def load_cams(opt):
        base_path = opt.source
        image_list = sorted([f for f in os.listdir(base_path) if f.endswith('.png')])
        
        opt.data_num = len(image_list)
        print("Data number:", opt.data_num)
        
        image_paths = [os.path.join(base_path, f) for f in image_list]
        
        cams = [load_cam(idx, image_path) for (idx, image_path) in enumerate(image_paths)]
        
        # intrinsic, extrinsic, ext0
        root_path = '/'.join(base_path.split('/')[:-1])
        scene = (base_path.split('/')[-1]).split('-')[0]
        calib_path = os.path.join(root_path, "calibs", f"{scene.zfill(2)}.txt")
        intr, gt_extr, ext0 = Loader.load_calib(calib_path)
        poses = Loader.load_poses(opt, root_path, scene, ext0)
        
        _, _, init_extr = Loader.load_calib_json(base_path)
        # poses = Loader.load_poses2(base_path, ext0)
        
        for i, cam in enumerate(cams):
            cam.update(
                intr=intr,
                pose = poses[i])
            projection_matrix = Loader.getProjectionMatrix(0.01, 100, cam)
            cam.update(projection_matrix=projection_matrix)
        return cams, gt_extr, init_extr

    @staticmethod
    def getProjectionMatrix(znear, zfar, cam):
        """
        #
        # Copyright (C) 2023, Inria
        # GRAPHDECO research group, https://team.inria.fr/graphdeco
        # All rights reserved.
        #
        # This software is free for non-commercial, research and evaluation use 
        # under the terms of the LICENSE.md file.
        #
        # For inquiries contact  george.drettakis@inria.fr
        #
        """
        tanHalfFovX = cam.width / (cam.intr.mat[0, 0] * 2)
        tanHalfFovY = cam.height / (cam.intr.mat[1, 1] * 2)

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4).cuda()

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P


    @staticmethod
    def load_poses(opt, path, scene, ext0):
        pose_path = os.path.join(path, "poses", f"{scene.zfill(2)}.txt")
        poses_np = np.loadtxt(pose_path)[:opt.data_num]
        poses = SE3(mat=torch.from_numpy(poses_np).float().reshape(-1, 3, 4).cuda())
        poses = poses @ ext0
        # breakpoint()
        
        poses = poses[0].invert() @ poses
        return poses.invert()
    
    @staticmethod
    def load_poses2(path, extr):
        pose_path = os.path.join(path, "LiDAR_poses.txt")
        poses_np = np.loadtxt(pose_path).reshape(-1, 4, 4)
        poses_np = extr.mat.cpu().numpy() @ poses_np
        poses = SE3(mat=torch.from_numpy(poses_np).float().cuda())
                
        poses = poses[0].invert() @ poses
        return poses.invert()
    
    @staticmethod
    def load_calib(calib_file):
        trinsics = torch.from_numpy(np.loadtxt(calib_file, usecols=range(1,13)).reshape(-1, 3, 4)).float()
        ext0 = SE3(mat=trinsics[4].cuda()) # Tr
        intr = SE3(R=trinsics[2,:,:3].cuda()) # P2
        cam2_0 = trinsics[2,0,3] / trinsics[2,0,0]
        extr = trinsics[4]
        extr[0, 3] += cam2_0
        extr = SE3(mat=extr.cuda())
        
        return intr, extr, ext0
    
    @staticmethod
    def load_calib_json(path):
        intr_file = os.path.join(path, "camera-intrinsic.json")
        intr_data = np.array(json.load(open(intr_file, 'r')))
        intr = SE3(R=torch.from_numpy(intr_data).float().cuda())
        
        extr_file = os.path.join(path, "LiDAR-to-camera.json")
        extr_data = json.load(open(extr_file, 'r'))
        extr_gt_np = np.array(extr_data["correct"])
        extr_init_np = np.array(extr_data["from_lidar"])

        extr_gt = SE3(mat=torch.from_numpy(extr_gt_np).float().cuda())
        extr_init = SE3(mat=torch.from_numpy(extr_init_np).float().cuda())
        
        return intr, extr_gt, extr_init
    
class CustomLoader():    
    @staticmethod
    def load_custom_intrinsics(path):
        intrinsics = {}
        current_cam = None
        
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("# CAM"):
                    current_cam = line.split(":")[0][2:].strip()
                    intrinsics[int(current_cam[-1])] = []
                elif current_cam:
                    intrinsics[int(current_cam[-1])].append([float(x) for x in line.split()])
        
        # Convert lists to matrices
        for cam in intrinsics:
            intrinsics[cam] = np.array([
                intrinsics[cam][0],
                intrinsics[cam][1],
                intrinsics[cam][2],
            ])
        return intrinsics

    @staticmethod
    def load_custom_extrinsics(path):
        interval = 1
        poses = []
        with open(path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if i % interval == 0:
                    pose = np.array([float(x) for x in line.split()]).reshape(4, 4)
                    poses.append(pose)
        poses = np.asarray(poses).reshape(-1, 4, 4)
        return poses

    @staticmethod
    def load_custom_rigs(path):
        rigs = {}
        with open(path, 'r') as f:        
            for line in f:
                values = list(map(float, line.strip().split()))
                cam_id = int(values[0])                         # First is ID
                rig = np.array(values[1:]).reshape(4, 4)        # Next 16 are 4x4 matrix
                rigs[cam_id] = rig
        return list(rigs.values())

    @staticmethod
    def load_cams(opt):
        base_path = os.path.join(opt.source, 'images', f'image_{str(opt.cam_id).zfill(2)}')
        image_list = sorted([f for f in os.listdir(base_path) if f.endswith('.png')])
        
        opt.data_num = len(image_list)
        print("Data number:", opt.data_num)
        
        image_paths = [os.path.join(base_path, f) for f in image_list]
        cams = [load_cam(idx, image_path) for (idx, image_path) in enumerate(image_paths)]
        
        intrinsic_path = os.path.join(opt.source, "params", "intrinsics.txt")
        intr = CustomLoader.load_custom_intrinsics(intrinsic_path)[int(opt.cam_id)]
        intr = SE3(R=torch.from_numpy(intr).float().cuda())
        
        cam2lidar_path = os.path.join(opt.source, "params", "cams_to_lidar_gt.txt")
        gt_extr = CustomLoader.load_custom_rigs(cam2lidar_path)[int(opt.cam_id)]
        gt_extr = SE3(mat=torch.from_numpy(np.linalg.inv(gt_extr)).float().cuda())

        cam2lidar_path = os.path.join(opt.source, "params", "cams_to_lidar_init.txt")
        init_extr = CustomLoader.load_custom_rigs(cam2lidar_path)[int(opt.cam_id)]
        init_extr = SE3(mat=torch.from_numpy(np.linalg.inv(init_extr)).float().cuda())
        
        lidar_poses_path = os.path.join(opt.source, "params", "lidars.txt")
        poses = CustomLoader.load_custom_extrinsics(lidar_poses_path)
        poses = SE3(mat=torch.from_numpy(poses).float().cuda()) # l2w
        
        # Make identity the first pose     
        poses = poses[0].invert() @ poses
        poses = poses.invert()
        
        for i, cam in enumerate(cams):
            cam.update(
                intr=intr,
                pose = poses[i]) # w2l
            projection_matrix = Loader.getProjectionMatrix(0.01, 100, cam)
            cam.update(projection_matrix=projection_matrix)
        return cams, gt_extr, init_extr

    @staticmethod
    def getProjectionMatrix(znear, zfar, cam):
        """
        #
        # Copyright (C) 2023, Inria
        # GRAPHDECO research group, https://team.inria.fr/graphdeco
        # All rights reserved.
        #
        # This software is free for non-commercial, research and evaluation use 
        # under the terms of the LICENSE.md file.
        #
        # For inquiries contact  george.drettakis@inria.fr
        #
        """
        tanHalfFovX = cam.width / (cam.intr.mat[0, 0] * 2)
        tanHalfFovY = cam.height / (cam.intr.mat[1, 1] * 2)

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4).cuda()

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

class VODB():
    def __init__(self, opt) -> None:
        base_path = opt.source
        if opt.cam_id == '-1':
            np_path = os.path.join(base_path, 'superpoint-superglue.npy')
        else:
            np_path = os.path.join(base_path, 'images', f'superpoint-superglue_{str(opt.cam_id).zfill(2)}.npy')
        self.features = []
        self.scores = []
        try:
            with open(np_path, "rb") as f:
                for _ in range(opt.data_num - 1):
                    a = np.load(f)
                    self.features.append((a[:, :2], a[:, 2:4]))
                    self.scores.append(a[:, -1])
        except FileNotFoundError:
            print(f"Error: File not found at {np_path}")
        except Exception as e:
            print(f"An error occurred while loading features: {e}")
    
    def load_corres_pixels(self, img1, img2):
        img1, img2 = int(img1.split(".")[0]), int(img2.split(".")[0])
        if img1 < img2:
            return self.features[img1]
        else:
            a, b = self.features[img2]
            return b, a