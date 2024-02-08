import torch
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from path import Path

# This function is borrowed from IDR: https://github.com/lioryariv/idr


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def compute_P_from_KT(K, T):

    P = torch.matmul(K, torch.linalg.inv(T))

    return P


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        self.device = torch.device('cuda')

        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')
        self.match_folder = conf.get_string('match_folder')

        print(f'Load data: Begin from {self.data_dir}')

        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        if len(self.images_lis) < 1:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.jpg')))
        if len(self.images_lis) < 1:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.png')))

        self.n_images = len(self.images_lis)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.intrinsics_all_inv = []
        self.pose_all = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]

            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics = torch.from_numpy(intrinsics).float()
            self.intrinsics_all.append(intrinsics)
            self.intrinsics_all_inv.append(torch.linalg.inv(intrinsics))
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.stack(self.intrinsics_all_inv).to(self.device)
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

        # Object scale mat: region of interest to **extract mesh**
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01,  1.01,  1.01, 1.0])
        self.object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ self.object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ self.object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        # load images
        images_np = []
        for im_name in self.images_lis:
            img = cv.imread(im_name)
            images_np.append(img)
        images_np = np.stack(images_np) / 256.0
        self.images = torch.from_numpy(images_np).float().permute(0, 3, 1, 2).to(self.device)  # [n_images, 3, H, W]
        del images_np

        self.H, self.W = self.images.shape[2], self.images.shape[3]

        # load gt pose for validation
        gt_camera_dict = np.load(os.path.join(self.data_dir, 'cameras.npz'))
        gt_world_mats_np = [gt_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.gt_pose_all = []
        for world_mat in gt_world_mats_np:
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.gt_pose_all.append(torch.from_numpy(pose).float())
        self.gt_pose_all = np.stack(self.gt_pose_all)  # [n_images, 4, 4]

        # two_view files
        two_view_files = sorted((Path(self.data_dir)/self.match_folder).files('*.npz'))
        self.two_views_all = []
        for f in two_view_files:
            self.two_views_all.append(np.load(f, allow_pickle=True))

        print('Load data: End')

    @torch.no_grad()
    def gen_rays_at(self, img_idx, pose_net, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3

        pose = pose_net(img_idx)
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).to(self.device)
        trans = torch.from_numpy(pose[:3, 3]).to(self.device)
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def gen_random_rays_at(self, img_idx, batch_size, pose):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).to(self.device)
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).to(self.device)

        color = self.images[img_idx].permute(1, 2, 0)[(pixels_y, pixels_x)]    # batch_size, 3

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3

        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape)  # batch_size, 3

        return torch.cat([rays_o, rays_v, color], dim=-1)  # mask    # batch_size, 10

    def get_gt_pose(self):
        return self.gt_pose_all

    def sample_matches(self, img_idx, pose_net, num_pairs=20, max_matches=5000):
        # ref frame
        pose = pose_net(img_idx)

        two_view = self.two_views_all[img_idx]
        num_src = len(two_view['src_idx'])

        match_list = []
        intrinsic_src_list = []
        pose_src_list = []
        for id in torch.randperm(num_src)[:num_pairs]:
            src_idx = two_view['src_idx'][id]

            match = two_view['match'][id]

            # downsample matches if there are too much
            if match.shape[0] > max_matches:
                match = match[np.random.randint(match.shape[0], size=max_matches)]

            match = torch.from_numpy(match).float().to(self.device)

            pose_src = pose_net(src_idx)

            match_list.append(match)
            pose_src_list.append(pose_src)
            intrinsic_src_list.append(self.intrinsics_all[src_idx])

        intrinsic = self.intrinsics_all[img_idx]
        return intrinsic, pose, intrinsic_src_list, pose_src_list, match_list
