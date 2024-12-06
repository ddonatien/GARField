import os
import logging
from glob import glob

import trimesh
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

import torch
import torch.nn.functional as F
from torch.torch_version import TorchVersion

from .LieAlgebra import se3


class MultiviewDataset:
    def __init__(self, conf):
        super().__init__()
        logging.getLogger("trimesh").setLevel(logging.ERROR)
        self.device = torch.device("cuda")
        self.conf = conf

        self.data_dir = conf.get_string("data_dir")
        self.render_cameras_name = conf.get_string("render_cameras_name")
        self.object_cameras_name = conf.get_string("object_cameras_name")

        self.camera_static = conf.get_bool("camera_static")
        self.camera_trainable = conf.get_bool("camera_trainable")

        self.camera_outside_sphere = conf.get_bool(
            "camera_outside_sphere", default=True
        )
        self.scale_mat_scale = conf.get_float("scale_mat_scale", default=1.1)
        self.sim_meshes = []

        self.use_depth = conf.get_bool("use_depth")
        if self.use_depth:
            self.depth_scale = conf.get_float("depth_scale", default=1000.0)

        self.omesh_path = os.path.join(self.data_dir, "simple_foot_nocap.stl")
        self.rmeshes_path = os.path.join(
            self.data_dir, "robot_mesh_scaled/collision_*.stl"
        )
        self.scene_params = np.load(
            os.path.join(self.data_dir, "scene_params.npz"), allow_pickle=True
        )

        self.cameras_dict = {}
        self.images_lis = []
        self.masks_lis = []
        self.depths_lis = []
        images_dict = {}
        masks_dict = {}
        depths_dict = {}

        self.n_views = 0
        for cam in os.scandir(self.data_dir):
            if not os.path.isdir(cam.path):
                continue
            if cam.name.startswith("."):
                continue
            if "mesh" in cam.name:
                continue
            if "vid" in cam.name:
                continue
            self.n_views += 1
            camera_dict = np.load(os.path.join(cam.path, self.render_cameras_name))
            self.cameras_dict[cam.name] = camera_dict
            tmp_imglis = sorted(glob(os.path.join(cam.path, "image/*.png")))
            tmp_masklis = sorted(glob(os.path.join(cam.path, "mask/*.png")))
            tmp_depthlis = sorted(glob(os.path.join(cam.path, "depth/*.png")))
            images_dict[cam.name] = tmp_imglis
            masks_dict[cam.name] = tmp_masklis
            depths_dict[cam.name] = tmp_depthlis

        for cam in sorted(images_dict):
            self.images_lis.append(images_dict[cam])
            self.masks_lis.append(masks_dict[cam])
            self.depths_lis.append(depths_dict[cam])

        self.n_images = len(self.images_lis[0])
        for u, _ in enumerate(self.images_lis):
            assert len(self.images_lis[u]) == self.n_images

        self.images_np = np.stack(
            [
                np.stack([cv.imread(im_name) for im_name in lis]) / 256.0
                for lis in self.images_lis
            ]
        )
        self.masks_np = np.stack(
            [
                np.stack([cv.imread(im_name) for im_name in lis]) / 256.0
                for lis in self.masks_lis
            ]
        )

        if self.use_depth:
            self.depths_np = np.stack(
                [
                    np.stack(
                        [cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in lis]
                    )
                    / self.depth_scale
                    for lis in self.depths_lis
                ]
            )
            self.depths_np[self.depths_np == 0] = -1.0  # avoid nan values
            self.depths = torch.from_numpy(self.depths_np.astype(np.float32)).cpu()

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        if self.camera_static:
            # world_mat is a projection matrix from world to image
            world_mats = []
            scale_mats = []
            camera_mats = []
            camera_poses = []
            for cam in sorted(self.cameras_dict):
                data = self.cameras_dict[cam]
                world_mats.append([data["world_mat_0"].astype(np.float32)])
                camera_mats.append([data["camera_mat_0"].astype(np.float32)])
                camera_poses.append([data["camera_pose_0"].astype(np.float32)])
                scale_mats.append([data["scale_mat_0"].astype(np.float32)])
            self.world_mats_np = np.stack(world_mats)
            self.scale_mats_np = np.stack(scale_mats)
            self.camera_mats_np = np.stack(camera_mats)
            self.camera_poses_np = np.stack(camera_poses)

        self.intrinsics_all = []
        self.pose_all = []
        self.scales_all = []

        for u in range(self.world_mats_np.shape[0]):
            intr_lis = []
            pse_lis = []
            scl_lis = []
            for scale_mat, cam_mat, cam_pos in zip(
                self.scale_mats_np[u], self.camera_mats_np[u], self.camera_poses_np[u]
            ):
                intr = np.eye(4)
                intr[:3, :3] = cam_mat
                pose = np.eye(4)
                pose[:3, :3] = np.transpose(cam_pos[:3, :3])
                tmp = np.eye(4)
                tmp[:3, :] = cam_pos
                pose[:3, 3] = np.linalg.inv(scale_mat[:3, :3]) @ (
                    np.linalg.inv(tmp)[:3, 3] - scale_mat[:3, 3]
                )
                intr_lis.append(torch.from_numpy(intr).float())
                pse_lis.append(torch.from_numpy(pose).float())
                scl_lis.append(torch.from_numpy(scale_mat).float())
            self.intrinsics_all.append(torch.stack(intr_lis))
            self.pose_all.append(torch.stack(pse_lis))
            self.scales_all.append(torch.stack(scl_lis))

        if os.path.exists(os.path.join(self.data_dir, "points.npy")):
            self.parts_gt = np.load(os.path.join(self.data_dir, "points.npy"))
            self.parts_gt = torch.from_numpy(self.parts_gt).float()
            self.parts_gt = self.parts_gt.to(self.device)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(
            self.device
        )  # [n_cams, n_images, H, W, 3]
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(
            self.device
        )  # [n_cams, n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(
            self.device
        )  # [n_cams, n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(
            self.intrinsics_all
        )  # [n_cams, n_images, 4, 4]
        self.focal = self.intrinsics_all[:][0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(
            self.device
        )  # [n_cams, n_images, 4, 4]
        self.scales_all = torch.stack(self.scales_all).to(
            self.device
        )  # [n_cams, n_images, 4, 4]
        self.H, self.W = self.images.shape[-3], self.images.shape[-2]
        self.image_pixels = self.H * self.W

        pose_paras = []
        for u in range(self.world_mats_np.shape[0]):
            pose_paras.append(se3.log(self.pose_all[u]))
        self.poses_paras = pose_paras
        if self.camera_trainable:
            for u, _ in enumerate(self.poses_paras):
                self.poses_paras[u].requires_grad_()

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        for cam in os.scandir(self.data_dir):
            if not os.path.isdir(cam.path):
                continue
            if cam.name.startswith("."):
                continue
            if "mesh" in cam.name:
                continue
            if "calibration" in cam.name:
                continue
            object_scale_mat = np.load(
                os.path.join(cam.path, self.object_cameras_name)
            )["scale_mat_0"]
            object_bbox_min = (
                np.linalg.inv(self.scale_mats_np[0][0])
                @ object_scale_mat
                @ object_bbox_min[:, None]
            )
            object_bbox_max = (
                np.linalg.inv(self.scale_mats_np[0][0])
                @ object_scale_mat
                @ object_bbox_max[:, None]
            )
            self.object_bbox_min = object_bbox_min[:3, 0]
            self.object_bbox_max = object_bbox_max[:3, 0]
            break

        print("Load data: End")

    def load_pose_from_mat(self, matrices):
        pose_paras = []
        for mat in matrices:
            pose_paras.append(se3.log(mat))
            pose_paras[-1].to(self.device)
        self.poses_paras = torch.stack(pose_paras, dim=0).unsqueeze(1)

    def pose_paras_to_mat(self, img_idx, cam_id):
        if self.camera_static:
            img_idx = 0
        if self.camera_trainable:
            pose_paras = self.poses_paras[cam_id][img_idx, :]
            pose = se3.exp(pose_paras)
        else:
            pose = self.pose_all[cam_id, img_idx]
        return pose.squeeze()

    def gen_rays_at(self, img_idx, resolution_level=1, cam_id=-1):
        """
        Generate rays at world space from one camera.
        """
        if cam_id == -1:
            cam_id = np.random.randint(self.n_views)
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        pixels_x = pixels_x.int()
        pixels_y = pixels_y.int()
        if torch.__version__ > TorchVersion("1.13.0+cu117"):
            color = self.images[cam_id][img_idx][(pixels_y, pixels_x)]  # batch_size, 3
            mask = self.masks[cam_id][0][(pixels_y, pixels_x)]  # batch_size, 3
        else:
            try:
                color = self.images[cam_id][img_idx][
                    (pixels_y.long(), pixels_x.long())
                ]  # batch_size, 3
                mask = self.masks[cam_id][0][
                    (pixels_y.long(), pixels_x.long())
                ]  # batch_size, 3
            except IndexError as e:
                print("IndexError")
                print(cam_id, img_idx, pixels_y.dtype, pixels_x.dtype)
                raise e
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        ).float()  # W, H, 3
        if self.camera_static:
            img_idx = 0
        if self.camera_trainable:
            pose = self.pose_paras_to_mat(img_idx, cam_id)
        else:
            pose = self.pose_all[cam_id][img_idx]
        p = torch.matmul(
            self.intrinsics_all_inv[cam_id][img_idx, None, None, :3, :3],
            p[:, :, :, None],
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(
            pose[None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), color, mask

    def gen_rays_at_depth(self, img_idx, resolution_level=1, cam_id=-1):
        """
        Generate rays at world space from one camera.
        """
        if cam_id == -1:
            cam_id = np.random.randint(self.n_views)
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        # Normalize
        npixels_x = (pixels_x - (self.W - 1) / 2) / ((self.W - 1) / 2)
        npixels_y = (pixels_y - (self.H - 1) / 2) / ((self.H - 1) / 2)
        mask = (
            self.masks[cam_id][img_idx].permute(2, 0, 1)[None, ...].to(self.device)
        )  # 1, 3, H, W
        mask = mask[:, :1, ...]  # 1, 1, H, W
        depth = self.depths[cam_id][img_idx][None, None, ...].to(
            self.device
        )  # 1, 1, H, W
        npixels = torch.cat(
            [
                npixels_x.unsqueeze(-1).unsqueeze(0),
                npixels_y.unsqueeze(-1).unsqueeze(0),
            ],
            dim=-1,
        )  # 1, W, H, 2
        # grid_sample: sample image on (x_i, y_i)
        mask = F.grid_sample(
            mask, npixels, mode="nearest", padding_mode="border", align_corners=True
        ).squeeze()[
            ..., None
        ]  # W, H, 1
        depth = F.grid_sample(
            depth, npixels, padding_mode="border", align_corners=True
        ).squeeze()[
            ..., None
        ]  # W, H, 1
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        p_d = p.clone().detach()
        # Camera
        if self.camera_static:
            img_idx = 0
        if self.camera_trainable:
            pose = self.pose_paras_to_mat(img_idx, cam_id)
        else:
            pose = self.pose_all[cam_id][img_idx]
        p = torch.matmul(
            self.intrinsics_all_inv[cam_id][img_idx, None, None, :3, :3],
            p[:, :, :, None],
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(
            pose[None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        p_d = (
            depth
            * torch.matmul(
                self.intrinsics_all_inv[cam_id][img_idx, None, None, :3, :3],
                p_d[:, :, :, None],
            ).squeeze()
            * (1 / self.scales_all[cam_id][img_idx, :][0, 0])
        )  # W, H, 3
        rays_s = torch.matmul(
            pose[None, None, :3, :3], p_d[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return (
            rays_o.transpose(0, 1),
            rays_v.transpose(0, 1),
            rays_s.transpose(0, 1),
            mask.transpose(0, 1),
        )

    def get_surf_pts(self, img_idx, cam_id, n_pts=200):
        path = os.path.join(self.data_dir, f"cam{cam_id}", "pcds", f"{img_idx:03}.ply")
        pcd = trimesh.load(path)
        idx = np.arange(pcd.vertices.shape[0])
        np.random.shuffle(idx)
        pts = pcd.vertices[idx[:n_pts], :]
        if self.camera_static:
            img_idx = 0
        return torch.tensor(pts).float() / self.scales_all[cam_id, img_idx, 0, 0]

    def gen_random_rays_at(self, img_idx, batch_size, cam_id=-1):
        """
        Generate random rays at world space from one camera.
        """
        if cam_id == -1:
            rays = []
            for c in range(self.n_views):
                rays.append(
                    self.gen_random_rays_at(
                        img_idx, batch_size // self.n_views, cam_id=c
                    )
                )
            return torch.cat(rays, dim=0)  # batch_size - batch_size % n_views, 14
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[cam_id][img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[cam_id][img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        ).float()  # batch_size, 3
        if self.camera_static:
            img_idx = 0
        if self.camera_trainable:
            pose = self.pose_paras_to_mat(img_idx, cam_id)
        else:
            pose = self.pose_all[cam_id][img_idx]
        p = torch.matmul(
            self.intrinsics_all_inv[cam_id][img_idx, None, :3, :3], p[:, :, None]
        ).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(
            pose[None, :3, :3], rays_v[:, :, None]
        ).squeeze()  # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1)  # batch_size, 10

    def gen_random_rays_at_depth(self, img_idx, batch_size, cam_id=-1):
        """
        Generate random rays at world space from one camera.
        """
        if cam_id == -1:
            rays = []
            for c in range(self.n_views):
                rays.append(
                    self.gen_random_rays_at_depth(
                        img_idx, batch_size // self.n_views, cam_id=c
                    )
                )
            return torch.cat(rays, dim=0)  # batch_size - batch_size % n_views, 14
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[cam_id][img_idx].to(self.device)
        mask = self.masks[cam_id][img_idx].to(self.device)
        depth = self.depths[cam_id][img_idx].to(self.device)
        color = color[(pixels_y, pixels_x)]  # batch_size, 3
        mask = mask[(pixels_y, pixels_x)]  # batch_size, 3
        depth = depth[(pixels_y, pixels_x)][..., None]  # batch_size, 1
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        ).float()  # batch_size, 3
        p_d = p.clone().detach()
        # pixel -> camera -> normalization space (w/o pose). ps: 'pose' is a gap between camera and world
        # Camera
        if self.camera_static:
            img_idx = 0
        if self.camera_trainable:
            pose = self.pose_paras_to_mat(img_idx, cam_id)
        else:
            pose = self.pose_all[cam_id][img_idx]
        p = torch.matmul(
            self.intrinsics_all_inv[cam_id][img_idx, None, :3, :3], p[:, :, None]
        ).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(
            pose[None, :3, :3], rays_v[:, :, None]
        ).squeeze()  # batch_size, 3
        p_d = (
            depth
            * torch.matmul(
                self.intrinsics_all_inv[cam_id][img_idx, None, :3, :3], p_d[:, :, None]
            ).squeeze()
            * (1 / self.scales_all[cam_id][img_idx, :][0, 0])
        )  # batch_size, 3
        rays_l = torch.linalg.norm(p_d, ord=2, dim=-1, keepdim=True)  # batch_size, 1
        rays_s = torch.matmul(
            pose[None, :3, :3], p_d[:, :, None]
        ).squeeze()  # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat(
            [rays_o, rays_v, rays_s, rays_l, color, mask[:, :1]], dim=-1
        )  # batch_size, 14

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1, cam_id=-1):
        """
        Interpolate pose between two cameras.
        """
        # if cam_id == -1:
        #     cam_id = np.random.randint(self.n_views)
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        # if self.camera_static:
        #     idx_0 = 0
        #     idx_1 = 0
        if self.camera_trainable:
            pose_0 = self.pose_paras_to_mat(0, idx_0)
            pose_1 = self.pose_paras_to_mat(0, idx_1)
        else:
            pose_0 = self.pose_all[cam_id][idx_0]
            pose_1 = self.pose_all[cam_id][idx_1]
        p = torch.matmul(
            self.intrinsics_all_inv[cam_id][0, None, None, :3, :3], p[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = 0.9 * (pose_0[:3, 3] * (1.0 - ratio) + pose_1[:3, 3] * ratio)
        pose_0 = pose_0.detach().cpu().numpy()
        pose_1 = pose_1.detach().cpu().numpy()
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
        pose = torch.from_numpy(pose).cuda()
        rot = pose[:3, :3]
        trans = pose[:3, 3]
        rays_v = torch.matmul(
            rot[None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), pose

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level, cam_id=-1):
        if cam_id == -1:
            cam_id = np.random.randint(self.n_views)
        img = cv.imread(self.images_lis[cam_id][idx])
        return (
            cv.resize(img, (self.W // resolution_level, self.H // resolution_level))
        ).clip(0, 255)

    def depth_at(self, idx, resolution_level, cam_id=-1):
        dmax = 2
        # dmax = self.depth_max(cam_id)
        if cam_id == -1:
            cam_id = np.random.randint(self.n_views)
        depth_img = cv.resize(
            self.depths_np[cam_id][idx],
            (self.W // resolution_level, self.H // resolution_level),
        )
        depth_img = 255.0 - np.clip(depth_img / dmax, a_max=1, a_min=0) * 255.0
        return cv.applyColorMap(np.uint8(depth_img), cv.COLORMAP_JET)

    def depth_max(self, cam_id):
        return self.depths_np[cam_id].max()
