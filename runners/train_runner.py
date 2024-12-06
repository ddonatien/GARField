import os
import logging
from shutil import copyfile

import yaml
import trimesh
import cv2 as cv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as SSIM

from pyquaternion import Quaternion
from models.fields import (
    NeRF,
    CompositeSDFNetwork,
    SingleVarianceNetwork,
    RenderingNetwork,
)
from models.renderer import NeuSRenderer
from models.parameters import MobileStlContainer, ListStlContainer

from runners.base_runner import BaseRunner


class TrainRunner(BaseRunner):
    def __init__(
        self,
        conf_path,
        wdb_run,
        mode="train",
        case="CASE_NAME",
        is_continue=False,
        model_name="",
    ):
        super().__init__(
            conf_path, wdb_run, mode=mode, case=case, is_continue=is_continue
        )
        logging.getLogger("trimesh").setLevel(logging.ERROR)
        self.model_list = []

        # Networks
        sdf_dict = {"Rigid": {**self.conf["model.rigid_sdf_network"]}}
        sdf_dict.update(
            {
                "Robot": {**self.conf["model.robot_sdf_network"]},  # Robot
                "Object": {**self.conf["model.object_sdf_network"]},
            }
        )  # Object

        self.nerf_outside = NeRF(**self.conf["model.nerf"]).to(self.device)
        self.sdf_network = CompositeSDFNetwork(sdf_dict).to(self.device)
        self.deviation_network = SingleVarianceNetwork(
            **self.conf["model.variance_network"]
        ).to(self.device)
        self.color_network = RenderingNetwork(
            **self.conf["model.rendering_network"]
        ).to(self.device)
        self.object_mesh = MobileStlContainer(
            self.dataset.omesh_path,
            self.dataset.scales_all[0][0].float().to(self.device),
            torch.tensor(self.dataset.scene_params["ee_pose"]).float().to(self.device),
            **self.conf["model.object_params"],
        ).to(self.device)
        self.robot_mesh = ListStlContainer(
            self.dataset.rmeshes_path, **self.conf["model.robot_params"]
        ).to(self.device)
        self.object_mesh.eval()
        self.robot_mesh.eval()
        self.color_network.eval()
        self.sdf_network.eval()

        param_lr = 9e-8
        self.show_param_count()

        params_to_train = []
        params_to_train += [
            {
                "name": "nerf_outside",
                "params": self.nerf_outside.parameters(),
                "lr": self.learning_rate,
            }
        ]
        params_to_train += [
            {
                "name": "sdf_network",
                "params": self.sdf_network.parameters(),
                "lr": self.learning_rate,
            }
        ]
        params_to_train += [
            {
                "name": "deviation_network",
                "params": self.deviation_network.parameters(),
                "lr": self.learning_rate,
            }
        ]
        params_to_train += [
            {
                "name": "color_network",
                "params": self.color_network.parameters(),
                "lr": self.learning_rate,
            }
        ]
        params_to_train += [
            {
                "name": "object_mesh",
                "params": self.object_mesh.parameters(),
                "lr": param_lr,
            }
        ]

        if self.dataset.camera_trainable:
            params_to_train += [
                {
                    "name": "poses_paras",
                    "params": self.dataset.poses_paras,
                    "lr": param_lr,
                }
            ]
            for p in self.dataset.poses_paras:
                if p.requires_grad:
                    self.n_params += p.numel()

        self.render_opt = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(
            self.nerf_outside,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            **self.conf["model.neus_renderer"],
        )
        self.renderer.training = False

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if model_name == "":
                model_list_raw = os.listdir(
                    os.path.join(self.base_exp_dir, "checkpoints")
                )
                model_list = []
                for name in model_list_raw:
                    if (
                        name[-3:] == "pth"
                        and int(name[10:-4]) <= self.end_iter
                        and "geom" in name
                    ):
                        model_list.append(name)
                model_list.sort()
                latest_model_name = model_list[-1]
            else:
                latest_model_name = model_name

        if latest_model_name is not None:
            logging.info("Find checkpoint: %s", latest_model_name)
            self.continue_name = latest_model_name[5:-4]
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == "train":
            with torch.no_grad():
                self.file_backup()

    def show_param_count(self):
        names = []
        params = []
        self.n_params = 0
        names.append("NeRF outside")
        params.append(0)
        for p in self.nerf_outside.parameters():
            if p.requires_grad:
                lnb = p.numel()
                params[-1] += lnb
                self.n_params += lnb
        names.append("SDF net")
        params.append(0)
        for p in self.sdf_network.parameters():
            if p.requires_grad:
                lnb = p.numel()
                params[-1] += lnb
                self.n_params += lnb
        names.append("S inv")
        params.append(0)
        for p in self.deviation_network.parameters():
            if p.requires_grad:
                lnb = p.numel()
                params[-1] += lnb
                self.n_params += lnb
        names.append("Color net")
        params.append(0)
        for p in self.color_network.parameters():
            if p.requires_grad:
                lnb = p.numel()
                params[-1] += lnb
                self.n_params += lnb
        names.append("Object params")
        params.append(0)
        for p in self.object_mesh.parameters():
            if p.requires_grad:
                lnb = p.numel()
                params[-1] += lnb
                self.n_params += lnb
        if self.dataset.camera_trainable:
            names.append("Camera params")
            params.append(0)
            for p in self.dataset.poses_paras:
                if p.requires_grad:
                    lnb = p.numel()
                    params[-1] += lnb
                    self.n_params += lnb

        logging.info("Parameters summary:")
        for i, n in enumerate(names):
            logging.info(
                "%s || %s learnable params - %5.2f. %% of total",
                n.ljust(16),
                str(params[i]).ljust(8),
                params[i] / self.n_params * 100,
            )

    def train(self):
        res_step = self.end_iter - self.iter_step
        num_epochs = res_step // self.dataset.n_images
        logging.info("num_epochs: %i", num_epochs)
        self.wdb_run.watch(self.sdf_network)

        # Dropout
        self.object_mesh.train()
        self.robot_mesh.train()
        self.color_network.train()
        self.sdf_network.train()
        self.renderer.training = True

        self.update_learning_rate()

        for epoch_i in tqdm(range(num_epochs)):
            wandb_dict = {"epoch": epoch_i, "n_params": self.n_params}

            # TRAIN GEOMETRY
            image_perm = self.get_image_perm()
            for iter_i in range(0, self.dataset.n_images, 1):

                image_idx = image_perm[iter_i]

                object_mesh = self.object_mesh(image_idx)
                robot_mesh = self.robot_mesh(image_idx)
                meshes = (robot_mesh, object_mesh)

                # Depth
                if self.use_depth:
                    data = self.dataset.gen_random_rays_at_depth(
                        image_idx, self.batch_size
                    )
                    rays_o, rays_d, rays_s, rays_l, true_rgb, mask = (
                        data[:, :3],
                        data[:, 3:6],
                        data[:, 6:9],
                        data[:, 9:10],
                        data[:, 10:13],
                        data[:, 13:14],
                    )
                else:
                    data = self.dataset.gen_random_rays_at(image_idx, self.batch_size)
                    rays_o, rays_d, true_rgb, mask = (
                        data[:, :3],
                        data[:, 3:6],
                        data[:, 6:9],
                        data[:, 9:10],
                    )

                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3])

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).float()
                else:
                    mask = torch.ones_like(mask)

                mask_sum = mask.sum() + 1e-5
                details = {}
                perturb_overwrite = -1
                if self.end_iter - self.iter_step < 1e4:
                    perturb_overwrite = 0

                render_out = self.renderer.render(
                    rays_o,
                    rays_d,
                    near,
                    far,
                    meshes,
                    details,
                    perturb_overwrite=perturb_overwrite,
                    background_rgb=background_rgb,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                )

                for c, _ in enumerate(self.dataset.poses_paras):
                    poses_paras = self.dataset.poses_paras[c][0, :]
                    for j in range(6):
                        wandb_dict.update({f"Cam {c} pose/{j}": poses_paras[j]})
                if self.object_mesh.rigid_adjust:
                    for j in range(6):
                        wandb_dict.update(
                            {
                                f"Object pose correction/{j}": self.object_mesh.adjust_transform[
                                    j
                                ]
                            }
                        )

                # Depth
                if self.use_depth:
                    sdf_loss, valid_depth_region = self.renderer.errorondepth(
                        rays_o,
                        rays_d,
                        rays_s,
                        mask,
                        meshes,
                    )

                color_fine = render_out["color_fine"]
                s_val = render_out["s_val"]
                cdf_fine = render_out["cdf_fine"]
                gradient_error = render_out["gradient_error"]
                weight_max = render_out["weight_max"]
                weight_sum = render_out["weight_sum"]
                depth_map = render_out["depth_map"]
                mesh_sdf_loss = render_out["mesh_sdf_error"]

                # Loss
                if color_fine is not None:
                    color_error = (color_fine - true_rgb) * mask
                    color_fine_loss = (
                        F.l1_loss(
                            color_error, torch.zeros_like(color_error), reduction="sum"
                        )
                        / mask_sum
                    )
                    psnr = 20.0 * torch.log10(
                        1.0
                        / (
                            ((color_fine - true_rgb) ** 2 * mask).sum()
                            / (mask_sum * 3.0)
                        ).sqrt()
                    )
                else:
                    color_fine_loss = 0.0
                    psnr = 0

                eikonal_loss = gradient_error
                mask_loss = F.binary_cross_entropy(
                    weight_sum.clip(1e-3, 1.0 - 1e-3), mask
                )

                # Depth
                if self.use_depth:
                    depth_minus = (depth_map - rays_l) * valid_depth_region
                    depth_loss = (
                        F.l1_loss(
                            depth_minus, torch.zeros_like(depth_minus), reduction="sum"
                        )
                        / mask_sum
                    )
                    geo_loss = 0.5 * (depth_loss + sdf_loss)

                if self.use_depth:
                    loss = color_fine_loss + geo_loss * self.geo_weight
                else:
                    loss = color_fine_loss

                if self.iter_step < self.warm_up_end:
                    loss += (
                        eikonal_loss * self.igr_weight
                        + mask_loss * self.mask_weight
                        + mesh_sdf_loss * 0.01
                    )
                else:
                    loss += (
                        eikonal_loss * self.igr_weight + mask_loss * self.mask_weight
                    )

                self.render_opt.zero_grad()
                loss.backward()
                self.render_opt.step()

                wandb_dict.update({"iter_step": self.iter_step})
                wandb_dict.update({"Loss/loss": loss})
                wandb_dict.update({"Loss/color_loss": color_fine_loss})
                wandb_dict.update({"Loss/eikonal_loss": eikonal_loss})
                wandb_dict.update({"Loss/mesh_sdf_loss": mesh_sdf_loss})

                # Depth
                if self.use_depth:
                    wandb_dict.update({"Loss/sdf_loss": sdf_loss})
                    wandb_dict.update({"Loss/depth_loss": depth_loss})
                    del sdf_loss
                    del depth_loss
                wandb_dict.update({"Statistics/s_val": s_val.mean()})
                wandb_dict.update(
                    {"Statistics/cdf": (cdf_fine[:, :1] * mask).sum() / mask_sum}
                )
                wandb_dict.update(
                    {"Statistics/weight_max": (weight_max * mask).sum() / mask_sum}
                )
                wandb_dict.update({"Statistics/psnr": psnr})
                wandb_dict.update(
                    {"learning rate": self.render_opt.param_groups[0]["lr"]}
                )
                for g in self.render_opt.param_groups:
                    if g["name"] in ["poses_paras"]:
                        wandb_dict.update({"learning rate cameras": g["lr"]})

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    val_ssim, val_psnr = self.validate_image()
                    wandb_dict.update(
                        {
                            "Statistics/Validation PSNR": val_psnr,
                            "Statistics/Validation SSIM": val_ssim,
                        }
                    )
                    if self.use_depth:
                        self.validate_image_with_depth(-1)
                        depth_error = self.validate_depth(idx=-1, cam_id=-1)
                        wandb_dict.update(
                            {"Statistics/Validation depth error": depth_error}
                        )

                self.wdb_run.log(wandb_dict)
                self.update_learning_rate()
                self.iter_step += 1

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            ratio = 1.0
        else:
            ratio = np.min([1.0, self.iter_step / self.anneal_end])
        return ratio

    def update_learning_rate(self, scale_factor=1):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (
                self.end_iter - self.warm_up_end
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                1 - alpha
            ) + alpha
        learning_factor *= scale_factor

        current_learning_rate = self.learning_rate * learning_factor
        for g in self.render_opt.param_groups:
            if g["name"] in ["poses_paras"]:
                g["lr"] = 0.5 * current_learning_rate
            elif g["name"] in ["object_mesh"]:
                g["lr"] = 0.01 * current_learning_rate
            else:
                g["lr"] = current_learning_rate

    def file_backup(self):
        dir_lis = self.conf["general.recording"]
        os.makedirs(os.path.join(self.base_exp_dir, "recording"), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, "recording", dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(os.path.join(self.base_run_dir, dir_name))
            for f_name in files:
                if f_name[-3:] == ".py":
                    copyfile(
                        os.path.join(self.base_run_dir, dir_name, f_name),
                        os.path.join(cur_dir, f_name),
                    )
        if self.dataset.camera_trainable:
            if self.is_multiview:
                for i in range(self.dataset.n_views):
                    pose = self.dataset.pose_paras_to_mat(0, i)
                    np.save(
                        os.path.join(
                            self.base_exp_dir, "recording", f"poses_mat_{i}.npy"
                        ),
                        pose.detach().cpu().numpy(),
                    )
            else:
                pose = self.dataset.pose_paras_to_mat(0)
                np.save(
                    os.path.join(self.base_exp_dir, "recording", "poses_mat.npy"),
                    pose.detach().cpu().numpy(),
                )

        copyfile(
            self.conf_path, os.path.join(self.base_exp_dir, "recording", "config.conf")
        )

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(
            os.path.join(self.base_exp_dir, "checkpoints", checkpoint_name),
            map_location=self.device,
        )

        self.nerf_outside.load_state_dict(checkpoint["nerf"])
        self.sdf_network.load_state_dict(checkpoint["sdf_network_fine"])
        self.deviation_network.load_state_dict(checkpoint["variance_network_fine"])
        self.color_network.load_state_dict(checkpoint["color_network_fine"])
        self.render_opt.load_state_dict(checkpoint["render_opt"])
        if "parameters" in checkpoint:
            self.object_mesh.load_state_dict(checkpoint["parameters"])
        self.iter_step = checkpoint["iter_step"]

        if len(checkpoint_name) > 20:
            sub_name = checkpoint_name[17:-4]
            cameras = np.load(
                os.path.join(
                    self.base_exp_dir, "checkpoints", f"cameras_{sub_name}.npz"
                )
            )
        else:
            cameras = np.load(
                os.path.join(
                    self.base_exp_dir, "checkpoints", f"cameras_{sub_name}.npz"
                )
            )
        mats = []
        for i, _ in enumerate(cameras.files):
            mats.append(torch.tensor(cameras[f"cam{i}"]))
        self.dataset.load_pose_from_mat(mats)
        if self.dataset.camera_trainable:
            for p in self.dataset.poses_paras:
                p.requires_grad_()
        logging.info("Loading checkpoint %s ended", checkpoint_name)

    def save_pose_yaml(self, final=False):
        cam_dict = {}
        for c, _ in enumerate(self.dataset.poses_paras):
            path = os.path.join(self.base_exp_dir, "checkpoints", f"cam{c}.yaml")
            tmat = self.dataset.pose_paras_to_mat(0, c).detach().cpu().numpy()
            s = self.dataset.scale_mats_np[0][0]
            cam_dict[f"cam{c}"] = tmat
            q = Quaternion(matrix=tmat[:3, :3], rtol=1e-5, atol=1e-5)
            t = s[:3, :3] @ tmat[:3, 3] + s[:3, 3]

            yaml_dict0 = {
                "cam_to_base": {
                    "translation": {
                        "x": float(t[0]),
                        "y": float(t[1]),
                        "z": float(t[2]),
                    },
                    "quaternion": {
                        "w": float(q[0]),
                        "x": float(q[1]),
                        "y": float(q[2]),
                        "z": float(q[3]),
                    },
                }
            }
            with open(path, "w") as outfile:
                yaml.dump(yaml_dict0, outfile, default_flow_style=False)
        cam_file = os.path.join(self.base_exp_dir, "checkpoints", "cameras.npz")
        if final:
            cam_file = os.path.join(
                self.base_exp_dir,
                "checkpoints",
                f"cameras_{self.wdb_run.name.replace(' ', '').replace('/', '')}.npz",
            )
        np.savez(cam_file, **cam_dict)

    def save_checkpoint(self, final=False):
        checkpoint = {
            "nerf": self.nerf_outside.state_dict(),
            "sdf_network_fine": self.sdf_network.state_dict(),
            "variance_network_fine": self.deviation_network.state_dict(),
            "color_network_fine": self.color_network.state_dict(),
            "render_opt": self.render_opt.state_dict(),
            "parameters": self.object_mesh.state_dict(),
            "iter_step": self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, "checkpoints"), exist_ok=True)
        ckpt_file = os.path.join(
            self.base_exp_dir, "checkpoints", f"ckpt_geom_{self.iter_step:0>6d}.pth"
        )
        if final:
            ckpt_file = os.path.join(
                self.base_exp_dir,
                "checkpoints",
                f"ckpt_geom_{self.iter_step:0>6d}_{self.wdb_run.name.replace(' ', '').replace('/', '')}.pth",
            )
        torch.save(checkpoint, ckpt_file)
        pp = torch.stack(self.dataset.poses_paras, dim=0)
        pose_file = os.path.join(self.base_exp_dir, "checkpoints", "pose_paras.npz")
        if final:
            pose_file = os.path.join(
                self.base_exp_dir,
                "checkpoints",
                f"pose_paras_{self.wdb_run.name.replace(' ', '').replace('/', '')}.npz",
            )
        np.savez(pose_file, pp.detach().cpu().numpy())
        self.save_pose_yaml()

    def validate_image(self, idx=-1, cam_id=-1, resolution_level=-1, out_dir=None):
        self.object_mesh.eval()
        self.robot_mesh.eval()
        self.color_network.eval()
        self.sdf_network.eval()
        self.renderer.training = False
        ssim = None
        psnr = None

        if out_dir is None:
            out_dir = self.base_exp_dir
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if cam_id < 0:
            cam_id = np.random.randint(self.dataset.n_views)

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        rays_o, rays_d, true_rgb, mask = self.dataset.gen_rays_at(
            idx, resolution_level=resolution_level, cam_id=cam_id
        )
        object_mesh = self.object_mesh(idx)
        robot_mesh = self.robot_mesh(idx)
        meshes = (robot_mesh, object_mesh)

        intr = self.dataset.intrinsics_all[cam_id][0]

        pose = self.dataset.pose_paras_to_mat(idx, cam_id).detach()
        proj_mat = intr @ torch.inverse(pose)
        mesh_pts = []
        for m in meshes:
            verts = m[0]
            pts = torch.ones((verts.shape[0], 4))
            pts[:, :3] = verts
            px = torch.matmul(proj_mat, pts.T).T
            px[:, :2] = px[:, :2] / px[:, 2][:, None]
            mesh_pts.append(px / resolution_level)

        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        true_rgb = torch.permute(true_rgb, (1, 0, 2)).reshape(-1, 3)
        mask = mask.reshape(-1, 3)

        out_rgb_fine = []
        out_masks_fine = []
        out_normal_fine = []
        out_depth_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                meshes=meshes,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible("color_fine"):
                out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())
            if feasible("masks"):
                out_masks_fine.append(render_out["masks"].detach().cpu().numpy())
            if feasible("gradients") and feasible("weights"):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = (
                    render_out["gradients"] * render_out["weights"][:, :n_samples, None]
                )
                if feasible("inside_sphere"):
                    normals = normals * render_out["inside_sphere"][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible("depth_map"):
                out_depth_fine.append(render_out["depth_map"].detach().cpu().numpy())
            del render_out

        img_fine = None

        if len(out_rgb_fine) > 0:
            clr = np.concatenate(out_rgb_fine, axis=0)
            true_rgb = true_rgb.detach().cpu().numpy().reshape((-1, 3))
            mask = mask.detach().cpu().numpy().reshape((-1, 3))
            mask_sum = mask.sum() + 1e-5
            psnr = 20.0 * np.log10(
                1.0 / np.sqrt(((clr - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0))
            )
            img_fine = (
                np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256
            ).clip(0, 255)
            img_gt = self.dataset.image_at(
                idx, resolution_level=resolution_level, cam_id=cam_id
            )
            ssim = SSIM(
                img_fine[..., 0],
                img_gt,
                data_range=img_fine.max() - img_fine.min(),
                channel_axis=2,
            )
            for i in range(img_fine.shape[-1]):
                for j, m in enumerate(mesh_pts):
                    color = (100, 100, 100)
                    if j == 0:
                        color = (0, 255, 0)
                    if j == 1:
                        color = (0, 0, 255)
                    for p in m[::5]:
                        cv.circle(img_gt, (int(p[0]), int(p[1])), 1, color, -1)

        masks_fine = None
        if len(out_masks_fine) > 0:
            u = out_masks_fine[0].shape[-1]
            masks_fine = np.zeros((H, W, 3, 1))
            masks_fine[:, :, :u, :] = (
                np.concatenate(out_masks_fine, axis=0).reshape([H, W, u, -1]) * 256
            ).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            if self.dataset.camera_static:
                pos_idx = 0
            else:
                pos_idx = idx
            pose = self.dataset.pose_paras_to_mat(pos_idx, cam_id)
            rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
            normal_img = (
                np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape(
                    [H, W, 3, -1]
                )
                * 128
                + 128
            ).clip(0, 255)

        depth_img = None
        if len(out_depth_fine) > 0:
            dmax = 2
            scale = self.dataset.scales_all[0][0, 0, 0].cpu().numpy()
            depth_img = np.concatenate(out_depth_fine, axis=0)
            depth_img = depth_img.reshape([H, W, 1, -1])
            depth_img = (
                255.0 - np.clip(scale * depth_img / dmax, a_max=1, a_min=0) * 255.0
            )
            depth_img = np.uint8(depth_img)

        os.makedirs(os.path.join(out_dir, "validations_fine"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "masks_fine"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "normals"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "depths"), exist_ok=True)

        if img_fine is None:
            return None, None
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        out_dir,
                        "validations_fine",
                        f"{self.iter_step:0>8d}_{i}_{idx}_{cam_id}.png",
                    ),
                    np.concatenate([img_fine[..., i], img_gt]),
                )
                cv.imwrite(
                    os.path.join(
                        out_dir,
                        "masks_fine",
                        f"{self.iter_step:0>8d}_{i}_{idx}_{cam_id}.png",
                    ),
                    np.concatenate(
                        [
                            masks_fine[..., i],
                            self.dataset.image_at(
                                idx, resolution_level=resolution_level, cam_id=cam_id
                            ),
                        ]
                    ),
                )
            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        out_dir,
                        "normals",
                        f"{self.iter_step:0>8d}_{i}_{idx}_{cam_id}.png",
                    ),
                    normal_img[..., i],
                )

            if len(out_depth_fine) > 0:
                if self.use_depth:
                    cv.imwrite(
                        os.path.join(
                            out_dir,
                            "depths",
                            f"{self.iter_step:0>8d}_{i}_{idx}_{cam_id}.png",
                        ),
                        np.concatenate(
                            [
                                cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET),
                                self.dataset.depth_at(
                                    idx,
                                    resolution_level=resolution_level,
                                    cam_id=cam_id,
                                ),
                            ]
                        ),
                    )
                else:
                    cv.imwrite(
                        os.path.join(
                            out_dir,
                            "depths",
                            f"{self.iter_step:0>8d}_{i}_{idx}_{cam_id}.png",
                        ),
                        cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET),
                    )
        self.object_mesh.train()
        self.robot_mesh.train()
        self.color_network.train()
        self.sdf_network.train()
        self.renderer.training = True
        return ssim, psnr

    def validate_depth(self, idx=-1, cam_id=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if cam_id < 0:
            cam_id = np.random.randint(self.dataset.n_views)
        pts = self.dataset.get_surf_pts(idx, cam_id)
        robot_mesh = self.robot_mesh(idx)
        object_mesh = self.object_mesh(idx)
        meshes = (robot_mesh, object_mesh)
        data_dict = {"meshes": meshes}
        sdf = self.sdf_network.sdf(pts, data_dict)
        geom_err = torch.mean(torch.abs(sdf))
        return geom_err

    def validate_image_with_depth(self, idx=-1, resolution_level=-1, mode="train"):
        self.object_mesh.eval()
        self.robot_mesh.eval()
        self.color_network.eval()
        self.sdf_network.eval()
        self.renderer.training = False
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        cam_id = np.random.randint(self.dataset.n_views)

        if mode == "train":
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, rays_s, mask = self.dataset.gen_rays_at_depth(
            idx, resolution_level=resolution_level, cam_id=cam_id
        )
        object_mesh = self.object_mesh(idx)
        robot_mesh = self.robot_mesh(idx)
        meshes = (robot_mesh, object_mesh)

        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)
        rays_s = rays_s.reshape(-1, 3).split(batch_size)
        mask = (mask > 0.5).float().detach().cpu().numpy()[..., None]

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch, rays_s_batch in zip(rays_o, rays_d, rays_s):
            color_batch, gradients_batch = self.renderer.renderondepth(
                rays_o_batch,
                rays_d_batch,
                rays_s_batch,
                meshes,
            )

            out_rgb_fine.append(color_batch.detach().cpu().numpy())
            out_normal_fine.append(gradients_batch.detach().cpu().numpy())
            del color_batch, gradients_batch

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (
                np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256
            ).clip(0, 255)
            img_fine = img_fine * mask

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            pose = self.dataset.pose_paras_to_mat(idx, cam_id)
            rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
            normal_img = (
                np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape(
                    [H, W, 3, -1]
                )
                * 128
                + 128
            ).clip(0, 255)
            normal_img = normal_img * mask

        os.makedirs(os.path.join(self.base_exp_dir, "rgbsondepth"), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, "normalsondepth"), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.base_exp_dir,
                        "rgbsondepth",
                        f"{self.iter_step:0>8d}_depth_{idx}_{cam_id}.png",
                    ),
                    np.concatenate(
                        [
                            img_fine[..., i],
                            self.dataset.image_at(
                                idx, resolution_level=resolution_level, cam_id=cam_id
                            ),
                        ]
                    ),
                )
            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.base_exp_dir,
                        "normalsondepth",
                        f"{self.iter_step:0>8d}_depth_{idx}_{cam_id}.png",
                    ),
                    normal_img[..., i],
                )
        self.object_mesh.train()
        self.robot_mesh.train()
        self.color_network.train()
        self.sdf_network.train()
        self.renderer.training = True

    def render_novel_image(
        self, idx_0, idx_1, ratio, timestep, resolution_level, omesh=None
    ):
        """
        Interpolate view between two cameras.
        """
        self.object_mesh.eval()
        self.robot_mesh.eval()
        self.color_network.eval()
        self.sdf_network.eval()
        self.renderer.training = False
        rays_o, rays_d, pose = self.dataset.gen_rays_between(
            idx_0, idx_1, ratio, resolution_level=resolution_level
        )
        robot_mesh = self.robot_mesh(idx_0)
        if omesh is not None:
            meshes = (robot_mesh, omesh)
        else:
            object_mesh = self.object_mesh(idx_0)
            meshes = (robot_mesh, object_mesh)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_masks_fine = []
        out_normal_fine = []
        out_depth_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                meshes=meshes,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())

            if feasible("masks"):
                out_masks_fine.append(render_out["masks"].detach().cpu().numpy())
            if feasible("gradients") and feasible("weights"):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = (
                    render_out["gradients"] * render_out["weights"][:, :n_samples, None]
                )
                if feasible("inside_sphere"):
                    normals = normals * render_out["inside_sphere"][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible("depth_map"):
                out_depth_fine.append(render_out["depth_map"].detach().cpu().numpy())
            del render_out

        masks_fine = None
        if len(out_masks_fine) > 0:
            u = out_masks_fine[0].shape[-1]
            masks_fine = np.zeros((H, W, 3))
            masks_fine[:, :, :u] = (
                np.concatenate(out_masks_fine, axis=0).reshape([H, W, u]) * 256
            ).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
            normal_img = (
                np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3])
                * 128
                + 128
            ).clip(0, 255)

        depth_img = None
        if len(out_depth_fine) > 0:
            depth_img = np.concatenate(out_depth_fine, axis=0)
            depth_img = depth_img.reshape([H, W])
            cmap = plt.cm.jet
            norm = plt.Normalize(vmin=depth_img.min(), vmax=depth_img.max())
            depth_img = cmap(norm(depth_img))

        img_fine = (
            (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256)
            .clip(0, 255)
            .astype(np.uint8)
        )
        self.object_mesh.train()
        self.robot_mesh.train()
        self.color_network.train()
        self.sdf_network.train()
        self.renderer.training = True
        return img_fine, depth_img, masks_fine, normal_img

    def render_image(self, prefix=""):
        self.object_mesh.eval()
        self.robot_mesh.eval()
        self.color_network.eval()
        self.sdf_network.eval()
        self.renderer.training = False
        mesh_id = 3
        omesh_file = os.path.join(
            self.dataset.data_dir, f"gen_mesh_scaled/{mesh_id:05}.stl"
        )
        tmesh = trimesh.load(omesh_file, process=True, maintain_order=True)
        omesh = (
            torch.tensor(tmesh.vertices, dtype=torch.float32),
            torch.tensor(tmesh.faces),
            self.object_mesh.get_pos_codes().to(self.device),
        )
        ratio = 0.2
        img, dpth, msk, nrml = self.render_novel_image(0, 1, ratio, 0, 1, omesh=omesh)
        os.makedirs(os.path.join(self.base_exp_dir, "generated_images"), exist_ok=True)
        img_path = os.path.join(
            self.base_exp_dir, f"generated_images/{prefix}{mesh_id:05}_{ratio}_0_1.png"
        )
        cv.imwrite(img_path, img)
        dpth_path = os.path.join(
            self.base_exp_dir,
            f"generated_images/{prefix}depth_{mesh_id:05}_{ratio}_0_1.png",
        )
        plt.imsave(dpth_path, dpth)
        msk_path = os.path.join(
            self.base_exp_dir,
            f"generated_images/{prefix}mask_{mesh_id:05}_{ratio}_0_1.png",
        )
        cv.imwrite(msk_path, msk)
        nrml_path = os.path.join(
            self.base_exp_dir,
            f"generated_images/{prefix}normal_{mesh_id:05}_{ratio}_0_1.png",
        )
        cv.imwrite(nrml_path, nrml)
        self.object_mesh.train()
        self.robot_mesh.train()
        self.color_network.train()
        self.sdf_network.train()
        self.renderer.training = True
        logging.info("Image written at %s", img_path)

    def render_scene_zslices(self, prefix=""):
        self.object_mesh.eval()
        self.robot_mesh.eval()
        self.color_network.eval()
        self.sdf_network.eval()
        self.renderer.training = False
        centers = torch.zeros(10, 3)
        centers[:, 1] = torch.linspace(-1, 1, 10)
        normal = "y"

        idx = np.random.randint(self.dataset.n_images)
        for i in range(10):
            self.render_sdf_slice(
                centers[i], normal, idx=idx, sub_name=f"{prefix}{i:02}"
            )
            logging.info("Rendered slice %i", i)
        self.object_mesh.train()
        self.robot_mesh.train()
        self.color_network.train()
        self.sdf_network.train()
        self.renderer.training = True

    def interpolate_view(self, img_idx_0, img_idx_1, n_frames=60):
        images = []
        base_name = self.wdb_run.name.replace(" ", "").replace("/", "")
        video_dir = os.path.join(self.base_exp_dir, "render", base_name)
        os.makedirs(video_dir, exist_ok=True)
        n_cams = self.dataset.n_views - 1

        mesh_id = np.random.randint(100)
        omesh_file = os.path.join(
            self.dataset.data_dir, f"gen_mesh_scaled/{mesh_id:05}.stl"
        )
        tmesh = trimesh.load(omesh_file, process=True, maintain_order=True)
        omesh = (
            torch.tensor(tmesh.vertices, dtype=torch.float32),
            torch.tensor(tmesh.faces),
            self.object_mesh.get_pos_codes(),
        )

        for i in range(n_frames):
            pgr = n_cams * (np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5)
            idx_0 = int(np.floor(pgr))
            idx_1 = int(np.ceil(pgr))
            ratio = pgr - idx_0

            img, dpth, msks, nrml = self.render_novel_image(
                idx_0, idx_1, ratio, 0, resolution_level=4, omesh=omesh
            )
            image = np.concatenate(
                (
                    np.concatenate((img, 255 * dpth[..., :3]), axis=0),
                    np.concatenate((msks, nrml), axis=0),
                ),
                axis=1,
            )
            images.append(image)
            output_path = os.path.join(
                video_dir, f"{self.iter_step:0>8d}_timestep_{i}.png"
            )
            cv.imwrite(output_path, image)

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(
                video_dir,
                f"{self.iter_step:0>8d}_{img_idx_0}_{img_idx_1}.mp4",
            ),
            fourcc,
            30,
            (w, h),
        )
        for image in images:
            writer.write(image.astype(np.uint8))

        writer.release()

    def get_stats(self):
        psnr = []
        ssim = []
        geom_err = []
        self.object_mesh.eval()
        self.robot_mesh.eval()
        self.color_network.eval()
        self.sdf_network.eval()
        self.renderer.training = False
        nsplits = 3000
        os.makedirs(os.path.join(self.base_exp_dir, "validation"), exist_ok=True)

        image_perm = self.get_image_perm()
        for i in range(self.dataset.n_images):
            image_idx = image_perm[i]
            cam_id = np.random.randint(self.dataset.n_views)

            object_mesh = self.object_mesh(image_idx)
            robot_mesh = self.robot_mesh(image_idx)
            meshes = (robot_mesh, object_mesh)
            rays_o, rays_d, true_rgb, mask = self.dataset.gen_rays_at(
                image_idx, cam_id=cam_id
            )

            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(H * W // nsplits)
            rays_d = rays_d.reshape(-1, 3).split(H * W // nsplits)
            true_rgb = torch.permute(true_rgb, (1, 0, 2)).reshape(-1, 3)
            mask = mask.reshape(-1, 3)

            _clr = []
            for u, rays_o_batch in enumerate(rays_o):
                rays_d_batch = rays_d[u]

                near, far = self.dataset.near_far_from_sphere(
                    rays_o_batch, rays_d_batch
                )
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

                render_out = self.renderer.render(
                    rays_o_batch,
                    rays_d_batch,
                    near,
                    far,
                    meshes=meshes,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    background_rgb=background_rgb,
                )

                color_fine = render_out["color_fine"]
                _clr.append(color_fine.detach())

            clr = torch.cat(_clr, dim=0)
            mask_sum = mask.sum() + 1e-5

            geom_err.append(
                self.validate_depth(image_idx, cam_id).detach().cpu().numpy()
            )

            _psnr = 20.0 * torch.log10(
                1.0 / (((clr - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt()
            )
            img = (
                (clr.reshape([H, W, 3]).cpu().numpy() * 256)
                .clip(0, 255)
                .astype(np.uint8)
            )
            gt_img = (
                (true_rgb.reshape([H, W, 3]).cpu().numpy() * 256)
                .clip(0, 255)
                .astype(np.uint8)
            )
            _ssim = SSIM(img, gt_img, data_range=img.max() - img.min(), channel_axis=2)
            self.wdb_run.log({"PSNR": _psnr, "SSIM": _ssim})
            psnr.append(_psnr)
            ssim.append(_ssim)
            base_name = self.wdb_run.name.replace(" ", "").replace("/", "")
            img_path = os.path.join(
                self.base_exp_dir,
                f"validation/{base_name}_reconstructed_{image_idx}_{cam_id}.png",
            )
            cv.imwrite(img_path, img)
            gt_img_path = os.path.join(
                self.base_exp_dir,
                f"validation/{base_name}_ground_truth_{image_idx}_{cam_id}.png",
            )
            cv.imwrite(gt_img_path, gt_img)
            if i % 10 == 0:
                _psnr = torch.stack(psnr, dim=0).detach().cpu().numpy()
                _ssim = np.array(ssim)
                _geom_err = np.array(geom_err)

        psnr = torch.stack(psnr, dim=0).detach().cpu().numpy()
        ssim = np.array(ssim)
        geom_err = np.array(geom_err)

        return psnr, ssim, geom_err

    def make_vid(self, img_idx_0, img_idx_1):
        images = []
        base_name = self.wdb_run.name.replace(" ", "").replace("/", "")
        video_dir = os.path.join(self.base_exp_dir, "render_vid", base_name)
        os.makedirs(video_dir, exist_ok=True)

        mesh_id = np.random.randint(100)
        for mesh_id in range(101):
            omesh_file = os.path.join(
                self.dataset.data_dir, f"gen_vid_scaled/{mesh_id:05}.stl"
            )
            tmesh = trimesh.load(omesh_file, process=True, maintain_order=True)
            omesh = (
                torch.tensor(tmesh.vertices, dtype=torch.float32),
                torch.tensor(tmesh.faces),
                self.object_mesh.get_pos_codes(),
            )

            idx_0 = 2
            idx_1 = idx_0 + 1
            ratio = 0

            img, dpth, msks, nrml = self.render_novel_image(
                idx_0, idx_1, ratio, 0, resolution_level=4, omesh=omesh
            )
            image = np.concatenate(
                (
                    np.concatenate((img, 255 * dpth[..., :3]), axis=0),
                    np.concatenate((msks, nrml), axis=0),
                ),
                axis=1,
            )
            images.append(image)
            output_path = os.path.join(
                video_dir, f"{self.iter_step:0>8d}_timestep_{mesh_id}.png"
            )
            cv.imwrite(output_path, image)

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(
                video_dir, f"{self.iter_step:0>8d}_{img_idx_0}_{img_idx_1}.mp4"
            ),
            fourcc,
            30,
            (w, h),
        )
        for image in images:
            writer.write(image.astype(np.uint8))

        writer.release()

    def render_sdf_slice(
        self,
        center,
        norm_axis,
        idx=-1,
        resolution_level=-1,
        out_dir=None,
        sub_name=None,
    ):
        if "object" in self.scene_components:
            self.object_mesh.eval()
        if "robot" in self.scene_components:
            self.robot_mesh.eval()
        self.color_network.eval()
        self.sdf_network.eval()
        self.renderer.training = False

        if sub_name is None:
            sub_name = str(center.tolist())
        if out_dir is None:
            out_dir = self.base_exp_dir
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        l = resolution_level
        tx = torch.linspace(0, self.dataset.H - 1, self.dataset.H // l)
        ty = torch.linspace(0, self.dataset.W - 1, self.dataset.W // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")

        pts = torch.zeros(list(pixels_x.shape) + [3])
        if norm_axis == "x":
            pts[..., 0] = center[0]
            pts[..., 1] = pixels_x / self.dataset.H * 2 - 1
            pts[..., 2] = pixels_y / self.dataset.W * 2 - 1
        elif norm_axis == "y":
            pts[..., 0] = pixels_x / self.dataset.H * 2 - 1
            pts[..., 1] = center[1]
            pts[..., 2] = pixels_y / self.dataset.W * 2 - 1
        elif norm_axis == "z":
            pts[..., 0] = pixels_x / self.dataset.H * 2 - 1
            pts[..., 1] = pixels_y / self.dataset.W * 2 - 1
            pts[..., 2] = center[2]

        meshes = []
        if "robot" in self.scene_components:
            robot_mesh = self.robot_mesh(idx)
            meshes.append(robot_mesh)
        if "object" in self.scene_components:
            sim_mesh = self.object_mesh(idx)
            meshes.append(sim_mesh)

        with torch.no_grad():
            data_dict = {"meshes": meshes}
            details = {"partial_sdf": torch.empty(0)}

            out = (
                self.renderer.sdf_network.sdf(
                    pts.reshape(-1, 3), data_dict, details=details
                )
                .reshape(pixels_x.shape)
                .detach()
                .cpu()
                .numpy()
            )

        if sim_mesh is not None:
            if norm_axis == "x":
                nodes_px = sim_mesh[0][:, 1:].detach().cpu().numpy()
            elif norm_axis == "y":
                nodes_px = sim_mesh[0][:, ::2].detach().cpu().numpy()
            else:
                nodes_px = sim_mesh[0][:, :-1].detach().cpu().numpy()

            nodes_px[:, 0] = (nodes_px[:, 0] + 1) / 2 * self.dataset.H // l
            nodes_px[:, 1] = (nodes_px[:, 1] + 1) / 2 * self.dataset.W // l
        if len(details["partial_sdf"].shape) == 3:
            for i in range(details["partial_sdf"].shape[0]):
                data_raw = (
                    details["partial_sdf"][i]
                    .reshape(pixels_x.shape)
                    .detach()
                    .cpu()
                    .numpy()
                )
                os.makedirs(os.path.join(out_dir, "slices", f"sdf{i}"), exist_ok=True)
                os.makedirs(
                    os.path.join(out_dir, "log_slices", f"sdf{i}"), exist_ok=True
                )
                slice_log_img = self.log_slice_img(data_raw)
                slice_img = self.slice_img(data_raw)
                cv.imwrite(
                    os.path.join(
                        out_dir,
                        "slices",
                        f"sdf{i}",
                        f"{self.iter_step:0>8d}_{idx}_{sub_name}_{norm_axis}.png",
                    ),
                    slice_img,
                )
                cv.imwrite(
                    os.path.join(
                        out_dir,
                        "log_slices",
                        f"sdf{i}",
                        f"{self.iter_step:0>8d}_{idx}_{sub_name}_{norm_axis}.png",
                    ),
                    slice_log_img,
                )

        slice_log_img = self.log_slice_img(out)
        slice_img = self.slice_img(out)

        if sim_mesh is not None:
            for p in nodes_px:
                cv.circle(slice_img, (int(p[1]), int(p[0])), 1, (0, 0, 0), -1)
                cv.circle(slice_log_img, (int(p[1]), int(p[0])), 1, (0, 0, 0), -1)

        os.makedirs(os.path.join(out_dir, "slices"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "log_slices"), exist_ok=True)
        cv.imwrite(
            os.path.join(
                out_dir,
                "slices",
                f"{self.iter_step:0>8d}_{idx}_{sub_name}_{norm_axis}.png",
            ),
            slice_img,
        )
        cv.imwrite(
            os.path.join(
                out_dir,
                "log_slices",
                f"{self.iter_step:0>8d}_{idx}_{sub_name}_{norm_axis}.png",
            ),
            slice_log_img,
        )

        if "object" in self.scene_components:
            self.object_mesh.train()
        if "robot" in self.scene_components:
            self.robot_mesh.train()
        self.color_network.train()
        self.sdf_network.train()
        self.renderer.training = True
