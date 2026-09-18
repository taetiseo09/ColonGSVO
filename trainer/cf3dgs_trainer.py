# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from tqdm import tqdm
from random import randint
import math
import numpy as np
import random
from collections import defaultdict, OrderedDict
import json
import gzip
import torch
import torch.nn.functional as F
from torchvision import io
from PIL import Image
from einops import rearrange
import pickle
import imageio
import glob
import cv2
import open3d as o3d

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene.gaussian_model_cf import CF3DGS_Render as GS_Render

from utils.graphics_utils import BasicPointCloud, focal2fov, procrustes
from scene.cameras import Camera
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr, colorize, align_brightness_with_flow
from utils.mask_utils import compute_epipolar_loss, adaptive_thresholding, generate_intensity_diff_mask, save_masks_as_png
from kornia.geometry.depth import depth_to_3d, depth_to_3d_v2, depth_to_normals
from kornia.geometry.camera import project_points
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
import time
from .trainer import GaussianTrainer
from .losses import Loss, compute_scale_and_shift
from .timing_utils import TimingStats

from copy import copy

# def attach_debugger():
#     import debugpy
#     debugpy.listen(5678)
#     print("Waiting for Debuger to Attach on Port 5678...")
#     debugpy.wait_for_client()
#     print("Attached!")
# attach_debugger()

def contruct_pose(poses):
    n_trgt = poses.shape[0]
    for i in range(n_trgt-1, 0, -1):
        poses = torch.cat(
            (poses[:i], poses[[i-1]]@poses[i:]), 0)
    return poses


class CFGaussianTrainer(GaussianTrainer):
    def __init__(self, data_root, model_cfg, pipe_cfg, optim_cfg):
        super().__init__(data_root, model_cfg, pipe_cfg, optim_cfg)
        self.model_cfg = model_cfg
        self.pipe_cfg = pipe_cfg
        self.optim_cfg = optim_cfg

        self.gs_render = GS_Render(white_background=False,
                                   view_dependent=model_cfg.view_dependent,)
        self.gs_render_local = GS_Render(white_background=False,
                                         view_dependent=model_cfg.view_dependent,)
        self.use_mask = self.pipe_cfg.use_mask
        self.use_mono = self.pipe_cfg.use_mono
        self.near = 0.01
        self.flow_model = None

        # 初始化时间统计器（与 EndoGSLAM 兼容）
        self.timing_stats = None  # 在 train_from_progressive 中初始化
        self.setup_losses()

        if getattr(model_cfg, 'use_flow_loss', False):
            try:

                # 选择光流模型
                flow_model_name = getattr(model_cfg, 'flow_model_name', 'unimatch')
                flow_ckpt_name = getattr(model_cfg, 'flow_ckpt_name', 'mix')

                print(f"加载光流模型: {flow_model_name}，使用权重: {flow_ckpt_name}")
                self.flow_model = ptlflow.get_model(flow_model_name, pretrained_ckpt=flow_ckpt_name)
                self.flow_model.to(model_cfg.data_device)
                self.flow_model.eval()
                print("光流模型加载成功")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load PTLFlow model '{flow_model_name}' "
                    f"with checkpoint '{flow_ckpt_name}'. Disable --use_flow_loss "
                    "or verify network/cache access and the ptlflow installation."
                ) from e
    def setup_losses(self):
        self.loss_func = Loss(self.optim_cfg)

    def compute_flow_between_views(self, image1, image2):
        """使用PTLFlow计算两个图像之间的光流"""
        if self.flow_model is None:
            return torch.zeros(
                (1, 2, image1.shape[-2], image1.shape[-1]),
                device=image1.device,
                dtype=image1.dtype,
            )
        device = image1.device

        with torch.no_grad():
            # 确保输入格式正确
            if image1.dim() == 3:
                image1 = image1.unsqueeze(0)
            if image2.dim() == 3:
                image2 = image2.unsqueeze(0)

            if image1.max() <= 1.0:
                image1_input = image1 * 255.0
                image2_input = image2 * 255.0
            else:
                image1_input = image1.clone()
                image2_input = image2.clone()

            # 创建输入适配器
            io_adapter = IOAdapter(self.flow_model, image1.shape[2:])
            img1_np = image1_input.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            img2_np = image2_input.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

            # 准备输入
            inputs = io_adapter.prepare_inputs([img1_np[0], img2_np[0]])

            # 将准备好的输入转移到GPU
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(device)
            # 计算光流
            self.flow_model = self.flow_model.to(device)
            predictions = self.flow_model(inputs)
            flow = predictions['flows'][0]  # [1,2,H,W]

            return flow.to(image1.device)

    def get_pointcloud(self, depth, intrinsics, w2c, sampled_indices, return_sample=False):
        CX = intrinsics[0][2]
        CY = intrinsics[1][2]
        FX = intrinsics[0][0]
        FY = intrinsics[1][1]

        # Compute indices of sampled pixels
        xx = (sampled_indices[:, 1] - CX)/FX
        yy = (sampled_indices[:, 0] - CY)/FY
        depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]

        # Remove points at camera origin
        A = torch.abs(torch.round(pts, decimals=4)).cuda().float()
        B = torch.zeros((1, 3)).cuda().float()
        _, idx, counts = torch.cat([A, B], dim=0).unique(
            dim=0, return_inverse=True, return_counts=True)
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        invalid_pt_idx = mask[:len(A)]
        valid_pt_idx = ~invalid_pt_idx
        valid_pt_idx = valid_pt_idx.to(pts.device)
        pts = pts[valid_pt_idx]
        if return_sample:
            sampled_indices_ = sampled_indices[valid_pt_idx]
            return pts, sampled_indices_
        else:
            return pts

    def projection_flow_loss(self, gt_flow, render_depth_1, w2c_1, w2c_2, intrinsic, rigid_mask=None):
        gt_flow = gt_flow[0]
        render_depth_1 = render_depth_1
        width, height = render_depth_1.shape[2], render_depth_1.shape[1]

        # Create a depth mask
        depth_mask = render_depth_1[0]
        if rigid_mask is not None:
            depth_mask = depth_mask * rigid_mask

        # Find valid depth indices
        valid_depth_indices = torch.where(depth_mask > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)

        # Back Project the selected pixels to 3D Pointcloud
        w2c_1 = w2c_1
        pts, valid_depth_indices = self.get_pointcloud(render_depth_1, intrinsic, w2c_1, valid_depth_indices, True)

        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (w2c_2 @ pts4.T).T[:, :3]

        # Project the 3D pointcloud to the keyframe's image space
        intrinsic = intrinsic
        points_2d = torch.matmul(intrinsic, transformed_pts.transpose(0, 1)).transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5  # Small epsilon to prevent division by zero
        points_2d = points_2d / points_z

        projected_pts = points_2d[:, :2]
        edge = 20
        mask = (projected_pts[:, 0] < width - edge) & (projected_pts[:, 0] > edge) & \
            (projected_pts[:, 1] < height - edge) & (projected_pts[:, 1] > edge) & \
            (points_z[:, 0] > 0)

        projected_pts = projected_pts[mask]
        valid_depth_indices = valid_depth_indices[mask]

        if projected_pts.numel() == 0 or valid_depth_indices.numel() == 0:
            # Handle the case where no valid points remain
            return torch.tensor(0.0, device=gt_flow.device)  # Or another appropriate value

        valid_depth_indices = valid_depth_indices[:, [1, 0]]  # Ensure correct indexing

        # Calculate projection flow and ground truth backward flow
        projection_flow = projected_pts - valid_depth_indices.float()  # Ensure float type
        gt_bw_flow = gt_flow[:, valid_depth_indices[:, 1], valid_depth_indices[:, 0]].permute(1, 0)

        # Ensure no NaNs before loss calculation
        if torch.isnan(projection_flow).any() or torch.isnan(gt_bw_flow).any():
            return torch.tensor(0.0, device=gt_flow.device)  # Or handle appropriately

        loss = l1_loss(projection_flow, gt_bw_flow)

        return loss

    def train_step(self,
                   gs_render,
                   viewpoint_cam,
                   iteration,
                   pipe,
                   optim_opt,
                   colors_precomp=None,
                   update_gaussians=True,
                   update_cam=True,
                   update_distort=False,
                   densify=True,
                   prev_gaussians=None,
                   use_reproject=False,
                   use_matcher=False,
                   flow_weight=0.1,
                   ref_fidx=None,
                   reset=True,
                   reproj_loss=None,
                   **kwargs,
                   ):
        # Render - 添加渲染计时
        if self.timing_stats is not None:
            self.timing_stats.start_rendering()

        render_pkg = gs_render.render(
            viewpoint_cam,
            compute_cov3D_python=pipe.compute_cov3D_python,
            convert_SHs_python=pipe.convert_SHs_python,
            override_color=colors_precomp)

        if self.timing_stats is not None:
            self.timing_stats.stop_rendering()

        if prev_gaussians is not None:
            with torch.no_grad():
                # Render
                render_pkg_prev = prev_gaussians.render(
                    viewpoint_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python,
                    override_color=colors_precomp)
            mask = (render_pkg["alpha"] > 0.5).float()
            render_pkg["image"] = render_pkg["image"] * \
                mask + render_pkg_prev["image"] * (1 - mask)
            render_pkg["depth"] = render_pkg["depth"] * \
                mask + render_pkg_prev["depth"] * (1 - mask)

        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["image"],
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss_dict = self.compute_loss(render_pkg, viewpoint_cam,
                                      pipe, iteration,
                                      use_reproject, use_matcher,
                                      ref_fidx, **kwargs)

        loss = loss_dict['loss']

        loss.backward()

        with torch.no_grad():
            # Progress bar
            # try:
            #     self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            # except:
            #     pdb.set_trace()
            # mask = visibility_filter.reshape(gt_image.shape[1:])[None]
            psnr_train = psnr(image, gt_image).mean().double()
            self.just_reset = False
            if iteration < optim_opt.densify_until_iter and densify:
                # Keep track of max radii in image-space for pruning
                try:
                    gs_render.gaussians.max_radii2D[visibility_filter] = torch.max(gs_render.gaussians.max_radii2D[visibility_filter],
                                                                                   radii[visibility_filter])
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Failed to update max_radii2D; visibility and radii shapes "
                        f"are {tuple(visibility_filter.shape)} and {tuple(radii.shape)}."
                    ) from exc
                gs_render.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > optim_opt.densify_from_iter and iteration % optim_opt.densification_interval == 0:
                    size_threshold = 20 if iteration > optim_opt.opacity_reset_interval else None
                    self.gs_render.gaussians.densify_and_prune(optim_opt.densify_grad_threshold, 0.005,
                                                               gs_render.radius, size_threshold)

                if iteration % optim_opt.opacity_reset_interval == 0 and reset and iteration < optim_opt.reset_until_iter:
                    gs_render.gaussians.reset_opacity()
                    self.just_reset = True

            if update_gaussians:
                gs_render.gaussians.optimizer.step()
                gs_render.gaussians.optimizer.zero_grad(set_to_none=True)
            if getattr(gs_render.gaussians, "camera_optimizer", None) is not None and update_cam:
                current_fidx = gs_render.gaussians.seq_idx
                gs_render.gaussians.camera_optimizer[current_fidx].step()
                gs_render.gaussians.camera_optimizer[current_fidx].zero_grad(
                    set_to_none=True)


        return loss_dict, render_pkg, psnr_train

    def train_step_flow(self,
                   gs_render,
                   viewpoint_cam,
                   viewpoint_cam_prev,
                   render_dict_prev,
                   pts,
                   iteration,
                   pipe,
                   optim_opt,
                   colors_precomp=None,
                   update_gaussians=True,
                   update_cam=True,
                   update_distort=False,
                   densify=True,
                   prev_gaussians=None,
                   use_reproject=False,
                   use_matcher=False,
                   flow_weight=0.1,
                   ref_fidx=None,
                   reset=True,
                   reproj_loss=None,
                   precomputed_flow=None,
                   **kwargs,
                   ):

        if not hasattr(self, '_flow_cache'):
            self._flow_cache = None
        if not hasattr(self, '_flow_loss_cache'):
            self._flow_loss_cache = None
        # Render - 添加渲染计时
        if self.timing_stats is not None:
            self.timing_stats.start_rendering()

        render_pkg = gs_render.render(
            viewpoint_cam,
            compute_cov3D_python=pipe.compute_cov3D_python,
            convert_SHs_python=pipe.convert_SHs_python,
            override_color=colors_precomp)

        if self.timing_stats is not None:
            self.timing_stats.stop_rendering()

        if prev_gaussians is not None:
            with torch.no_grad():
                # Render
                render_pkg_prev = prev_gaussians.render(
                    viewpoint_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python,
                    override_color=colors_precomp)
            mask = (render_pkg["alpha"] > 0.5).float()
            render_pkg["image"] = render_pkg["image"] * \
                mask + render_pkg_prev["image"] * (1 - mask)
            render_pkg["depth"] = render_pkg["depth"] * \
                mask + render_pkg_prev["depth"] * (1 - mask)

        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["image"],
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])
        # Loss
        w2c_current = gs_render.gaussians.get_RT()

        # 优化：直接在GPU上构建，减少CPU-GPU传输
        if not hasattr(self, '_w2c_prev_cache'):
            self._w2c_prev_cache = torch.eye(4, device='cuda')
        w2c_prev = self._w2c_prev_cache
        w2c_prev[:3, :3] = torch.from_numpy(viewpoint_cam_prev.R).float().cuda()
        w2c_prev[:3, 3] = torch.from_numpy(viewpoint_cam_prev.T).float().cuda()
        intrinsics = viewpoint_cam_prev.intrinsics

        gt_image = viewpoint_cam.original_image.cuda()
        gt_image_prev = viewpoint_cam_prev.original_image.cuda()

        h, w = gt_image.shape[1], gt_image.shape[2]
        _, sampson_dist = compute_epipolar_loss(precomputed_flow[0], w2c_prev, w2c_current, intrinsics, h, w)

        rigid_mask = sampson_dist < adaptive_thresholding(sampson_dist)

        # 优化：缓存亮度对齐结果，减少重复计算
        if not hasattr(self, '_brightness_aligned_cache'):
            self._brightness_aligned_cache = None
            self._brightness_cache_iter = -1

        # 优化：只在必要时进行亮度对齐，减少计算开销
        if self.model_cfg.align_brightness:
            if iteration % 10 == 1 or self._brightness_aligned_cache is None:
                # Align brightness between two images (每10次迭代计算一次)
                gt_image_aligned = align_brightness_with_flow(
                    gt_image_prev, gt_image, precomputed_flow,
                    flow_threshold=self.optim_cfg.brightness_flow_threshold,
                    edge_smoothness=self.optim_cfg.brightness_edge_smoothness,
                    dilate_size=self.optim_cfg.brightness_dilate_size,
                    dilate_iter=self.optim_cfg.brightness_dilate_iter,
                    edge_window=self.optim_cfg.brightness_edge_window,
                    inverse_mask=True
                )
                # 优化：直接转为tensor，避免中间步骤
                if isinstance(gt_image_aligned, np.ndarray):
                    gt_image = torch.from_numpy(gt_image_aligned).cuda()
                else:
                    gt_image = gt_image_aligned
                if gt_image.shape[0] != 3:
                    gt_image = gt_image.permute(2, 0, 1)
                self._brightness_aligned_cache = gt_image
                self._brightness_cache_iter = iteration
            else:
                # 使用缓存的结果
                gt_image = self._brightness_aligned_cache

        mask = generate_intensity_diff_mask(
            gt_image_prev, gt_image, precomputed_flow,
            intensity_threshold=self.optim_cfg.intensity_threshold,
            kernel_size=self.optim_cfg.intensity_kernel_size,
            dilate_iter=self.optim_cfg.intensity_dilate_iter,
            visualize=False
        )
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(rigid_mask.device)
        elif mask.dtype == torch.bool:
            mask = mask.int()
        if rigid_mask.dtype == torch.bool:
            rigid_mask = rigid_mask.int()

        mask_bool = mask > 0
        rigid_mask_bool = rigid_mask > 0
        mask_bool = ~ mask_bool
        combined_mask = mask_bool & rigid_mask_bool

        view_idx = viewpoint_cam.uid if hasattr(viewpoint_cam, 'uid') else 0

        if hasattr(self.model_cfg, 'visualize_mask') and self.model_cfg.visualize_mask:
            if iteration % 300 == 1:
                mask_save_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "mask_visualizations",
                    f"intensity_{self.optim_cfg.intensity_threshold}",
                    f"iter{iteration:06d}_view{view_idx:03d}"
                )
                save_masks_as_png(rigid_mask, mask, iteration, mask_save_dir, view_idx)
        # if hasattr(self.model_cfg, 'visualize_mask') and self.model_cfg.visualize_mask:
        #     # 使用可配置的mask保存目录
        #     mask_dir_name = getattr(self.model_cfg, 'mask_save_dir', f'mask_visualizations_{self.optim_cfg.intensity_threshold}')
        #     # 优化：减少保存频率，降低IO开销
        #     if iteration % 300 == 1:  # 每30次迭代保存一次
        #         mask_save_dir = os.path.join(
        #             os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        #             mask_dir_name,
        #             f"iter{iteration:06d}_view{view_idx:03d}"
        #         )
        #         save_masks_as_png(rigid_mask, mask, iteration, mask_save_dir, view_idx)
            # colored_mask = torch.zeros((3, mask.shape[0], mask.shape[1]), device=mask.device)
            # colored_mask[0] = mask  # 将强度差异掩码映射到红色通道
            # colored_mask[1] = rigid_mask  # 将刚性掩码映射到绿色通道
            # colored_mask[2] = mask & rigid_mask  # 重叠区域映射到蓝色通道

            # # 保存可视化结果
            # colored_mask_np = colored_mask.cpu().numpy()
            # cv2.imwrite("combined_masks.png", colored_mask_np * 255)

        loss_dict = self.compute_loss(render_pkg, viewpoint_cam,
                                      pipe, iteration,
                                      use_reproject, use_matcher,
                                      ref_fidx, **kwargs)

        loss = loss_dict['loss']

        if hasattr(self, 'flow_model') and self.flow_model is not None:

            # 提取深度图
            depth_current = render_pkg["depth"]
            depth_prev = render_dict_prev["depth"]

            # 获取内参矩阵
            intrinsics = torch.from_numpy(viewpoint_cam.intrinsics).float().cuda()

            # 优化：降低光流计算频率，减少开销
            if iteration % 50 == 1:
                rendered_image_current = image
                rendered_image_prev = render_dict_prev["image"]
                with torch.no_grad():
                    flow_prev_to_current = self.compute_flow_between_views(rendered_image_prev, rendered_image_current)
                    flow_loss = self.projection_flow_loss(flow_prev_to_current,
                                                        depth_prev,
                                                        w2c_prev,
                                                        w2c_current,
                                                        intrinsics,
                                                        rigid_mask=combined_mask)
                    self._flow_cache = flow_prev_to_current
                    self._flow_loss_cache = flow_loss
            else:
                if self._flow_cache is not None:
                    flow_prev_to_current = self._flow_cache
                if self._flow_loss_cache is not None:
                    flow_loss = self._flow_loss_cache

            if flow_loss > 0:
                # λ2: 光流 loss 权重
                lambda_flow = self.optim_cfg.flow_loss_weight
                if lambda_flow is None:
                    lambda_flow = 0.1
                weighted_flow_loss = lambda_flow * flow_loss
                # weighted_flow_loss.backward(retain_graph=True)
                loss += weighted_flow_loss
                loss_dict['flow_loss'] = weighted_flow_loss.item()
                loss_dict['flow_loss_raw'] = flow_loss.item()

        loss.backward()

        with torch.no_grad():
            # Progress bar
            # try:
            #     self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            # except:
            #     pdb.set_trace()
            # mask = visibility_filter.reshape(gt_image.shape[1:])[None]
            psnr_train = psnr(image, gt_image).mean().double()
            self.just_reset = False
            if iteration < optim_opt.densify_until_iter and densify:
                # Keep track of max radii in image-space for pruning
                try:
                    gs_render.gaussians.max_radii2D[visibility_filter] = torch.max(gs_render.gaussians.max_radii2D[visibility_filter],
                                                                                   radii[visibility_filter])
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Failed to update max_radii2D; visibility and radii shapes "
                        f"are {tuple(visibility_filter.shape)} and {tuple(radii.shape)}."
                    ) from exc
                gs_render.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > optim_opt.densify_from_iter and iteration % optim_opt.densification_interval == 0:
                    size_threshold = 20 if iteration > optim_opt.opacity_reset_interval else None
                    self.gs_render.gaussians.densify_and_prune(optim_opt.densify_grad_threshold, 0.005,
                                                               gs_render.radius, size_threshold)

                if iteration % optim_opt.opacity_reset_interval == 0 and reset and iteration < optim_opt.reset_until_iter:
                    gs_render.gaussians.reset_opacity()
                    self.just_reset = True

            if update_gaussians:
                gs_render.gaussians.optimizer.step()
                gs_render.gaussians.optimizer.zero_grad(set_to_none=True)
            if getattr(gs_render.gaussians, "camera_optimizer", None) is not None and update_cam:
                current_fidx = gs_render.gaussians.seq_idx
                gs_render.gaussians.camera_optimizer[current_fidx].step()
                gs_render.gaussians.camera_optimizer[current_fidx].zero_grad(
                    set_to_none=True)


        return loss_dict, render_pkg, psnr_train

    def init_two_view(self, view_idx_1, view_idx_2, pipe, optim_opt):
        # prepare data
        self.loss_func.depth_loss_type = "invariant"
        cam_info, pcd, viewpoint_cam, pts = self.prepare_data(view_idx_1,
                                                         orthogonal=True,
                                                         down_sample=True)
        radius = np.linalg.norm(pcd.points, axis=1).max()

        # Initialize gaussians
        self.gs_render.reset_model()
        self.gs_render.init_model(pcd,) #load pcd from depth_to_3d
        # self.gs_render.init_model(num_pts=300_000,)
        self.gs_render.gaussians.init_RT_seq(self.seq_len)
        self.gs_render.gaussians.set_seq_idx(view_idx_1)
        self.gs_render.gaussians.rotate_seq = False
        # Fit relative pose
        print(f"optimizing frame {view_idx_1:03d}")
        optim_opt.iterations = self.optim_cfg.init_two_view_iterations
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress init two view")
        self.gs_render.gaussians.training_setup(optim_opt, fix_pos=True,)
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render.gaussians.update_learning_rate(iteration)
            loss, rend_dict, psnr_train = self.train_step(self.gs_render,
                                                          viewpoint_cam, iteration,
                                                          pipe, optim_opt,
                                                          depth_gt=self.mono_depth[view_idx_1],
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          )
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        self.pcd_stack = []
        self.pcd_stack.append(self.gs_render.gaussians.get_xyz.detach())
        model_params = self.gs_render.gaussians.capture()
        return model_params

    def add_view_v2(self, view_idx, view_idx_prev, reverse=False):
        # Initialize gaussians
        self.loss_func.depth_loss_type = "invariant"
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        # prepare data
        cam_info, pcd, viewpoint_cam, pts = self.prepare_data(view_idx_prev,
                                                         orthogonal=True,
                                                         down_sample=True)
        radius = np.linalg.norm(pcd.points, axis=1).max()
        self.gs_render_local.reset_model()
        self.gs_render_local.init_model(pcd)
        # Fit current gaussian
        optim_opt.iterations = self.optim_cfg.add_view_gs_local_iterations
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress add view v2")
        self.gs_render_local.gaussians.training_setup(
            optim_opt, fix_pos=True,)
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render_local.gaussians.update_learning_rate(iteration)
            loss, rend_dict, psnr_train = self.train_step(self.gs_render_local,
                                                          viewpoint_cam, iteration,
                                                          pipe, optim_opt,
                                                          #   depth_gt=self.mono_depth[view_idx_prev],
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          updata_distort=False,
                                                          densify=False,
                                                          )
            if psnr_train > 35 and iteration > 500:
                progress_bar.close()
                break

            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        print(f"optimizing frame {view_idx:03d}")
        viewpoint_cam_ref = self.load_viewpoint_cam(view_idx,
                                                    load_depth=True)
        optim_opt.iterations = self.optim_cfg.add_view_tracking_iterations
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        self.gs_render_local.gaussians.init_RT(None)
        self.gs_render_local.gaussians.training_setup_fix_position(
            optim_opt, gaussian_rot=False)

        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress update camera")

        # 提前停止机制：Tracking阶段（可通过参数控制）
        psnr_history = []
        early_stop_window = 10  # 检查最近10次迭代
        early_stop_threshold = 0.1  # PSNR变化小于0.1认为收敛
        target_psnr = 30.0  # 目标PSNR
        enable_early_stop = optim_opt.enable_early_stop  # 从参数读取是否启用

        precomputed_flow = None
        if hasattr(self, 'flow_model') and self.flow_model is not None:
            render_pkg_ref = self.gs_render_local.render(
                viewpoint_cam_ref,
                compute_cov3D_python=pipe.compute_cov3D_python,
                convert_SHs_python=pipe.convert_SHs_python)

            gt_image_current = viewpoint_cam_ref.original_image.cuda()
            gt_image_prev = viewpoint_cam.original_image.cuda()

            with torch.no_grad():
                precomputed_flow = self.compute_flow_between_views(gt_image_prev, gt_image_current)

        # 开始 Tracking 帧计时
        if self.timing_stats is not None:
            self.timing_stats.start_tracking_frame()

        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render_local.gaussians.update_learning_rate(iteration)
            loss, rend_dict_ref, psnr_train = self.train_step_flow(self.gs_render_local,
                                                              viewpoint_cam_ref,
                                                              viewpoint_cam,
                                                              rend_dict,
                                                              pts,
                                                              iteration,
                                                              pipe,
                                                              optim_opt,
                                                              densify=False,
                                                              use_reproject=True,
                                                              ref_fidx=view_idx_prev,
                                                              precomputed_flow=precomputed_flow
                                                              )

            # 提前停止检测（仅在启用时执行）
            if enable_early_stop:
                psnr_history.append(psnr_train.item())
                if iteration >= early_stop_window:
                    recent_psnrs = psnr_history[-early_stop_window:]
                    psnr_change = max(recent_psnrs) - min(recent_psnrs)
                    avg_psnr = sum(recent_psnrs) / early_stop_window

                    if avg_psnr > target_psnr and psnr_change < early_stop_threshold:
                        print(f"\n[Tracking] Early stop at iter {iteration}: PSNR={avg_psnr:.2f}, change={psnr_change:.3f}")
                        progress_bar.close()
                        break

            # 优化：减少进度条更新频率，降低开销
            if iteration % 20 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}",
                                          "flow loss": f"{loss.get('flow_loss', 0.0):.4f}"})
                progress_bar.update(20)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        # 结束 Tracking 帧计时
        if self.timing_stats is not None:
            self.timing_stats.stop_tracking_frame()

        local_model_params = self.gs_render_local.gaussians.capture()

        # pcd under view_idx_prev frame
        pcd = self.gs_render_local.gaussians._xyz.detach()
        rel_pose = self.gs_render_local.gaussians.get_RT().detach()
        pose = rel_pose @ self.gs_render.gaussians.get_RT(
            view_idx_prev).detach()
        self.gs_render.gaussians.update_RT_seq(pose, view_idx)

        self.gs_render.gaussians.rotate_seq = False
        pipe.convert_SHs_python = self.gs_render.gaussians.rotate_seq

        if self.just_reset:
            num_iterations = 500
            self.just_reset = False
            for iteration in range(1, num_iterations):
                fidx = randint(0, view_idx_prev)
                self.global_iteration += 1
                self.gs_render.gaussians.update_learning_rate(
                    self.global_iteration)
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        load_depth=True)
                loss, rend_dict_ref, psnr_train = self.train_step(self.gs_render,
                                                                  viewpoint_cam,
                                                                  self.global_iteration,
                                                                  pipe, self.optim_cfg,
                                                                  update_gaussians=True,
                                                                  update_cam=False,
                                                                  #   depth_gt=self.mono_depth[fidx],
                                                                  update_distort=False,
                                                                  )

        num_iterations = self.single_step
        if max(view_idx, view_idx_prev) > min(int(self.seq_len * 0.8), self.seq_len-5):
            num_iterations = 1000
        # elif min(view_idx, view_idx_prev) < int(self.single_step // 100):
        elif min(view_idx, view_idx_prev) < 50:
            num_iterations = 100

        progress_bar = tqdm(range(num_iterations), desc="Training progress global")

        # 提前停止机制：Mapping阶段（可通过参数控制）
        mapping_psnr_history = []
        mapping_early_stop_window = 20  # Mapping阶段检查更长窗口
        mapping_early_stop_threshold = 0.1  # 更严格的收敛标准
        mapping_target_psnr = 32.0  # 更高的目标PSNR
        enable_early_stop = optim_opt.enable_early_stop  # 从参数读取是否启用

        # 开始 Mapping 帧计时
        if self.timing_stats is not None:
            self.timing_stats.start_mapping_frame()

        for iteration in range(1, num_iterations+1):
            last_frame = max(1, view_idx//2)
            if random.random() < 0.7:
                fidx = randint(last_frame, view_idx)
            else:
                fidx = randint(1, last_frame)

            self.global_iteration += 1
            if self.gs_render.gaussians.rotate_seq:
                self.gs_render.gaussians.set_seq_idx(fidx)
            viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                    pose=self.gs_render.gaussians.get_RT(
                                                        fidx).detach().cpu()
                                                    if not self.gs_render.gaussians.rotate_seq
                                                    else None,
                                                    load_depth=True)
            # Update learning rate
            self.gs_render.gaussians.update_learning_rate(
                self.global_iteration)

            loss, rend_dict_ref, psnr_train = self.train_step(self.gs_render,
                                                              viewpoint_cam,
                                                              self.global_iteration,
                                                              pipe, self.optim_cfg,
                                                              update_gaussians=True,
                                                              update_cam=False,
                                                              #   depth_gt=self.mono_depth[fidx],
                                                              update_distort=self.pipe_cfg.distortion,
                                                              )

            # 提前停止检测（仅在启用时执行）
            if enable_early_stop:
                mapping_psnr_history.append(psnr_train.item())
                if iteration >= mapping_early_stop_window:
                    recent_psnrs = mapping_psnr_history[-mapping_early_stop_window:]
                    psnr_change = max(recent_psnrs) - min(recent_psnrs)
                    avg_psnr = sum(recent_psnrs) / mapping_early_stop_window

                    if avg_psnr > mapping_target_psnr and psnr_change < mapping_early_stop_threshold:
                        print(f"\n[Mapping] Early stop at iter {iteration}: PSNR={avg_psnr:.2f}, change={psnr_change:.3f}")
                        progress_bar.close()
                        break

            if self.global_iteration % 1000 == 0:
                self.gs_render.gaussians.oneupSHdegree()

            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)

            if iteration == num_iterations:
                progress_bar.close()

        # 结束 Mapping 帧计时
        if self.timing_stats is not None:
            self.timing_stats.stop_mapping_frame()

        return pcd, local_model_params

    def create_pcd_from_render(self, render_dict, viewpoint_cam):
        intrinsics = torch.from_numpy(viewpoint_cam.intrinsics).float().cuda()
        depth = render_dict["depth"].squeeze()
        image = render_dict["image"]
        pts = depth_to_3d(depth[None, None],
                          intrinsics[None],
                          normalize_points=False)
        points = pts.squeeze().permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()
        colors = image.permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data = pcd_data.farthest_point_down_sample(num_samples=30_000)
        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)
        pcd = BasicPointCloud(points, colors, normals)
        return pcd

    def train_from_progressive(self, ):
        pipe = copy(self.pipe_cfg)
        self.single_step = 300 # 300 for faster training; 500 for better results

        num_iterations = self.single_step * (self.seq_len // 10) * 10
        self.optim_cfg.iterations = num_iterations
        self.optim_cfg.position_lr_max_steps = num_iterations
        self.optim_cfg.opacity_reset_interval = num_iterations // 10
        self.optim_cfg.densify_until_iter = num_iterations
        self.optim_cfg.reset_until_iter = int(num_iterations * 0.8)
        self.optim_cfg.densify_from_iter = 1000
        self.optim_cfg.densify_from_iter = self.single_step


        if pipe.expname == "":
            expname = "progressive"
        else:
            expname = pipe.expname
        pipe.convert_SHs_python = True
        optim_opt = copy(self.optim_cfg)
        result_path = f"output/{expname}/{self.seq_name}_{self.model_cfg.start_cam}_{self.model_cfg.end_cam}"
        os.makedirs(result_path, exist_ok=True)

        # 初始化时间统计器
        self.timing_stats = TimingStats(output_dir=result_path)
        print("\n" + "="*70)
        print("Time measurement enabled (EndoGSLAM compatible format)")
        print("="*70 + "\n")

        pose_dict = dict()
        poses_gt = []
        for seq_data in self.data:
            if self.data_type == "co3d":
                R, t, _, _, _ = self.load_camera(seq_data)
            else:
                try:
                    R = seq_data.R.transpose()
                    t = seq_data.T
                except:
                    R = np.eye(3)
                    t = np.zeros(3)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            poses_gt.append(torch.from_numpy(pose))
        pose_dict["poses_gt"] = torch.stack(poses_gt)
        max_frame = self.seq_len
        start_frame = 1
        end_frame = max_frame

        os.makedirs(f"{result_path}/pose", exist_ok=True)
        os.makedirs(f"{result_path}/mesh", exist_ok=True)

        num_eppch = 1
        reverse = False
        for epoch in range(num_eppch):
            # 计时：初始化两帧
            self.timing_stats.start_init()
            gauss_params = self.init_two_view(
                0, end_frame, pipe, copy(self.optim_cfg))
            self.timing_stats.stop_init()

            self.global_iteration = 0
            optim_opt = copy(self.optim_cfg)
            self.gs_render.gaussians.rotate_seq = True
            self.gs_render.gaussians.training_setup(self.optim_cfg,
                                                    fit_pose=True,)
            self.match_results = OrderedDict()
            for fidx in range(start_frame, end_frame):
                # 开始帧计时
                self.timing_stats.start_frame(fidx)

                # pcd_new, local_gauss_params = self.add_view(
                #     None, fidx, fidx-1, pipe, optim_opt, reverse=reverse)
                pcd_new, local_gauss_params = self.add_view_v2(
                    fidx, fidx-1)
                self.gs_render.gaussians.rotate_seq = False
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        )
                render_dict = self.gs_render.render(viewpoint_cam,
                                                    compute_cov3D_python=pipe.compute_cov3D_python,
                                                    convert_SHs_python=pipe.convert_SHs_python)
                gt_image = viewpoint_cam.original_image.cuda()
                try:
                    image_name = int(viewpoint_cam.image_name.split('_')[0])
                except:
                    suffix = viewpoint_cam.image_name.split('_')[1]
                    if suffix.startswith('L'):
                        image_name = int(suffix.strip('L'))
                    else:
                        image_name = int(suffix)
                psnr_train = psnr(render_dict["image"],
                                    gt_image).mean().double()
                print(
                    'Frames {:03d}/{:03d}, PSNR : {:.03f}'.format(fidx, self.seq_len-1, psnr_train))
                self.visualize(render_dict,
                                f"{result_path}/train/{self.global_iteration:06d}_{image_name:04d}.png",
                                gt_image=gt_image, save_ply=False)

                # 结束帧计时
                self.timing_stats.end_frame()

                # 每10帧打印一次时间统计
                if (fidx - start_frame + 1) % 10 == 0:
                    print(f"\n========== Progress: {fidx - start_frame + 1}/{end_frame - start_frame} frames ==========")
                    self.timing_stats.print_statistics()
                    print("=" * 70 + "\n")

            with torch.no_grad():
                psnr_test = 0.0
                pose_dict["poses_pred"] = []
                self.render_depth = OrderedDict()
                self.gs_render.gaussians.rotate_seq = False
                self.gs_render.gaussians.rotate_xyz = False

                for val_idx in range(end_frame):
                    viewpoint_cam = self.load_viewpoint_cam(val_idx,
                                                            pose=self.gs_render.gaussians.get_RT(
                                                                val_idx).detach().cpu(),
                                                            )
                    render_dict = self.gs_render.render(viewpoint_cam,
                                                        compute_cov3D_python=pipe.compute_cov3D_python,
                                                        convert_SHs_python=pipe.convert_SHs_python)
                    self.render_depth[val_idx] = render_dict["depth"]
                    gt_image = viewpoint_cam.original_image.cuda()
                    try:
                        image_name = int(viewpoint_cam.image_name.split('_')[0])
                    except (ValueError, IndexError):
                        try:
                            parts = viewpoint_cam.image_name.split('_')
                            if len(parts) > 1:
                                image_name = int(parts[1])
                            else:
                                import re
                                numbers = re.findall(r'\d+', viewpoint_cam.image_name)
                                image_name = int(numbers[0]) if numbers else val_idx
                        except:
                            image_name = val_idx

                    psnr_test += psnr(render_dict["image"],
                                        gt_image).mean().double()
                    self.visualize(render_dict,
                                    f"{result_path}/eval/ep{epoch:02d}_{self.global_iteration:06d}_{image_name:04d}.png",
                                    gt_image=gt_image, save_ply=False)
                print('Number of {:03d} to {:03d} frames: PSNR : {:.03f}'.format(
                    start_frame,
                    end_frame,
                    psnr_test / (end_frame)))

                for idx in range(self.seq_len):
                    pose = self.gs_render.gaussians.get_RT(idx)
                    pose_dict["poses_pred"].append(pose.detach().cpu())

            pose_dict["poses_pred"] = torch.stack(pose_dict["poses_pred"])
            pose_dict["poses_gt"] = torch.stack(poses_gt)
            pose_dict["match_results"] = self.match_results
            torch.save(
                pose_dict, f"{result_path}/pose/ep{epoch:02d}_init.pth")
            os.makedirs(f"{result_path}/chkpnt", exist_ok=True)
            torch.save(self.gs_render.gaussians.capture(),
                        f"{result_path}/chkpnt/ep{epoch:02d}_init.pth")

        # 打印和保存时间统计
        self.timing_stats.print_statistics()
        self.timing_stats.save_statistics()
        self.timing_stats.save_frame_details()



    def eval_reconstruction(self):
        """Evaluate RGB reconstruction quality and renderer throughput only."""
        if not self.model_cfg.model_path:
            raise ValueError("--model_path is required for reconstruction evaluation")
        if self.seq_len == 0:
            raise ValueError("The reconstruction evaluation set is empty")

        pipe = copy(self.pipe_cfg)
        pipe.convert_SHs_python = True
        optim_opt = copy(self.optim_cfg)
        num_iterations = self.optim_cfg.eval_reconstruction_epochs * self.seq_len
        optim_opt.iterations = num_iterations
        optim_opt.position_lr_max_steps = num_iterations
        optim_opt.densify_until_iter = num_iterations // 2
        optim_opt.reset_until_iter = num_iterations // 2
        optim_opt.opacity_reset_interval = max(1, num_iterations // 10)
        optim_opt.densification_interval = 100
        optim_opt.densify_from_iter = 500

        result_path = os.path.dirname(
            self.model_cfg.model_path
        ).replace("chkpnt", "test")
        frame_dir = os.path.join(result_path, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        self.gs_render.gaussians.restore(
            torch.load(self.model_cfg.model_path), self.optim_cfg
        )
        pose_dict_train = torch.load(
            self.model_cfg.model_path.replace("chkpnt", "pose")
        )
        total_frames = len(pose_dict_train["poses_pred"])
        indices = np.linspace(0, total_frames - 1, self.seq_len).astype(int)
        pose_test_init = torch.stack(
            [pose_dict_train["poses_pred"][index] for index in indices]
        )

        self.gs_render.gaussians.init_RT_seq(
            self.seq_len, pose_test_init.float()
        )
        self.gs_render.gaussians.rotate_seq = True
        self.gs_render.gaussians.training_setup(
            optim_opt,
            fix_pos=True,
            fix_feat=True,
            fit_pose=True,
        )

        progress_bar = tqdm(
            range(num_iterations), desc="Optimizing evaluation camera poses"
        )
        iteration = 0
        for _ in range(self.optim_cfg.eval_reconstruction_epochs):
            for frame_index in range(self.seq_len):
                iteration += 1
                self.gs_render.gaussians.rotate_seq = True
                self.gs_render.gaussians.set_seq_idx(frame_index)
                viewpoint_cam = self.load_viewpoint_cam(
                    frame_index, pose=None, load_depth=False
                )
                _, _, psnr_train = self.train_step(
                    self.gs_render,
                    viewpoint_cam,
                    iteration,
                    pipe,
                    optim_opt,
                    densify=False,
                    depth_gt=None,
                    update_cam=True,
                    update_gaussians=False,
                    reset=False,
                )
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"PSNR": f"{psnr_train:.2f}"})
                    progress_bar.update(10)
        progress_bar.close()

        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0
        rendering_time = 0.0
        frame_paths = []

        with torch.no_grad():
            for frame_index in range(self.seq_len):
                self.gs_render.gaussians.rotate_seq = False
                viewpoint_cam = self.load_viewpoint_cam(
                    frame_index,
                    pose=self.gs_render.gaussians.get_RT(
                        frame_index
                    ).detach().cpu(),
                    load_depth=False,
                )

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                render_dict = self.gs_render.render(
                    viewpoint_cam,
                    compute_cov3D_python=False,
                    convert_SHs_python=False,
                )
                torch.cuda.synchronize()
                rendering_time += time.perf_counter() - start_time

                gt_image = viewpoint_cam.original_image.cuda()
                psnr_total += psnr(render_dict["image"], gt_image).mean().item()
                ssim_total += ssim(render_dict["image"], gt_image).mean().item()
                lpips_total += lpips(
                    render_dict["image"], gt_image, net_type="vgg"
                ).mean().item()

                frame_path = os.path.join(frame_dir, f"{frame_index:04d}.png")
                self.visualize(
                    render_dict, frame_path, gt_image=gt_image, save_ply=False
                )
                frame_paths.append(frame_path)

        reconstruction = {
            "frames": self.seq_len,
            "psnr": psnr_total / self.seq_len,
            "ssim": ssim_total / self.seq_len,
            "lpips": lpips_total / self.seq_len,
        }
        rendering_fps = self.seq_len / rendering_time

        video_path = os.path.join(result_path, "reconstruction.mp4")
        with imageio.get_writer(
            video_path,
            fps=self.model_cfg.video_fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
            output_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        ) as video:
            for frame_path in frame_paths:
                video.append_data(imageio.imread(frame_path))

        report = {
            "reconstruction": {
                **reconstruction,
                "video": os.path.basename(video_path),
            },
            "rendering": {
                "fps": rendering_fps,
            },
        }
        report_path = os.path.join(result_path, "report.json")
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)

        markdown_path = os.path.join(result_path, "report.md")
        with open(markdown_path, "w", encoding="utf-8") as markdown_file:
            markdown_file.write(
                "# Reconstruction report\n\n"
                f"- Reconstruction: PSNR {reconstruction['psnr']:.3f}, "
                f"SSIM {reconstruction['ssim']:.3f}, "
                f"LPIPS {reconstruction['lpips']:.3f}; "
                f"video `{os.path.basename(video_path)}` "
                f"({self.seq_len} frames).\n"
                f"- Rendering FPS: {rendering_fps:.2f}.\n"
            )

        print(
            f"Reconstruction: PSNR {reconstruction['psnr']:.3f}, "
            f"SSIM {reconstruction['ssim']:.3f}, "
            f"LPIPS {reconstruction['lpips']:.3f}"
        )
        print(f"Rendering FPS: {rendering_fps:.2f}")
        print(f"Video: {video_path}")
        print(f"Report: {report_path}")

    def save_model(self, epoch):
        pass

    def compute_loss(self,
                     render_dict,
                     viewpoint_cam,
                     pipe_opt,
                     iteration,
                     use_reproject=False,
                     use_matcher=False,
                     ref_fidx=None,
                     **kwargs):
        loss = 0.0
        if "image" in render_dict:
            image = render_dict["image"]
            gt_image = viewpoint_cam.original_image.cuda()
        if "depth" in render_dict:
            depth = render_dict["depth"]
            depth[depth < self.near] = self.near
            fidx = viewpoint_cam.uid
            kwargs['depth_pred'] = depth

        loss_dict = self.loss_func(image, gt_image, **kwargs)
        return loss_dict

    def visualize(self, render_pkg, filename, gt_image=None, save_ply=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if "acc" in render_pkg:
            rend_acc = Image.fromarray(
                colorize(render_pkg["acc"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            rend_acc.save(filename.replace(".png", "_acc.png"))

        rend_img = Image.fromarray(
            np.asarray(render_pkg["image"].detach().cpu().permute(1, 2, 0).numpy()
                       * 255.0, dtype=np.uint8)).convert("RGB")
        # if gt_image is not None:
        #     gt_image = Image.fromarray(
        #         np.asarray(
        #             gt_image.permute(1, 2, 0).cpu().numpy() * 255.0,
        #             dtype=np.uint8)).convert("RGB")
        #     rend_img = Image.fromarray(np.hstack([np.asarray(gt_image),
        #                                           np.asarray(rend_img)]))
        rend_img.save(filename)

        if save_ply:
            points = self.gs_render.gaussians._xyz.detach().cpu().numpy()
            pcd_data = o3d.geometry.PointCloud()
            pcd_data.points = o3d.utility.Vector3dVector(points)
            pcd_data.colors = o3d.utility.Vector3dVector(np.ones_like(points))
            o3d.io.write_point_cloud(
                filename.replace('.png', '.ply'), pcd_data)


    def construct_point(self, gs_model, poses, iteration, result_path, stop_frame=-1):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=3 * 0.01,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        pipe = self.pipe_cfg
        optim_opt = self.optim_cfg
        pipe.convert_SHs_python = True
        if poses is None:
            poses = torch.stack(
                [gs_model.gaussians.get_RT(idx).detach().cpu()
                 for idx in range(self.seq_len)])

        self.gs_render.gaussians.rotate_seq = False
        stop_frame = len(poses) if stop_frame == -1 else stop_frame
        with torch.no_grad():
            progress_bar = tqdm(range(self.seq_len),
                                desc="Reconstructing point cloud")
            for idx in range(len(poses)):
                if idx > stop_frame:
                    break

                viewpoint_cam = self.load_viewpoint_cam(
                    idx, pose=poses[idx], load_depth=True)
                # if idx not in self.render_depth:
                render_dict = gs_model.render(
                    viewpoint_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python)
                render_depth = render_dict['depth'].detach().squeeze()

                rgb = viewpoint_cam.original_image.cuda().permute(1, 2, 0).detach().cpu().numpy()
                rgb = (rgb * 255).astype(np.uint8)

                depth = render_depth.detach().cpu().numpy()


                H, W = depth.shape
                intrinsic = viewpoint_cam.intrinsics
                fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

                rgb = o3d.geometry.Image(rgb)
                depth = o3d.geometry.Image(depth)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb, depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
                )
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
                # pose = self.gs_render.gaussians.get_RT(idx).detach().cpu().numpy()
                volume.integrate(rgbd, intrinsic, poses[idx])
                progress_bar.update(1)
        progress_bar.close()

        self.gs_render.gaussians.rotate_seq = True
        mesh = volume.extract_triangle_mesh()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            f"{result_path}/{self.gs_render.rot_type}_{iteration:06d}.ply", mesh)

        points = np.asarray(mesh.vertices, dtype=np.float32)
        colors = np.asarray(mesh.vertex_colors)
        normals = np.asarray(mesh.vertex_normals)
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data.normals = o3d.utility.Vector3dVector(normals)

        pcd_data = pcd_data.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(
            f"{result_path}/{self.gs_render.rot_type}_{iteration:06d}.ply", pcd_data)
        points = np.asarray(pcd_data.points)
        colors = np.asarray(pcd_data.colors)
        normals = np.asarray(pcd_data.normals)
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        return pcd
