# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import copy
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.main_ape import ape
from evo.tools import plot
from evo.core import sync
from evo.tools import file_interface
from evo.core import metrics
import evo
import torch
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import scipy.interpolate as si


def interp_poses(c2ws, N_views):
    N_inputs = c2ws.shape[0]
    trans = c2ws[:, :3, 3:].permute(2, 1, 0)
    rots = c2ws[:, :3, :3]
    render_poses = []
    rots = R.from_matrix(rots)
    slerp = Slerp(np.linspace(0, 1, N_inputs), rots)
    interp_rots = torch.tensor(
        slerp(np.linspace(0, 1, N_views)).as_matrix().astype(np.float32))
    interp_trans = torch.nn.functional.interpolate(
        trans, size=N_views, mode='linear').permute(2, 1, 0)
    render_poses = torch.cat([interp_rots, interp_trans], dim=2)
    render_poses = convert3x4_4x4(render_poses)
    return render_poses


def interp_poses_bspline(c2ws, N_novel_imgs, input_times, degree):
    target_trans = torch.tensor(scipy_bspline(
        c2ws[:, :3, 3], n=N_novel_imgs, degree=degree, periodic=False).astype(np.float32)).unsqueeze(2)
    rots = R.from_matrix(c2ws[:, :3, :3])
    slerp = Slerp(input_times, rots)
    target_times = np.linspace(input_times[0], input_times[-1], N_novel_imgs)
    target_rots = torch.tensor(
        slerp(target_times).as_matrix().astype(np.float32))
    target_poses = torch.cat([target_rots, target_trans], dim=2)
    target_poses = convert3x4_4x4(target_poses)
    return target_poses


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # c = np.dot(c2w[:3,:4], np.array([0.7*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.1, 1.]) * rads)
        # c = np.dot(c2w[:3,:4], np.array([0.3*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.01, 1.]) * rads)
        c = np.dot(c2w[:3, :4], np.array(
            [0.2*np.cos(theta), -0.2*np.sin(theta), -np.sin(theta*zrate) * 0.1, 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def scipy_bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree, count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate(
            (cv,) * factor + (cv[:fraction],)), -1, axis=0)
        degree = np.clip(degree, 1, degree)

    # Opened curve
    else:
        degree = np.clip(degree, 1, count-1)
        kv = np.clip(np.arange(count+degree+1)-degree, 0, count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0, max_param, n))


def generate_spiral_nerf(learned_poses, bds, N_novel_views, hwf):
    learned_poses_ = np.concatenate((learned_poses[:, :3, :4].detach(
    ).cpu().numpy(), hwf[:len(learned_poses)]), axis=-1)
    c2w = poses_avg(learned_poses_)
    print('recentered', c2w.shape)
    # Get spiral
    # Get average pose
    up = normalize(learned_poses_[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset

    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = learned_poses_[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_rots = 2
    c2ws = render_path_spiral(
        c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_novel_views)
    c2ws = torch.tensor(np.stack(c2ws).astype(np.float32))
    c2ws = c2ws[:, :3, :4]
    c2ws = convert3x4_4x4(c2ws)
    return c2ws


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(
                input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor(
                [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate(
                [input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


plt.rc('legend', fontsize=20)  # using a named size


def plot_pose(ref_poses, est_poses, output_path, vid=False,fig_size=(10, 8),
              set_limits=True, pose_name = 'pose_vis.png', view_angle=None, view_ele = None):
    ref_poses = [pose for pose in ref_poses]
    if isinstance(est_poses, dict):
        est_poses = [pose for k, pose in est_poses.items()]
    else:
        est_poses = [pose for pose in est_poses]
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True,
                           correct_only_scale=False)

    def compute_traj_bounds(traj_list):
        all_positions = np.vstack([traj.positions_xyz for traj in traj_list])
        min_vals = np.min(all_positions, axis=0)
        max_vals = np.max(all_positions, axis=0)
        return min_vals, max_vals

    # 计算轨迹的边界
    min_pos, max_pos = compute_traj_bounds([traj_ref, traj_est_aligned])

    # 添加边界缓冲
    # range_vals = max_pos - min_pos
    # buffer = range_vals * 0.1  # 10% 缓冲区
    # min_pos -= buffer
    # max_pos += buffer

    # 确保XYZ轴具有相同的比例尺
    # center = (min_pos + max_pos) / 2
    # max_range = np.max(max_pos - min_pos) / 2
    # min_bounds = center - max_range
    # max_bounds = center + max_range

    if vid:
        for p_idx in range(len(ref_poses)):
            fig = plt.figure()
            current_est_aligned = traj_est_aligned.poses_se3[:p_idx+1]
            current_ref = traj_ref.poses_se3[:p_idx+1]
            current_est_aligned = PosePath3D(poses_se3=current_est_aligned)
            current_ref = PosePath3D(poses_se3=current_ref)
            traj_by_label = {
                # "estimate (not aligned)": traj_est,
                "Ours (aligned)": current_est_aligned,
                "Ground-truth": current_ref
            }
            plot_mode = plot.PlotMode.xyz
            # ax = plot.prepare_axis(fig, plot_mode, 111)
            ax = fig.add_subplot(111, projection="3d")
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelleft=False)
            colors = ['r', 'b']
            styles = ['-', '--']

            for idx, (label, traj) in enumerate(traj_by_label.items()):
                plot.traj(ax, plot_mode, traj,
                          styles[idx], colors[idx], label)
                # break
            # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            ax.view_init(elev=10., azim=45)
            plt.tight_layout()
            os.makedirs(os.path.join(os.path.dirname(
                output_path), 'pose_vid'), exist_ok=True)
            pose_vis_path = os.path.join(os.path.dirname(
                output_path), 'pose_vid', 'pose_vis_{:03d}.png'.format(p_idx))
            print(pose_vis_path)
            fig.savefig(pose_vis_path)

    # else:

    # 主静态轨迹图
    fig = plt.figure(figsize=fig_size)
    traj_by_label = {
        "Ours (aligned)": traj_est_aligned,
        "Ground-truth": traj_ref
    }
    plot_mode = plot.PlotMode.xyz
    ax = fig.add_subplot(111, projection="3d")

    # 隐藏刻度
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.zaxis.set_tick_params(labelleft=False)

    # 设置坐标轴范围
    # if set_limits:
    #     ax.set_xlim(min_bounds[0], max_bounds[0])
    #     ax.set_ylim(min_bounds[1], max_bounds[1])
    #     ax.set_zlim(min_bounds[2], max_bounds[2])
    #     ax.set_box_aspect((1, 1, 1))  # 确保XYZ轴具有相同的比例尺

    # 添加轴标签（可选）
    ax.set_xlabel('X', labelpad=-10)
    ax.set_ylabel('Y', labelpad=-10)
    ax.set_zlabel('Z', labelpad=-10)

    colors = ['r', 'b']
    styles = ['-', '--']

    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj,
                  styles[idx], colors[idx], label)

    # 设置更好的视角
    elev = view_ele if view_ele is not None else 15.
    azim = view_angle if view_angle is not None else 45.
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()


    pose_vis_path = os.path.join(os.path.dirname(output_path), pose_name)
    fig.savefig(pose_vis_path, dpi=300)  # 增加DPI获得更高分辨率
    plt.close(fig)
    # return pose_vis_path  # 返回保存的图像路径
def plot_fixed_gt_pose(gt_poses, pred_poses, save_dir, fig_size=(12, 10),
                        view_angle=30, view_ele=30, pose_name="pose_fixed_gt.png"):
    """
    绘制轨迹对比图，保持GT轨迹固定，仅对齐预测轨迹

    Args:
        gt_poses: 真实轨迹，形状为[N, 4, 4]
        pred_poses: 对齐后的预测轨迹，形状为[N, 4, 4]
        save_dir: 保存目录
        fig_size: 图像大小
        view_angle: 观察角度（水平方向）
        view_ele: 观察仰角
        pose_name: 保存的文件名
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # 提取位置信息
    gt_positions = gt_poses[:, :3, 3]
    pred_positions = pred_poses[:, :3, 3]

    # 绘制GT轨迹（蓝色）
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'b-', linewidth=1, label='Ground Truth')
    ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], c='b', marker='o', s=50)  # 起点

    # 绘制预测轨迹（红色）
    ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 'r-', linewidth=1, label='Predicted')
    ax.scatter(pred_positions[0, 0], pred_positions[0, 1], pred_positions[0, 2], c='r', marker='o', s=50)  # 起点

    # 绘制相机位姿（仅绘制每10帧一个）
    camera_size = 0.05  # 相机可视化大小
    for i in range(0, len(gt_poses), 10):
        # 绘制GT相机位姿
        R_gt = gt_poses[i, :3, :3]
        t_gt = gt_poses[i, :3, 3]
        plot_camera(ax, R_gt, t_gt, camera_size, 'b')

        # 绘制预测相机位姿
        R_pred = pred_poses[i, :3, :3]
        t_pred = pred_poses[i, :3, 3]
        plot_camera(ax, R_pred, t_pred, camera_size, 'r')

    # 设置图表属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # 设置视角
    ax.view_init(elev=view_ele, azim=view_angle)

    # 设置相等的坐标轴比例
    max_range = np.max([
        np.max(np.abs(gt_positions.max(axis=0) - gt_positions.min(axis=0))),
        np.max(np.abs(pred_positions.max(axis=0) - pred_positions.min(axis=0)))
    ])

    # 获取轨迹中心点
    mid_gt = (gt_positions.max(axis=0) + gt_positions.min(axis=0)) / 2

    # 设置轴范围以保持GT轨迹居中
    ax.set_xlim(mid_gt[0] - max_range / 2, mid_gt[0] + max_range / 2)
    ax.set_ylim(mid_gt[1] - max_range / 2, mid_gt[1] + max_range / 2)
    ax.set_zlim(mid_gt[2] - max_range / 2, mid_gt[2] + max_range / 2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, pose_name), dpi=150)
    plt.close()

def plot_camera(ax, R, t, size, color):
    """
    在3D图上绘制简化的相机模型
    """
    # 相机坐标系
    origin = t
    x_axis = origin + size * R[:, 0]
    y_axis = origin + size * R[:, 1]
    z_axis = origin + size * R[:, 2]

    # 绘制坐标轴
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color=color, linewidth=0.5)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color=color, linewidth=0.5)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color=color, linewidth=0.5)

def umeyama_alignment(pred, gt):
    """
    计算从pred到gt的相似变换（缩放、旋转和平移）
    返回缩放因子s，旋转矩阵R和平移向量t
    """
    # 确保输入是numpy数组
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    # 计算质心
    pred_mean = np.mean(pred, axis=0)
    gt_mean = np.mean(gt, axis=0)

    # 去中心化
    pred_demean = pred - pred_mean
    gt_demean = gt - gt_mean

    # 计算缩放因子
    pred_var = np.mean(np.sum(pred_demean**2, axis=1))
    s = 1.0 if pred_var == 0 else np.sqrt(np.mean(np.sum(gt_demean**2, axis=1)) / pred_var)

    # 计算旋转矩阵
    U, _, Vt = np.linalg.svd(gt_demean.T @ pred_demean)
    R = U @ Vt

    # 确保旋转矩阵是正交的
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # 计算平移向量
    t = gt_mean - s * (R @ pred_mean)

    return s, R, t