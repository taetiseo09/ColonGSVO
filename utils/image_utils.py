# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE_inria.md file.
#
# For inquiries contact  george.drettakis@inria.fr


import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import cv2
from PIL import Image

class CameraPoseVisualizer:
    def __init__(self, xlim=None, ylim=None, zlim=None):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        if xlim is not None and ylim is not None and zlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_zlim(zlim)

        self.ax.view_init(elev=10., azim=45)
        self.ax.xaxis.set_tick_params(labelbottom=False)
        self.ax.yaxis.set_tick_params(labelleft=False)
        self.ax.zaxis.set_tick_params(labelleft=False)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        # plt.tight_layout()
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

    def add_traj(self, poses, c='r', alpha=0.5):
        x = [float(pose[0, 3]) for pose in poses]
        y = [float(pose[1, 3]) for pose in poses]
        z = [float(pose[2, 3]) for pose in poses]
        self.ax.plot(x,y,z, c=c, alpha=alpha)

    def save(self, path):
        plt.tight_layout()
        self.fig.savefig(path, bbox_inches='tight')


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    if hasattr(mpl, 'colormaps'):
        # 新版本 Matplotlib 3.7+
        cmapper = mpl.colormaps[cmap]
    else:
        # 旧版本 Matplotlib
        cmapper = mpl.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img



def cm_prune(x_):
    """Custom colormap to visualize pruning"""
    if isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
    max_i = max(x_)
    norm_x = np.where(x_ == max_i, -1, (x_ - 1) / 9)
    return cm_BlRdGn(norm_x)

def align_brightness_with_flow(img1, img2, flow, flow_threshold=1.0, edge_smoothness=21,
                              dilate_size=7, dilate_iter=3, edge_window=15, inverse_mask=False):
    """基于光流的亮度对齐函数

    Args:
        img1: 参考图像 (numpy数组或PIL图像或PyTorch张量)
        img2: 目标图像 (numpy数组或PIL图像或PyTorch张量)
        flow: 光流张量 [1, 2, H, W]
        flow_threshold: 光流大小阈值，决定掩码区域
        edge_smoothness: 边缘平滑度，值越大过渡越平滑
        dilate_size: 掩码扩张核大小
        dilate_iter: 掩码扩张迭代次数
        edge_window: 局部边缘平滑窗口大小

    Returns:
        亮度对齐后的图像 (numpy数组，与输入图像格式相同)
    """
    def visualize_mask(mask, output_path="./", prefix=""):
        """可视化并保存掩码图像

        Args:
            mask: 二值掩码，值为0或1的numpy数组
            output_path: 输出目录
            prefix: 文件名前缀
        """
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # 保存原始掩码为灰度图
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f'{prefix}mask_binary.png'), mask_uint8)

        # 保存彩色可视化版本
        # 创建一个3通道图像，全黑背景
        h, w = mask.shape
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)

        # 设置掩码区域为蓝色
        mask_color[mask > 0.5] = [255, 0, 0]  # 对于OpenCV, 这是蓝色 (BGR顺序)

        # 保存彩色掩码
        cv2.imwrite(os.path.join(output_path, f'{prefix}mask_color.png'), mask_color)

        # 可选: 保存轮廓叠加图
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        edges = mask_dilated - mask_eroded

        # 保存轮廓图
        cv2.imwrite(os.path.join(output_path, f'{prefix}mask_edges.png'), edges)

        print(f"掩码已保存到 {output_path}")

        non_zero = np.count_nonzero(mask)
        total = mask.size
        percentage = (non_zero / total) * 100
        print(f"掩码覆盖率: {percentage:.2f}% ({non_zero} / {total} 像素)")

    def check_mask(mask, name="掩码"):
        """检查掩码的统计信息，判断是否全为1或全为0"""
        # 计算非零元素的数量和比例
        non_zero = np.count_nonzero(mask)
        total = mask.size
        percentage = (non_zero / total) * 100

        # 打印详细信息
        # print(f"--- {name}统计信息 ---")
        # print(f"总像素数: {total}")
        # print(f"非零像素数: {non_zero}")
        # print(f"覆盖率: {percentage:.2f}%")

        # 判断掩码状态
        if non_zero == total or percentage > 90:
            # print(f"警告: {name}全为1!")
            return True
        elif non_zero == 0:
            # print(f"警告: {name}全为0!")
            return True
        else:
            # print(f"{name}正常 (部分为1，部分为0)")
            return False

    if torch.is_tensor(img1):
        if img1.dim() == 4:  # [1, C, H, W] 格式
            img1_np = img1[0].permute(1, 2, 0).cpu().numpy()
            img2_np = img2[0].permute(1, 2, 0).cpu().numpy()
        else:  # [C, H, W] 格式
            img1_np = img1.permute(1, 2, 0).cpu().numpy()
            img2_np = img2.permute(1, 2, 0).cpu().numpy()
    elif isinstance(img1, Image.Image):
        img1_np = np.array(img1)
        img2_np = np.array(img2)
    else:  # 假设已经是numpy数组
        img1_np = img1.copy()
        img2_np = img2.copy()

    # 确保图像是uint8类型或在[0,1]范围内的浮点型
    if img1_np.dtype != np.uint8 and img1_np.max() > 1.0:
        img1_np = img1_np.astype(np.uint8)
        img2_np = img2_np.astype(np.uint8)

    # 创建光流掩码
    flow_mag = torch.sqrt(flow[0, 0, :, :]**2 + flow[0, 1, :, :]**2)
    if inverse_mask:
        binary_mask = (flow_mag <= flow_threshold).float().cpu().numpy()  # 反向选择
        mask_type = "inverse"
    else:
        binary_mask = (flow_mag > flow_threshold).float().cpu().numpy()  # 正向选择（原始方式）
        mask_type = "standard"

    is_invalid = check_mask(binary_mask, name=f"初始{mask_type}")
    if is_invalid:
        # print("尝试自动调整阈值...")

        # 计算光流大小的分位数
        flow_mag_np = flow_mag.cpu().numpy()
        q25 = np.percentile(flow_mag_np, 25)
        q50 = np.percentile(flow_mag_np, 50)
        q75 = np.percentile(flow_mag_np, 75)

        # print(f"光流分位数 - 25%: {q25:.2f}, 50%(中位数): {q50:.2f}, 75%: {q75:.2f}")

        # 根据分布建议合适的阈值
        if inverse_mask:
            if np.count_nonzero(binary_mask) == binary_mask.size:  # 全为1
                suggested_threshold = q75  # 使用更高阈值
                # print(f"建议尝试更高的阈值，例如: {suggested_threshold:.2f}")
            else:  # 全为0
                suggested_threshold = q25  # 使用更低阈值
                # print(f"建议尝试更低的阈值，例如: {suggested_threshold:.2f}")
        else:
            if np.count_nonzero(binary_mask) == binary_mask.size:  # 全为1
                suggested_threshold = q25  # 使用更低阈值
                # print(f"建议尝试更低的阈值，例如: {suggested_threshold:.2f}")
            else:  # 全为0
                suggested_threshold = q75  # 使用更高阈值
                # print(f"建议尝试更高的阈值，例如: {suggested_threshold:.2f}")

        if inverse_mask:
            binary_mask = (flow_mag <= suggested_threshold).float().cpu().numpy()  # 反向选择
            mask_type = "inverse"
        else:
            binary_mask = (flow_mag > suggested_threshold).float().cpu().numpy()  # 正向选择（原始方式）
            mask_type = "standard"

    # visualize_mask(binary_mask, output_path="./mask_debug", prefix=f"{mask_type}_threshold{flow_threshold}_")
    # 应用形态学操作使掩码更平滑
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=2)
    binary_mask = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)

    # 第1步: 亮度对齐函数
    def rgb_to_hsi(rgb_img):
        """将RGB图像转换为HSI空间"""
        # 归一化到[0,1]
        if rgb_img.dtype == np.uint8:
            rgb_img = rgb_img.astype(np.float32) / 255.0

        # 提取RGB通道
        r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]

        # 计算强度I
        intensity = (r + g + b) / 3.0

        # 避免除零错误
        epsilon = 1e-8
        min_rgb = np.minimum(np.minimum(r, g), b)

        # 计算饱和度S
        saturation = 1.0 - (3.0 / (r + g + b + epsilon)) * min_rgb

        # 计算色相H
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g)**2 + (r - b)*(g - b)) + epsilon
        theta = np.arccos(np.clip(num / den, -1.0, 1.0))

        hue = theta.copy()
        # 如果b>g，则H=360-H
        hue[b > g] = 2*np.pi - theta[b > g]

        # 归一化色相到[0,1]
        hue /= (2*np.pi)

        # 构建HSI图像
        hsi_img = np.stack([hue, saturation, intensity], axis=2)

        return hsi_img

    def hsi_to_rgb(hsi_img):
        """将HSI图像转换回RGB空间"""
        # 提取HSI通道
        h, s, i = hsi_img[:, :, 0], hsi_img[:, :, 1], hsi_img[:, :, 2]

        # 将色相H缩放回[0,2π]
        h = h * 2 * np.pi

        # RGB转换
        r = np.zeros_like(i)
        g = np.zeros_like(i)
        b = np.zeros_like(i)

        # RG区域 (0 <= H < 2π/3)
        idx = (h < 2*np.pi/3)
        b[idx] = i[idx] * (1 - s[idx])
        r[idx] = i[idx] * (1 + s[idx] * np.cos(h[idx]) / np.cos(np.pi/3 - h[idx]))
        g[idx] = 3*i[idx] - (r[idx] + b[idx])

        # GB区域 (2π/3 <= H < 4π/3)
        idx = (h >= 2*np.pi/3) & (h < 4*np.pi/3)
        h_tmp = h[idx] - 2*np.pi/3
        r[idx] = i[idx] * (1 - s[idx])
        g[idx] = i[idx] * (1 + s[idx] * np.cos(h_tmp) / np.cos(np.pi/3 - h_tmp))
        b[idx] = 3*i[idx] - (r[idx] + g[idx])

        # BR区域 (4π/3 <= H < 2π)
        idx = (h >= 4*np.pi/3)
        h_tmp = h[idx] - 4*np.pi/3
        g[idx] = i[idx] * (1 - s[idx])
        b[idx] = i[idx] * (1 + s[idx] * np.cos(h_tmp) / np.cos(np.pi/3 - h_tmp))
        r[idx] = 3*i[idx] - (g[idx] + b[idx])

        # 构建RGB图像并裁剪到[0,1]范围
        rgb_img = np.stack([r, g, b], axis=2)
        rgb_img = np.clip(rgb_img, 0.0, 1.0)

        return rgb_img

    # 第2步: 创建平滑掩码
    def smooth_mask_edges_enhanced(mask, edge_smoothness=15, dilate_size=5, dilate_iter=2):
        """增强版掩码边缘平滑"""
        # 扩张掩码，创建更宽的过渡区
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilate_iter)

        # 应用高斯模糊
        smooth_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32),
                                      (edge_smoothness, edge_smoothness), 0)

        # 创建S形曲线过渡
        alpha = 2.0
        smooth_mask = 1.0 / (1.0 + np.exp(-alpha * (smooth_mask - 0.5)))

        # 确保原始掩码内部区域保持为1
        eroded_core = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        smooth_mask = np.maximum(eroded_core, smooth_mask)

        # 确保值范围在[0,1]之间
        smooth_mask = np.clip(smooth_mask, 0.0, 1.0)

        return smooth_mask

    # 第3步: 亮度对齐实现
    # 确保图像是浮点数
    img1_float = img1_np.astype(np.float32) / 255.0 if img1_np.dtype == np.uint8 else img1_np.copy()
    img2_float = img2_np.astype(np.float32) / 255.0 if img2_np.dtype == np.uint8 else img2_np.copy()

    # 转换为HSI空间
    img1_hsi = rgb_to_hsi(img1_float)
    img2_hsi = rgb_to_hsi(img2_float)

    # 提取亮度通道
    intensity1 = img1_hsi[:, :, 2]
    intensity2 = img2_hsi[:, :, 2]

    # 创建平滑掩码
    smooth_mask = smooth_mask_edges_enhanced(
        binary_mask,
        edge_smoothness,
        dilate_size,
        dilate_iter
    )

    # 创建掩码的布尔版本（用于计算统计信息）
    mask_bool = binary_mask > 0.5

    # 确保掩码内有像素
    if np.sum(mask_bool) > 0:
        # 计算掩码区域内的平均亮度
        mean_intensity1 = np.mean(intensity1[mask_bool])
        mean_intensity2 = np.mean(intensity2[mask_bool])

        # 计算亮度调整因子
        adjustment_factor = mean_intensity1 / (mean_intensity2 + 1e-6)

        # 限制调整幅度，防止过度调整
        adjustment_factor = np.clip(adjustment_factor, 0.7, 1.5)

        # 自适应调整策略
        if abs(mean_intensity1/mean_intensity2 - 1.0) > 0.2:  # 亮度差异超过20%
            # 创建核心区域和渐变区域
            core_mask = cv2.erode(binary_mask.astype(np.uint8),
                                np.ones((3, 3), np.uint8), iterations=2)
            grad_mask = smooth_mask.copy()
            grad_mask[core_mask > 0] = 1.0  # 核心区域使用完全调整

            # 渐进调整
            adjusted_intensity = intensity2 * (1.0 + (adjustment_factor - 1.0) * grad_mask)
        else:
            # 普通调整
            adjusted_intensity = intensity2 * adjustment_factor

        # 使用平滑掩码混合原始和调整后的亮度
        final_intensity = intensity2 * (1 - smooth_mask) + adjusted_intensity * smooth_mask

        # 更新HSI图像的亮度通道
        img2_hsi_adjusted = img2_hsi.copy()
        img2_hsi_adjusted[:, :, 2] = final_intensity

        # 转换回RGB
        img2_adjusted_float = hsi_to_rgb(img2_hsi_adjusted)

        # 第4步: 局部边缘平滑处理
        def smooth_edges_locally(img_hsi, mask, window_size=15):
            # 找到掩码边缘
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
            mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
            edge_mask = mask_dilated - mask_eroded

            # 扩大边缘区域
            edge_mask = cv2.dilate(edge_mask, kernel, iterations=window_size//3)

            # 提取亮度通道
            intensity = img_hsi[:, :, 2].copy()

            # 应用高斯模糊到边缘区域
            blurred_intensity = cv2.GaussianBlur(intensity, (window_size, window_size), 0)

            # 仅在边缘区域应用模糊结果
            intensity[edge_mask > 0] = blurred_intensity[edge_mask > 0]

            # 更新HSI的亮度通道
            result_hsi = img_hsi.copy()
            result_hsi[:, :, 2] = intensity

            return result_hsi

        # 对边缘区域进行局部平滑
        img2_hsi_smooth = smooth_edges_locally(img2_hsi_adjusted, binary_mask, edge_window)
        img2_smooth_float = hsi_to_rgb(img2_hsi_smooth)

        # 转换回原始格式
        if img2_np.dtype == np.uint8:
            return (img2_smooth_float * 255).astype(np.uint8)
        else:
            return img2_smooth_float
    else:
        # 掩码区域为空，返回原始图像
        return img2_np