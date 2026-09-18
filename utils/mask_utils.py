import torch
from kornia.geometry.epipolar import essential_from_Rt
from kornia.geometry.epipolar import sampson_epipolar_distance
from kornia.geometry.epipolar import fundamental_from_essential
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import os

def adaptive_thresholding(tensor, factor=2.0):
    """
    Apply adaptive thresholding to segment the object from the background.

    Args:
        tensor (torch.Tensor): The input tensor to threshold.
        factor (float): The factor to determine the threshold based on mean and std deviation.

    Returns:
        torch.Tensor: Binary mask tensor where object pixels are 1 and background pixels are 0.
    """
    # Calculate mean and standard deviation of the tensor
    mean_value = tensor.mean().item()
    std_value = tensor.std().item()

    # Set threshold as mean + factor * std
    threshold = mean_value + factor * std_value
    # print(mean_value, std_value, threshold)
    # Create a binary mask based on the threshold
    mask = tensor <= threshold
    return mask

def get_fundamental_matrix(w2c_1, w2c_2, intrinsic):
    Rt_1 = w2c_1
    R_1, t_1 = Rt_1[:3, :3], Rt_1[:3, 3]
    Rt_2 = w2c_2
    R_2, t_2 = Rt_2[:3, :3], Rt_2[:3, 3]
    intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float().cuda()
    F = fundamental_from_essential(essential_from_Rt(R_1.unsqueeze(0), t_1.reshape(1, 3, 1), \
        R_2.unsqueeze(0), t_2.reshape(1, 3, 1)), intrinsic, intrinsic)
    return F

def get_matches(flow_fw, h, w, seed=None):
        flow = flow_fw.permute(1,2,0).cuda()
        # flow = self.get_flow(timestep_1, timestep_2).squeeze(0).permute(1,2,0)
        h, w = h, w
        x = torch.arange(w, dtype=torch.long).cuda()
        y = torch.arange(h, dtype=torch.long).cuda()
        yy, xx = torch.meshgrid(y, x)
        pts_source = torch.stack([xx, yy], dim=-1)

        # Yield Correspondence
        xx_source, yy_source = torch.split(pts_source, 1, dim=-1)
        xx_target, yy_target = torch.split(pts_source + flow, 1, dim=-1)
        xx_source, yy_source, xx_target, yy_target = xx_source.squeeze(), yy_source.squeeze(), xx_target.squeeze(), yy_target.squeeze()

        # Ignore Regeion
        # if valid_regeion is None:
        valid_regeion = torch.ones((h,w)).cuda()
        valid_regeion = valid_regeion * (xx_target > 0) * (xx_target < w) * (yy_target > 0) * (yy_target < h) #* (depth > 0)
        valid_regeion = valid_regeion.bool()
        # Seeding for reproduction
        # if seed is not None: np.random.seed(seed)

        # pts_idx = np.random.randint(0, torch.sum(valid_regeion).item(), self.match_num)

        # Sample Correspondence
        # xxf_source, yyf_source = xx_source[valid_regeion][pts_idx], yy_source[valid_regeion][pts_idx]
        # xxf_target, yyf_target = xx_target[valid_regeion][pts_idx], yy_target[valid_regeion][pts_idx]
        pts_source, pts_target = torch.stack([xx_source, yy_source], axis=0).float(), torch.stack([xx_target, yy_target], axis=0).float()
        matches = torch.cat([pts_source, pts_target], dim=0).unsqueeze(0).view([1, 4, -1])
        valid_regeion = valid_regeion.view(1, 1, -1)
        return matches, valid_regeion

def compute_epipolar_loss(flow_fw, w2c_1, w2c_2, intrinsic, h, w):
    # fmat: [b, 3, 3] match: [b, 4, h*w] mask: [b,1,h*w]
    match, mask = get_matches(flow_fw,h,w)
    fmat = get_fundamental_matrix(w2c_1, w2c_2, intrinsic)
    num_batch = match.shape[0]
    match_num = match.shape[-1]
    h, w = h,w
    
    points1 = match[:,:2,:].permute(0, 2, 1)
    points2 = match[:,2:,:].permute(0, 2, 1)
    
    samp_dis = sampson_epipolar_distance(points1, points2, fmat)
    img1_rigid_mask = (samp_dis.view([h, w])).float()
    loss = samp_dis.mean()
    return loss, img1_rigid_mask


def warp_image_with_flow(img, flow):
    """
    使用光流将图像扭曲到目标位姿
    
    Args:
        img: 输入图像张量 [B,C,H,W] 或 [C,H,W]
        flow: 光流张量 [B,2,H,W] 或 [2,H,W]
        
    Returns:
        扭曲后的图像张量，与输入形状相同
    """
    # 确保输入是批次形式
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    if len(flow.shape) == 3:
        flow = flow.unsqueeze(0)  # [2,H,W] -> [1,2,H,W]
        
    B, C, H, W = img.shape
    
    # 创建网格坐标
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W).float().to(img.device)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W).float().to(img.device)
    
    # 堆叠网格坐标
    grid = torch.cat([xx, yy], 1)  # [B,2,H,W]
    
    # 增加光流得到目标位置
    vgrid = grid + flow
    
    # 归一化到[-1, 1]范围用于grid_sample
    vgrid_x = 2.0 * vgrid[:, 0, :, :] / (W - 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1, :, :] / (H - 1) - 1.0
    vgrid_scaled = torch.stack([vgrid_x, vgrid_y], dim=3)
    
    # 使用grid_sample执行扭曲操作
    output = F.grid_sample(img, vgrid_scaled, mode='bilinear', 
                          padding_mode='zeros', align_corners=True)
    
    return output


def generate_intensity_diff_mask(img1, img2, flow, intensity_threshold=0.1, 
                                kernel_size=5, dilate_iter=1, visualize=False):
    """
    生成基于强度差异的掩码
    
    Args:
        img1: 第一个图像 (张量或PIL或numpy数组)
        img2: 第二个图像 (张量或PIL或numpy数组)
        flow: 从img1到img2的光流 [B,2,H,W]
        intensity_threshold: 强度差异阈值
        kernel_size: 差异区域扩展核大小
        dilate_iter: 扩张迭代次数
        visualize: 是否可视化掩码
        
    Returns:
        差异掩码 (numpy数组)
    """
    # 转换为PyTorch张量
    if not torch.is_tensor(img1):
        if isinstance(img1, Image.Image):
            img1 = np.array(img1) / 255.0
            img2 = np.array(img2) / 255.0
        
        # 确保为浮点数并缩放到[0,1]范围
        if img1.dtype == np.uint8:
            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0
            
        # 转换为pytorch张量并调整通道顺序为[C,H,W]
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
    
    # 确保形状正确
    if img1.dim() == 3:  # [C,H,W]
        img1 = img1.unsqueeze(0)  # [1,C,H,W]
        img2 = img2.unsqueeze(0)  # [1,C,H,W]
    
    # 设置设备
    device = flow.device
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # 使用光流扭曲img1
    warped_img1 = warp_image_with_flow(img1, flow)
    
    # 创建有效区域掩码(无效区域是扭曲后的零值区域)
    valid_mask = (warped_img1.sum(dim=1, keepdim=True) > 0).float()
    
    # 计算强度(亮度)
    def rgb_to_intensity(img):
        """RGB转换为强度(亮度)"""
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    
    intensity1 = rgb_to_intensity(warped_img1)  # [B,1,H,W]
    intensity2 = rgb_to_intensity(img2)         # [B,1,H,W]
    
    # 计算强度差异并应用阈值
    diff = torch.abs(intensity1 - intensity2)
    diff_mask = (diff > intensity_threshold).float() * valid_mask
    
    # 将掩码转换为numpy数组
    diff_mask_np = diff_mask[0, 0].cpu().numpy()
    
    # 扩展差异区域(使用形态学扩张)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(diff_mask_np.astype(np.uint8), kernel, iterations=dilate_iter)
    
    if visualize:
        # 可视化扭曲图像和掩码
        warped_img_np = warped_img1[0].permute(1, 2, 0).cpu().numpy()
        img2_np = img2[0].permute(1, 2, 0).cpu().numpy()
        diff_np = diff[0, 0].cpu().numpy()
        
        # 确保输出目录存在
        os.makedirs("./flow_mask_debug", exist_ok=True)
        
        # 保存扭曲后的图像
        cv2.imwrite("./flow_mask_debug/warped_img.png", cv2.cvtColor((warped_img_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite("./flow_mask_debug/target_img.png", cv2.cvtColor((img2_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # 保存差异图
        diff_viz = (diff_np / diff_np.max() * 255).astype(np.uint8)
        cv2.imwrite("./flow_mask_debug/intensity_diff.png", diff_viz)
        
        # 保存未扩展的掩码
        cv2.imwrite("./flow_mask_debug/diff_mask_raw.png", (diff_mask_np * 255).astype(np.uint8))
        
        # 保存扩展后的掩码
        cv2.imwrite("./flow_mask_debug/diff_mask_dilated.png", dilated_mask * 255)
        
        # 将掩码叠加到原图上
        overlay = img2_np.copy()
        overlay[dilated_mask > 0] = [1, 0, 0]  # 红色标记差异区域
        cv2.imwrite("./flow_mask_debug/mask_overlay.png", cv2.cvtColor((overlay*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        print("已保存可视化结果到 ./flow_mask_debug/")
    
    return dilated_mask


def flow_based_intensity_alignment(img1, img2, flow, intensity_threshold=0.1, 
                                  kernel_size=5, dilate_iter=1,
                                  blend_factor=0.8, visualize=False):
    """
    基于光流和强度差异进行图像亮度对齐
    
    Args:
        img1: 第一个图像 (张量或PIL或numpy数组)
        img2: 第二个图像 (张量或PIL或numpy数组)
        flow: 从img1到img2的光流 [B,2,H,W]
        intensity_threshold: 强度差异阈值
        kernel_size: 差异区域扩展核大小
        dilate_iter: 扩张迭代次数
        blend_factor: 混合因子，控制对齐强度
        visualize: 是否可视化中间结果
        
    Returns:
        亮度对齐后的图像
    """
    # 确定输入类型
    is_tensor = torch.is_tensor(img1)
    is_pil = isinstance(img1, Image.Image)
    original_type = None
    
    # 如果是PIL图像，转换为numpy数组
    if is_pil:
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        original_type = 'pil'
    # 如果是张量，转换为numpy数组
    elif is_tensor:
        if img1.dim() == 4:
            img1_np = img1[0].permute(1, 2, 0).cpu().numpy()
            img2_np = img2[0].permute(1, 2, 0).cpu().numpy()
        else:
            img1_np = img1.permute(1, 2, 0).cpu().numpy()
            img2_np = img2.permute(1, 2, 0).cpu().numpy()
        original_type = 'tensor'
    else:
        # 直接使用numpy数组
        img1_np = img1.copy()
        img2_np = img2.copy()
        original_type = 'numpy'
    
    # 确保为浮点数并缩放到[0,1]范围
    if img1_np.dtype == np.uint8:
        img1_np = img1_np.astype(np.float32) / 255.0
        img2_np = img2_np.astype(np.float32) / 255.0
    
    # 将numpy数组转换为张量用于扭曲操作
    img1_t = torch.from_numpy(img1_np).permute(2, 0, 1).float().unsqueeze(0).to(flow.device)
    img2_t = torch.from_numpy(img2_np).permute(2, 0, 1).float().unsqueeze(0).to(flow.device)
    
    # 使用光流扭曲img1
    warped_img1_t = warp_image_with_flow(img1_t, flow)
    
    # 将扭曲后的图像转换回numpy
    warped_img1_np = warped_img1_t[0].permute(1, 2, 0).cpu().numpy()
    
    # 生成强度差异掩码
    diff_mask = generate_intensity_diff_mask(
        img1_t, img2_t, flow, 
        intensity_threshold=intensity_threshold,
        kernel_size=kernel_size, 
        dilate_iter=dilate_iter,
        visualize=False  # 不在内部函数中可视化
    )
    
    # 创建平滑过渡的掩码
    smooth_mask = cv2.GaussianBlur(diff_mask.astype(np.float32), (kernel_size*2+1, kernel_size*2+1), 0)
    
    # 计算强度比例因子
    def get_intensity(img):
        """计算RGB图像的强度通道"""
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    
    # 只在掩码区域内计算强度
    mask_binary = diff_mask > 0.5
    if np.sum(mask_binary) > 0:
        intensity1 = get_intensity(warped_img1_np)
        intensity2 = get_intensity(img2_np)
        
        # 计算掩码区域内的平均强度
        mean_intensity1 = np.mean(intensity1[mask_binary])
        mean_intensity2 = np.mean(intensity2[mask_binary])
        
        # 计算缩放因子
        if mean_intensity2 > 0:
            intensity_scale = mean_intensity1 / mean_intensity2
            intensity_scale = np.clip(intensity_scale, 0.5, 2.0)  # 限制调整范围
        else:
            intensity_scale = 1.0
            
        # 调整img2的强度以匹配warped_img1
        adjusted_img2_np = img2_np.copy()
        
        # 转换到HSV空间进行亮度调整
        if adjusted_img2_np.max() <= 1.0:
            adjusted_img2_np_uint8 = (adjusted_img2_np * 255).astype(np.uint8)
        else:
            adjusted_img2_np_uint8 = adjusted_img2_np.astype(np.uint8)
            
        hsv = cv2.cvtColor(adjusted_img2_np_uint8, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] * intensity_scale, 0, 255)
        adjusted_img2_np = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        if img2_np.max() <= 1.0:
            adjusted_img2_np = adjusted_img2_np.astype(np.float32) / 255.0
            
        # 根据平滑掩码混合原始图像和调整后的图像
        blended_img = img2_np * (1 - smooth_mask * blend_factor) + adjusted_img2_np * (smooth_mask * blend_factor)
    else:
        # 如果掩码为空，返回原始图像
        blended_img = img2_np
    
    # 可视化结果
    if visualize:
        # 确保输出目录存在
        os.makedirs("./flow_alignment_debug", exist_ok=True)
        
        # 将浮点图像转换为uint8以便保存
        def to_uint8(img):
            if img.max() <= 1.0:
                return (img * 255).astype(np.uint8)
            else:
                return img.astype(np.uint8)
            
        warped_img_uint8 = to_uint8(warped_img1_np)
        img2_uint8 = to_uint8(img2_np)
        blended_img_uint8 = to_uint8(blended_img)
        
        # 保存图像
        cv2.imwrite("./flow_alignment_debug/warped_img1.png", 
                   cv2.cvtColor(warped_img_uint8, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./flow_alignment_debug/original_img2.png", 
                   cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./flow_alignment_debug/intensity_aligned.png", 
                   cv2.cvtColor(blended_img_uint8, cv2.COLOR_RGB2BGR))
        
        # 保存掩码
        cv2.imwrite("./flow_alignment_debug/diff_mask.png", (diff_mask * 255).astype(np.uint8))
        cv2.imwrite("./flow_alignment_debug/smooth_mask.png", (smooth_mask * 255).astype(np.uint8))
        
        print("已保存可视化结果到 ./flow_alignment_debug/")
    
    # 根据原始输入类型转换输出
    if original_type == 'pil':
        return Image.fromarray(to_uint8(blended_img))
    elif original_type == 'tensor':
        blended_tensor = torch.from_numpy(blended_img).permute(2, 0, 1).float()
        if img1.dim() == 4:
            blended_tensor = blended_tensor.unsqueeze(0)
        return blended_tensor
    else:  # numpy
        if img1.dtype == np.uint8:
            return to_uint8(blended_img)
        else:
            return blended_img


def save_masks_as_png(rigid_mask, mask, iteration=0, save_dir=None, view_idx=0):
    """
    将各种掩码保存为黑白PNG图像
    
    Args:
        rigid_mask: 刚性掩码
        mask: 强度差异掩码
        iteration: 当前迭代次数（用于文件名）
        save_dir: 保存目录，如果为None则使用默认位置
        view_idx: 视图索引（用于文件名）
    """
    import os
    import cv2
    import numpy as np
    import torch
    
    # 如果没有指定保存目录，则创建默认目录
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "mask_visualizations", 
            f"iter{iteration}"
        )
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保掩码在CPU上并且格式一致
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if isinstance(rigid_mask, np.ndarray):
        rigid_mask = torch.from_numpy(rigid_mask)
    
    # 转换为CPU NumPy数组
    mask_np = mask.detach().cpu().float().numpy()
    rigid_mask_np = rigid_mask.detach().cpu().float().numpy()
    
    # 创建各类掩码
    if mask_np.dtype == bool:
        mask_np = mask_np.astype(np.float32)
    if rigid_mask_np.dtype == bool:
        rigid_mask_np = rigid_mask_np.astype(np.float32)
    
    # 1. 刚性掩码 (rigid_mask)
    cv2.imwrite(os.path.join(save_dir, f"iter{iteration:06d}_view{view_idx:03d}_rigid_mask.png"), rigid_mask_np * 255)
    
    # 2. 强度差异掩码 (mask_bool)
    mask_bool_np = (mask_np > 0).astype(np.float32)
    cv2.imwrite(os.path.join(save_dir, f"iter{iteration:06d}_view{view_idx:03d}_intensity_mask.png"), mask_bool_np * 255)
    
    # 3. 取反的强度差异掩码 (~mask_bool)
    inverted_mask_bool_np = (~(mask_np > 0)).astype(np.float32)
    cv2.imwrite(os.path.join(save_dir, f"iter{iteration:06d}_view{view_idx:03d}_inverted_intensity_mask.png"), inverted_mask_bool_np * 255)
    
    # 4. 组合掩码 (combined_mask = ~mask_bool | rigid_mask_bool)
    rigid_mask_bool_np = (rigid_mask_np > 0).astype(np.float32)
    combined_mask_np = np.logical_and(inverted_mask_bool_np, rigid_mask_bool_np).astype(np.float32)
    cv2.imwrite(os.path.join(save_dir, f"iter{iteration:06d}_view{view_idx:03d}_combined_mask.png"), combined_mask_np * 255)
    
    print(f"已保存掩码图像到: {save_dir}")
    
    return {
        "rigid_mask": rigid_mask_np,
        "intensity_mask": mask_bool_np,
        "inverted_intensity_mask": inverted_mask_bool_np,
        "combined_mask": combined_mask_np
    }