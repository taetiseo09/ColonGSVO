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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.rot_type = "6d"
        self.view_dependent = True
        # self.model_type = 'original'
        self.data_type = "tanks"
        self.depth_model_type = "dpt"
        self.mode = "train"
        self.start_cam = 0
        self.end_cam = 0
        # self.vis_mesh = False
        # self.render_nvs = False
        self.traj_opt = "bspline"
        self.use_flow_loss = False
        self.flow_model_path = ""
        self.align_brightness = False
        self.visualize_mask = False  # 是否可视化mask
        self.mask_save_dir = "mask_visualizations"  # mask可视化保存目录名
        self.flow_model_name = 'unimatch'
        self.flow_ckpt_name = 'mix'
        self.inverse_depth = False  # 是否对深度图取反
        self.inverse_depth_interval = 0  # 每k张深度图取反一张，0表示不使用间隔取反
        self.visualize_depth = False  # 是否可视化保存深度图（使用紫到黄的颜色映射）
        self.depth_vis_dir = "depth_visualizations"  # 深度可视化保存目录
        self.depth_scale = 1.0  # 深度缩放因子，对深度值进行缩放
        self.use_local_depth_scale: bool = False
        self.local_depth_scale: float = 1.0
        self.local_depth_scale_size: int = 100
        self.video_fps = 10
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        # self.mode = "color"
        self.use_gt_pcd = False
        self.use_mask = False
        self.use_ref_img = False
        self.init_mode = "rand"
        self.use_mono = True
        self.interval = 15
        self.expname = ""
        self.use_sampon = False
        self.refine = False
        self.distortion = False
        self.train_test_split = 8  # 训练集测试集划分参数（原llffhold）

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2  # λ: 控制 L1 和 SSIM 的比例，L_rgb = (1-λ)*L1 + λ*SSIM
        self.lambda_rgb = 1.0    # λ1: RGB loss 的整体权重
        self.lambda_depth = 0.0
        self.lambda_dist_2nd_loss = 0.0
        self.lambda_pc = 0.0
        self.lambda_rgb_s = 0.0
        self.depth_loss_type = "invariant"
        # self.depth_loss_type = "l1"
        self.match_method = "dense"
        self.densification_interval = 100
        self.densify_interval = 500
        self.prune_interval = 2000
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.reset_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.flow_loss_weight = 0.1  # λ2: 光流 loss 的权重

        # 光强mask相关参数
        self.intensity_threshold = 0.1  # 亮度差异阈值
        self.intensity_kernel_size = 5  # 强度差异检测的邻域大小
        self.intensity_dilate_iter = 1  # 强度mask膨胀迭代次数

        # 亮度对齐相关参数
        self.brightness_flow_threshold = 2.0  # 光流阈值
        self.brightness_edge_smoothness = 21  # 边缘平滑参数
        self.brightness_dilate_size = 7  # 膨胀核大小
        self.brightness_dilate_iter = 3  # 膨胀迭代次数
        self.brightness_edge_window = 15  # 边缘窗口大小

        # 提前停止相关参数
        self.enable_early_stop = False  # 是否启用提前停止机制

        # Pose估计相关的iteration参数
        self.init_two_view_iterations = 1000  # 初始化两帧视图的迭代次数
        self.add_view_gs_local_iterations = 500  # 添加视图时局部GS拟合的迭代次数
        self.add_view_tracking_iterations = 300  # 添加视图时tracking阶段的迭代次数
        self.eval_reconstruction_epochs = 200

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
