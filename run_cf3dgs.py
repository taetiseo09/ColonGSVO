# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
from argparse import ArgumentParser

from trainer.cf3dgs_trainer import CFGaussianTrainer
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch
from datetime import datetime


def construct_pose(poses):
    n_trgt = poses.shape[0]
    # for i in range(n_trgt-1):
    #     poses[i+1] = poses[i+1] @ torch.inverse(poses[i])
    for i in range(n_trgt-1, 0, -1):
        poses = torch.cat(
            (poses[:i], poses[[i-1]]@poses[i:]), 0)
    return poses


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    model_cfg = lp.extract(args)
    pipe_cfg = pp.extract(args)
    optim_cfg = op.extract(args)

    if not model_cfg.source_path:
        parser.error("--source/-s is required")
    if model_cfg.end_cam and model_cfg.end_cam <= model_cfg.start_cam:
        parser.error("--end_cam must be greater than --start_cam, or 0 to use all frames")

    print("="*70)
    print("Loss Configuration:")
    print(f"  Total Loss = λ1 * L_rgb + λ2 * L_flow")
    print(f"  L_rgb = (1-λ) * L1 + λ * SSIM")
    print(f"  λ1 (lambda_rgb): {optim_cfg.lambda_rgb}")
    print(f"  λ  (lambda_dssim): {optim_cfg.lambda_dssim}")
    print(f"  λ2 (flow_loss_weight): {optim_cfg.flow_loss_weight}")
    print("="*70)

    print(f"Model path: {model_cfg.model_path}")
    print(f"Start camera: {model_cfg.start_cam}")
    print(f"End camera: {model_cfg.end_cam}")
    if model_cfg.align_brightness:
        print('Aligning brightness between two images')
    if model_cfg.use_flow_loss:
        print('Using flow loss')

    data_path = model_cfg.source_path
    trainer = CFGaussianTrainer(data_path, model_cfg, pipe_cfg, optim_cfg)
    start_time = datetime.now()
    if model_cfg.mode == "train":
        trainer.train_from_progressive()
    elif model_cfg.mode == "render":
        trainer.render_nvs(traj_opt=model_cfg.traj_opt)
    elif model_cfg.mode == "eval_nvs":
        trainer.eval_nvs()
    elif model_cfg.mode == "eval_pose":
        trainer.eval_pose()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
