# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.custom_loader import load_lmb_dataset
from scene.UC_loader import load_csv_dataset
from natsort import natsorted
import glob

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    intrinsics: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        if isinstance(width, (list, tuple)):
            width = width[0]
        if isinstance(height, (list, tuple)):
            height = height[0]

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            intr_mat = np.array(
                [[focal_length_x, 0, width/2], [0, focal_length_x, height/2], [0, 0, 1]])
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            intr_mat = np.array(
                [[focal_length_x, 0, width/2], [0, focal_length_y, height/2], [0, 0, 1]])
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, intrinsics=intr_mat,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, train_test_split=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        # train_cam_infos = [c for idx, c in enumerate(
        #     cam_infos) if idx % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(
        #     cam_infos) if idx % llffhold == 0]
        sample_rate = 2 if "Family" in path else 8
        # sample_rate = 8
        ids = np.arange(len(cam_infos))
        i_test = ids[int(sample_rate/2)::sample_rate]
        i_train = np.array([i for i in ids if i not in i_test])
        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    pcd = None
    ply_path = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCustomSceneInfo(path, images, eval, train_test_split=8, start_cam=0, end_cam=400):

    camera_type = "L"

    intrin, extrin = load_lmb_dataset(path, camera_type)

    reading_dir = "left" if camera_type == "L" else "right"
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=extrin, cam_intrinsics=intrin, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    cam_infos = cam_infos[start_cam:end_cam]
    print(f"使用 {start_cam} 到 {end_cam} 之间 camera")
    print(f"共加载 {len(cam_infos)} 个相机")

    if eval:
        # train_cam_infos = [c for idx, c in enumerate(
        #     cam_infos) if idx % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(
        #     cam_infos) if idx % llffhold == 0]
        # sample_rate = 2 if "Family" in path else 8
        sample_rate = 8
        ids = np.arange(len(cam_infos))
        i_test = ids[int(sample_rate/2)::sample_rate]
        i_train = np.array([i for i in ids if i not in i_test])
        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     storePly(ply_path, xyz, rgb)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None
    pcd = None
    ply_path = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readUCSceneInfo(path, images, eval, train_test_split=8, start_cam=0, end_cam=400):

    num_cameras=2000

    image_width=320
    image_height=320
    image_format="image_{:04d}.png"

    # 查找内参和外参文件
    intrinsic_file = os.path.join(path, "Calibration", "cam.txt")
    extrinsic_file = os.path.join(path, "Colon", "Poses", "colon_position_rotation.csv")

    if not os.path.exists(intrinsic_file):
        raise FileNotFoundError(f"未找到相机内参文件: {intrinsic_file}")
    if not os.path.exists(extrinsic_file):
        raise FileNotFoundError(f"未找到相机外参文件: {extrinsic_file}")

    print(f"读取相机参数:")
    print(f"  内参文件: {intrinsic_file}")
    print(f"  外参文件: {extrinsic_file}")

    # 使用csv_loader加载相机参数
    intrin, extrin = load_csv_dataset(
        intrinsic_file,
        extrinsic_file,
        image_width,
        image_height,
        image_format
    )

    # 确定读取图像的目录
    # reading_dir = images if images else ""
    reading_dir = "Frames"
    images_folder = os.path.join(path, 'Colon', reading_dir)

    print(f"从 {images_folder} 读取图像...")

    # 读取相机信息
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=extrin,
        cam_intrinsics=intrin,
        images_folder=images_folder
    )
    cam_infos = natsorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    cam_infos = cam_infos[start_cam:end_cam]
    print(f"使用 {start_cam} 到 {end_cam} 之间 camera")
    print(f"共加载 {len(cam_infos)} 个相机")

    if eval:
        sample_rate = train_test_split
        ids = np.arange(len(cam_infos))

        # 修改测试样本选择逻辑：每sample_rate张图取最后一张
        i_test = []
        for i in range(0, len(ids) - sample_rate + 1, sample_rate):  # 确保至少有一个完整组
            # 选择这个组的最后一个
            end_idx = i + sample_rate - 1
            if end_idx < len(ids):  # 双重检查，确保不会越界
                i_test.append(end_idx)

        i_test = np.array(i_test)
        i_train = np.array([i for i in ids if i not in i_test])

        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
        test_names = [cam_infos[i].image_name for i in i_test]
        print(f"测试集图像名称: {test_names}")
        print(f"训练集: {len(train_cam_infos)} 个相机, 测试集: {len(test_cam_infos)} 个相机")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        print(f"训练集: {len(train_cam_infos)} 个相机")

    # 计算场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 此处不加载点云，可以根据需要修改
    pcd = None
    ply_path = None

    # 构造并返回场景信息
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )

    return scene_info

def _load_endoscope_camera_params(path):
    """Load C3VD intrinsics, optionally overridden by <scene>/camera.json."""
    camera_params = {
        "image_height": 540,
        "image_width": 675,
        "fx": 401.1595,
        "fy": 400.9425,
        "cx": 334.143,
        "cy": 273.8665,
        "png_depth_scale": 2.55,
        "crop_edge": 0,
    }
    camera_path = os.path.join(path, "camera.json")
    if os.path.isfile(camera_path):
        with open(camera_path, "r", encoding="utf-8") as camera_file:
            overrides = json.load(camera_file)
        unknown = sorted(set(overrides) - set(camera_params))
        if unknown:
            raise ValueError(
                f"Unsupported keys in {camera_path}: {', '.join(unknown)}"
            )
        camera_params.update(overrides)
        print(f"Loaded camera intrinsics from {camera_path}")
    else:
        print("Using the default C3VD camera intrinsics")
    return camera_params


def readEndoscopeSceneInfo(path, images, eval, train_test_split=8, start_cam=0, end_cam=0):
    """
    读取内窥镜数据集的场景信息（简化版本）

    Args:
        path: 数据集根目录路径
        images: 图像子目录名称（如果为None则使用默认值）
        eval: 是否进行评估（如果为True则分割训练和测试集）
        train_test_split: 每隔多少帧取一个测试图像
        start_cam: 起始相机索引
        end_cam: 结束相机索引（如果为None则使用所有相机）

    Returns:
        SceneInfo: 包含场景信息的命名元组
    """

    camera_params = _load_endoscope_camera_params(path)

    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]
    width = camera_params["image_width"]
    height = camera_params["image_height"]

    FovY = focal2fov(fy, height)
    FovX = focal2fov(fx, width)

    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    print(f"相机内参:")
    print(f"  图像尺寸: {width}x{height}")
    print(f"  焦距: fx={fx}, fy={fy}")
    print(f"  主点: cx={cx}, cy={cy}")
    print(f"  视场角: FovX={FovX:.2f}°, FovY={FovY:.2f}°")

    # 读取外参文件
    pose_path = os.path.join(path, "pose.txt")
    print(f"从 {pose_path} 读取相机外参...")

    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"未找到相机外参文件: {pose_path}")

    # 读取相机位姿
    with open(pose_path, "r") as f:
        lines = f.readlines()

    reading_dir = "images" if images is None else images
    images_folder = os.path.join(path, reading_dir)

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"未找到图像文件夹: {images_folder}")

    image_files = natsorted([f for f in os.listdir(images_folder) if f.endswith(".png") or f.endswith(".jpg")])

    if len(lines) != len(image_files):
        raise ValueError(
            f"C3VD frame/pose count mismatch: {len(image_files)} images in "
            f"{images_folder}, but {len(lines)} poses in {pose_path}"
        )
    if start_cam < 0:
        raise ValueError("start_cam must be non-negative")
    if end_cam == 0:
        end_cam = len(image_files)
    if end_cam > len(image_files):
        raise ValueError(
            f"end_cam={end_cam} exceeds the available {len(image_files)} frames"
        )

    # 限制处理的图像范围
    image_files = image_files[start_cam:end_cam]
    pose_lines = lines[start_cam:end_cam]

    print(f"从 {images_folder} 读取 {len(image_files)} 张图像...")

    # 直接创建相机信息列表
    cam_infos = []

    for i, (image_file, line) in enumerate(zip(image_files, pose_lines)):
        # 解析位姿
        pose_values = list(map(float, line.strip().split(',')))
        if len(pose_values) != 16:
            raise ValueError(
                f"Pose line {start_cam + i + 1} must contain 16 comma-separated values"
            )
        pose_matrix = np.array(pose_values, dtype=np.float32).reshape(4, 4).transpose(1, 0)

        # 从位姿矩阵中直接提取旋转矩阵和平移向量
        R = pose_matrix[:3, :3]
        T = pose_matrix[:3, 3]

        # 构建图像路径
        image_path = os.path.join(images_folder, image_file)
        image_name = os.path.basename(image_path).split(".")[0]

        # 加载图像
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            continue

        # 创建相机信息对象
        cam_info = CameraInfo(
            uid=i+1,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            intrinsics=intrinsic_matrix,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height
        )

        cam_infos.append(cam_info)

    print(f"共加载 {len(cam_infos)} 个相机")

    # 分割训练集和测试集
    # if eval:
    #     sample_rate = llffhold
    #     ids = np.arange(len(cam_infos))
    #     i_test = ids[::sample_rate]  # 每隔sample_rate取一个测试样本
    #     i_train = np.array([i for i in ids if i not in i_test])

    #     train_cam_infos = [cam_infos[i] for i in i_train]
    #     test_cam_infos = [cam_infos[i] for i in i_test]

    #     print(f"训练集: {len(train_cam_infos)} 个相机, 测试集: {len(test_cam_infos)} 个相机")
    if eval:
        sample_rate = train_test_split
        ids = np.arange(len(cam_infos))

        # 修改测试样本选择逻辑：每sample_rate张图取最后一张
        i_test = []
        for i in range(0, len(ids) - sample_rate + 1, sample_rate):  # 确保至少有一个完整组
            # 选择这个组的最后一个
            end_idx = i + sample_rate - 1
            if end_idx < len(ids):  # 双重检查，确保不会越界
                i_test.append(end_idx)

        i_test = np.array(i_test)
        i_train = np.array([i for i in ids if i not in i_test])

        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
        test_names = [cam_infos[i].image_name for i in i_test]
        print(f"测试集图像名称: {test_names}")
        print(f"训练集: {len(train_cam_infos)} 个相机, 测试集: {len(test_cam_infos)} 个相机")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        print(f"训练集: {len(train_cam_infos)} 个相机")

    # 计算场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 不加载点云
    pcd = None
    ply_path = None

    # 构造并返回场景信息
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )

    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # R is stored transposed due to 'glm' in CUDA code
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readScaredSceneInfo(path, images, eval, train_test_split=8, start_cam=0, end_cam=-1):
    """
    读取SCARED数据集的场景信息
    数据集结构:
    - input/: RGB图像
    - flow/: 光流数据 (flow_fw_*.npz, flow_bw_*.npz)
    - monodep/: 单目深度 (depth_*.npz)
    - poses/: GT位姿 (JSON格式)

    Args:
        path: 数据集根目录路径
        images: 图像子目录名称（通常为'input'）
        eval: 是否进行评估
        train_test_split: 测试集采样率
        start_cam: 起始帧索引
        end_cam: 结束帧索引（-1表示使用所有帧）
    """

    # 读取RGB图像路径
    input_dir = 'input' if images is None else images
    rgb_paths = glob.glob(os.path.join(path, input_dir, '*.png')) + \
                glob.glob(os.path.join(path, input_dir, '*.jpeg')) + \
                glob.glob(os.path.join(path, input_dir, '*.jpg'))

    if len(rgb_paths) == 0:
        raise FileNotFoundError(f"在 {os.path.join(path, input_dir)} 中未找到图像文件")

    rgb_paths = sorted(rgb_paths)

    # 获取图像尺寸
    first_image = Image.open(rgb_paths[0])
    W, H = first_image.size
    print(f"图像尺寸: {W}x{H}")

    # 限制帧范围
    if end_cam == -1:
        num_cams = len(rgb_paths)
    else:
        print(f"从第 {start_cam} 帧到第 {end_cam} 帧")
        rgb_paths = rgb_paths[start_cam:end_cam]
        num_cams = len(rgb_paths)
        print(f"实际使用帧数: {num_cams}")

    pose_path_dir = os.path.join(path, "poses")
    cam_infos = []

    print(f"开始加载 {num_cams} 个相机...")

    for i, rgb_path in enumerate(rgb_paths):
        rgb_name = os.path.basename(rgb_path)

        # 解析文件名: {scene_ind}_{data_ind}_frame_{img_name}.png
        try:
            parts = rgb_name.split('_')
            scene_ind = parts[0]
            data_ind = parts[1]
            img_name = parts[3].split('.')[0]
        except IndexError:
            print(f"警告: 无法解析文件名 {rgb_name}，跳过该文件")
            continue

        # 读取GT位姿
        pose_path = os.path.join(pose_path_dir, f'{scene_ind}_{data_ind}', f'frame_{img_name}.json')

        if not os.path.exists(pose_path):
            print(f"警告: 位姿文件不存在 {pose_path}，跳过该帧")
            continue

        try:
            with open(pose_path, 'r') as json_file:
                data = json.load(json_file)

            # 读取相机位姿 (4x4矩阵)
            gt_pose = np.array(data["camera-pose"], dtype=np.float32)

            # 读取相机内参
            intrinsic = np.array(data["camera-calibration"]["KL"], dtype=np.float32)

            # 调整内参以适应图像尺寸变化（如果需要）
            # SCARED原始分辨率可能是1280x1024
            original_w, original_h = 1280, 1024
            if W != original_w or H != original_h:
                intrinsic[0, :] = intrinsic[0, :] * W / original_w
                intrinsic[1, :] = intrinsic[1, :] * H / original_h

            # 从世界到相机的变换矩阵 (gt_pose通常是world-to-camera)
            # 提取旋转和平移
            R = gt_pose[:3, :3]
            T = gt_pose[:3, 3]

            # 计算视场角
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            FovX = focal2fov(fx, W)
            FovY = focal2fov(fy, H)

            # 加载图像
            image = Image.open(rgb_path)
            image_name = os.path.splitext(rgb_name)[0]

            # 创建相机信息对象
            cam_info = CameraInfo(
                uid=i,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                intrinsics=intrinsic,
                image_path=rgb_path,
                image_name=image_name,
                width=W,
                height=H
            )

            cam_infos.append(cam_info)

            if (i + 1) % 10 == 0:
                print(f"已加载 {i+1}/{num_cams} 个相机")

        except Exception as e:
            print(f"加载相机 {i} 时出错: {e}")
            continue

    print(f"成功加载 {len(cam_infos)} 个相机")

    # 分割训练集和测试集
    if eval:
        sample_rate = train_test_split
        ids = np.arange(len(cam_infos))

        # 每sample_rate张图取最后一张作为测试集
        i_test = []
        for i in range(0, len(ids) - sample_rate + 1, sample_rate):
            end_idx = i + sample_rate - 1
            if end_idx < len(ids):
                i_test.append(end_idx)

        i_test = np.array(i_test)
        i_train = np.array([i for i in ids if i not in i_test])

        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]

        test_names = [cam_infos[i].image_name for i in i_test]
        print(f"测试集图像: {test_names}")
        print(f"训练集: {len(train_cam_infos)} 个相机, 测试集: {len(test_cam_infos)} 个相机")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        print(f"训练集: {len(train_cam_infos)} 个相机")

    # 计算场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 不加载点云（SCARED数据集不提供初始点云）
    pcd = None
    ply_path = None

    # 构造场景信息
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )

    return scene_info


sceneLoadTypeCallbacks = {
    "Custom": readCustomSceneInfo,
    "UC": readUCSceneInfo,
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Endoscope": readEndoscopeSceneInfo,
    "Scared": readScaredSceneInfo
}
