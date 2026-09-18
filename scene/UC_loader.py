"""Load camera intrinsics and CSV poses for the optional UC dataset format."""

import os
import numpy as np
import collections
import csv
import re
from pathlib import Path

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def read_camera_intrinsic(intrinsic_path, image_width=320, image_height=320):
    """从K矩阵文件中读取相机内参
    
    Args:
        intrinsic_path: cam.txt的路径，包含K矩阵
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        构造的相机内参字典，格式与COLMAP兼容
    """
    cameras = {}
    
    try:
        # 读取K矩阵
        with open(intrinsic_path, 'r') as f:
            content = f.read()
            
            # 提取所有数值
            values = re.findall(r'[-+]?\d*\.\d+|\d+', content)
            if len(values) >= 9:
                # K矩阵的结构: fx, 0, cx, 0, fy, cy, 0, 0, 1
                fx = float(values[0])
                fy = float(values[4])
                cx = float(values[2])
                cy = float(values[5])
                
                # 构造PINHOLE模型参数 [fx, fy, cx, cy]
                params = np.array([fx, fy, cx, cy])
                
                # 创建相机对象
                cameras[1] = Camera(id=1, model="PINHOLE", width=image_width, height=image_height, params=params)
                
                print(f"相机内参加载成功: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            else:
                raise ValueError(f"无效的K矩阵格式，找到 {len(values)} 个值，需要至少9个")
    
    except Exception as e:
        print(f"读取相机内参错误: {e}")
        raise
    
    return cameras

def read_camera_extrinsics_csv(csv_path, image_format="image_{:06d}.png"):
    """从CSV文件读取相机外参
    
    Args:
        csv_path: CSV文件路径，包含相机位姿
        image_format: 图像文件名格式，用于生成图像名称
        
    Returns:
        构造的相机外参字典，格式与COLMAP兼容
    """
    images = {}
    
    try:
        with open(csv_path, 'r') as f:
            # 读取CSV文件
            csv_reader = csv.reader(f)
            
            # 读取标题行
            header = next(csv_reader)
            expected_header = ["tX", "tY", "tZ", "rX", "rY", "rZ", "rW", "time(s)"]
            
            # 检查标题是否符合预期
            if not all(h.strip() in [eh.strip() for eh in expected_header] for h in header):
                print(f"警告: CSV标题不符合预期。预期: {expected_header}, 实际: {header}")
            
            # 读取每一行数据
            for i, row in enumerate(csv_reader, 1):
                if len(row) >= 7:  # 至少需要位置和旋转四元数
                    try:
                        # 解析位置和旋转
                        tx = float(row[0])
                        ty = float(row[1])
                        tz = float(row[2])
                        rx = float(row[3])
                        ry = float(row[4])
                        rz = float(row[5])
                        rw = float(row[6])
                        
                        # 在colmap中四元数顺序是 [qw, qx, qy, qz]
                        qvec = np.array([rw, rx, ry, rz])
                        tvec = np.array([tx, ty, tz])
                        
                        # 生成图像名称
                        image_name = image_format.format(i-1)
                        
                        # 创建图像对象
                        images[i] = Image(
                            id=i,
                            qvec=qvec,
                            tvec=tvec,
                            camera_id=1,  # 假设只有一个相机
                            name=image_name,
                            xys=np.array([]),
                            point3D_ids=np.array([])
                        )
                    except ValueError as e:
                        print(f"解析第{i}行时出错: {e}")
                        print(f"行内容: {row}")
    
    except Exception as e:
        print(f"读取相机外参错误: {e}")
        raise
    
    return images

def load_csv_dataset(intrinsic_path, extrinsic_path, image_width=320, image_height=320, image_format="image_{:04d}.png"):
    """加载CSV格式的相机参数数据集
    
    Args:
        intrinsic_path: cam.txt文件路径
        extrinsic_path: colon_position_rotation.csv文件路径
        image_width: 图像宽度
        image_height: 图像高度
        image_format: 图像文件名格式
        
    Returns:
        cameras: 相机内参字典
        images: 相机外参字典
    """
    # 读取相机内参
    print(f"正在从 {intrinsic_path} 读取相机内参...")
    cameras = read_camera_intrinsic(intrinsic_path, image_width, image_height)
    
    # 读取相机外参
    print(f"正在从 {extrinsic_path} 读取相机外参...")
    images = read_camera_extrinsics_csv(extrinsic_path, image_format)
    
    return cameras, images

def convert_cameras_to_colmap_txt(cameras, output_path):
    """将相机内参转换为COLMAP cameras.txt格式"""
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        for camera_id, camera in cameras.items():
            params_str = ' '.join([str(p) for p in camera.params])
            f.write(f"{camera_id} {camera.model} {camera.width} {camera.height} {params_str}\n")

def convert_images_to_colmap_txt(images, output_path):
    """将相机外参转换为COLMAP images.txt格式"""
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for image_id, image in sorted(images.items()):
            qw, qx, qy, qz = image.qvec
            tx, ty, tz = image.tvec
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {image.camera_id} {image.name}\n")
            # 写入空的第二行（没有特征点）
            f.write("\n")

def main(intrinsic_path, extrinsic_path, output_dir, image_width=320, image_height=320, image_format="image_{:04d}.png"):
    """处理CSV格式的相机参数并转换为COLMAP格式
    
    Args:
        intrinsic_path: cam.txt文件路径
        extrinsic_path: colon_position_rotation.csv文件路径
        output_dir: 输出COLMAP文件的目录
        image_width: 图像宽度
        image_height: 图像高度
        image_format: 图像文件名格式
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    cameras, images = load_csv_dataset(intrinsic_path, extrinsic_path, image_width, image_height, image_format)
    
    # 转换为COLMAP格式文件
    convert_cameras_to_colmap_txt(cameras, os.path.join(output_dir, "cameras.txt"))
    convert_images_to_colmap_txt(images, os.path.join(output_dir, "images.txt"))
    
    print(f"已成功转换数据集，输出到: {output_dir}")
    print(f"相机数量: {len(cameras)}")
    print(f"图像数量: {len(images)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转换CSV格式的相机参数为COLMAP格式")
    parser.add_argument("--intrinsic", type=str, required=True, help="cam.txt文件路径")
    parser.add_argument("--extrinsic", type=str, required=True, help="colon_position_rotation.csv文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--width", type=int, default=320, help="图像宽度")
    parser.add_argument("--height", type=int, default=320, help="图像高度")
    parser.add_argument("--format", type=str, default="image_{:04d}.png", help="图像文件名格式")
    
    args = parser.parse_args()
    
    print(f"命令行参数:")
    print(f"  图像格式: {args.format}")
    print(f"  图像尺寸: {args.width}x{args.height}")
    
    main(args.intrinsic, args.extrinsic, args.output, args.width, args.height, args.format)
