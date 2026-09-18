import os
import numpy as np
import collections
import re
from pathlib import Path

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def parse_float_values(line_str):
    """解析包含逗号分隔浮点数的字符串"""
    pattern = r'([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)'
    return [float(x) for x in re.findall(pattern, line_str)]

def read_lmb_intrinsic(intrinsic_path, image_width=640, image_height=480):
    """从LMB内参数据文件读取相机内参"""
    cameras = {}
    image_prefix = ""
    
    # 读取内参文件
    with open(intrinsic_path, 'r') as f:
        content = f.read()

    prefix_match = re.search(r'File Name Prefix\s*:\s*(\S+)', content)
    if prefix_match:
        image_prefix = prefix_match.group(1).strip()
        print(f"找到图片前缀: '{image_prefix}'")
    else:
        image_prefix = "image"
        print(f"未找到图片前缀，使用默认值: '{image_prefix}'")
    
    # 解析焦距
    focal_match = re.search(r'Focal Length \(mm\) : (\d+\.\d+)', content)
    sensor_width_match = re.search(r'Sensor Size \(mm\) : X=(\d+\.\d+)', content)
    sensor_height_match = re.search(r'Sensor Size \(mm\) : .*Y=(\d+\.\d+)', content)
    resolution_match = re.search(r'images Resolution \(pixel\) : (\d+)\*(\d+)', content)
    
    if resolution_match:
        image_height = int(resolution_match.group(1))
        image_width = int(resolution_match.group(2))
    
    if focal_match and sensor_width_match and sensor_height_match:
        focal_length = float(focal_match.group(1))
        sensor_width = float(sensor_width_match.group(1))
        sensor_height = float(sensor_height_match.group(1))
    
        # 计算像素单位的焦距
        fx = focal_length * image_width / sensor_width
        fy = focal_length * image_height / sensor_height
        
        # 主点通常是图像中心
        cx = image_width / 2.0
        cy = image_height / 2.0
        
        # 构造PINHOLE模型参数 [fx, fy, cx, cy]
        params = np.array([fx, fy, cx, cy])
        
        # 创建相机对象
        cameras[1] = Camera(id=1, model="PINHOLE", width=image_width, height=image_height, params=params)
    else:
        print('read camera intrinsics error')

    return cameras, image_prefix

def read_lmb_extrinsics(position_path, quaternion_path, image_prefix, camera_type="L"):
    """从LMB位置和四元数文件读取相机外参，使用ID进行匹配"""
    positions = {}
    quaternions = {}
    images = {}
    
    # 读取位置数据
    with open(position_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
                
            parts = line.split(None, 1)  # 分割ID和坐标部分
            if len(parts) >= 2:
                try:
                    image_id = int(parts[0])
                    
                    # 提取x, y, z坐标
                    coords = parse_float_values(parts[1])
                    if len(coords) >= 3:
                        tx, ty, tz = coords[:3]
                        
                        # 从左手系转为右手系 (切换y轴方向)
                        ty = -ty
                        
                        positions[image_id] = np.array([tx, ty, tz])
                except (ValueError, IndexError) as e:
                    print(f"位置解析错误，行: {line}, 错误: {e}")
    
    # 读取四元数数据
    with open(quaternion_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
                
            parts = line.split(None, 1)  # 分割ID和四元数部分
            if len(parts) >= 2:
                try:
                    image_id = int(parts[0])
                    
                    # 提取qx, qy, qz, qw
                    quat_values = parse_float_values(parts[1])
                    if len(quat_values) >= 4:
                        qx, qy, qz, qw = quat_values[:4]
                        
                        # 从左手系转为右手系 (切换y轴和w轴方向)
                        qy = -qy
                        qw = -qw
                        
                        # 重排为COLMAP的顺序 [qw, qx, qy, qz]
                        quaternions[image_id] = np.array([qw, qx, qy, qz])
                except (ValueError, IndexError) as e:
                    print(f"四元数解析错误，行: {line}, 错误: {e}")
    
    # 根据ID匹配位置和四元数数据
    common_ids = set(positions.keys()).intersection(set(quaternions.keys()))
    print(f"位置文件中的ID数量: {len(positions)}")
    print(f"四元数文件中的ID数量: {len(quaternions)}")
    print(f"共同ID数量: {len(common_ids)}")
    
    # 创建图像对象
    for image_id in common_ids:
        # 使用前缀和相机类型构建图像名称，例如: "LMB_L_00001.png"
        image_name = f"{image_prefix}_{camera_type}{image_id:05d}.png"
        images[image_id] = Image(
            id=image_id,
            qvec=quaternions[image_id],
            tvec=positions[image_id],
            camera_id=1,  # 假设只有一个相机
            name=image_name,
            xys=np.array([]),
            point3D_ids=np.array([])
        )
    
    return images

def load_lmb_dataset(dataset_path, camera_type="L",  image_width=640, image_height=480):
    """加载LMB数据集，返回兼容COLMAP格式的相机内外参"""
    dataset_path = Path(dataset_path)
    
    # 查找内参和外参文件
    intrinsic_files = list(dataset_path.glob("*Intrinsic*.txt"))
    position_files = list(dataset_path.glob(f"*Position {camera_type}*.txt"))
    quaternion_files = list(dataset_path.glob("*Quaternion*.txt"))
    
    if not intrinsic_files:
        raise FileNotFoundError(f"未找到内参文件，路径: {dataset_path}")
    if not position_files:
        raise FileNotFoundError(f"未找到相机位置文件，路径: {dataset_path}")
    if not quaternion_files:
        raise FileNotFoundError(f"未找到相机旋转文件，路径: {dataset_path}")
    
    intrinsic_file = intrinsic_files[0]
    position_file = position_files[0]
    quaternion_file = quaternion_files[0]
    
    print(f"使用文件:")
    print(f"  内参文件: {intrinsic_file.name}")
    print(f"  位置文件: {position_file.name}")
    print(f"  四元数文件: {quaternion_file.name}")
    
    # 读取相机内参和图片前缀
    cameras, image_prefix = read_lmb_intrinsic(intrinsic_file, image_width, image_height)
    # 读取相机外参
    images = read_lmb_extrinsics(position_file, quaternion_file, image_prefix, camera_type)
    
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

def main(dataset_path, output_dir, camera_type="L", image_width=640, image_height=480):
    """处理LMB数据集并转换为COLMAP格式"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    cameras, images = load_lmb_dataset(dataset_path, camera_type, image_width, image_height)
    
    # 转换为COLMAP格式文件
    convert_cameras_to_colmap_txt(cameras, os.path.join(output_dir, "cameras.txt"))
    convert_images_to_colmap_txt(images, os.path.join(output_dir, "images.txt"))
    
    print(f"已成功转换LMB数据集，输出到: {output_dir}")
    print(f"相机数量: {len(cameras)}")
    print(f"图像数量: {len(images)}")
    print(f"相机类型: {camera_type}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转换LMB数据集为COLMAP格式")
    parser.add_argument("--dataset", type=str, required=True, help="LMB数据集路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--camera", type=str, choices=["L", "R"], default="L", 
                        help="选择相机类型：L（左眼）或R（右眼），默认为L")
    parser.add_argument("--width", type=int, default=640, help="图像宽度")
    parser.add_argument("--height", type=int, default=480, help="图像高度")
    
    args = parser.parse_args()

    main(args.dataset, args.output, args.camera, args.width, args.height)