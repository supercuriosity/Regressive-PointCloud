import os
import numpy as np
import argparse
from matplotlib.image import imread
from glob import glob
import concurrent.futures

"""
本代码是ntu60数据集专用的深度图(depth)转(to)点云文件(point4)脚本。
将深度图序列转换为点云序列，保存为压缩npz文件
输入：每个视频目录下的深度图帧（png格式）
输出：每个视频对应的点云序列文件（npz格式）
"""

parser = argparse.ArgumentParser(description='Depth to Point Cloud')
parser.add_argument('--input', default='', type=str)
parser.add_argument('--output', default='', type=str)

args = parser.parse_args()

# 深度图与点云转换的关键参数
W = 512
H = 424
focal = 280 #焦距

# xx.shape = (H, W)，每个元素是该像素的列索引（x方向）
# yy.shape = (H, W)，每个元素是该像素的行索引（y方向）
xx, yy = np.meshgrid(np.arange(W), np.arange(H)) 

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_video(video_path, output_dir):
    """
    处理单个视频目录：将深度图帧序列转换为点云序列，保存为压缩npz文件
    核心逻辑：像素坐标 → 3D点云坐标（基于透视投影公式），处理每帧并汇总保存
    
    Args:
        video_path (str): 单个视频的深度图目录路径（如input/A001_s1，内含0001.png、0002.png等帧）
        output_dir (str): 点云文件的输出根目录（最终npz文件会保存在此目录下）
    """
    video_name = video_path.split('/')[-1]
    print(f"Processing: {video_name}")
    point_clouds = []

    for img_name in sorted(os.listdir(video_path)):
        img_path = os.path.join(video_path, img_name)

        # 读取深度图：返回数组shape=[H, W]，每个元素是对应像素的深度值
        img = imread(img_path)  

        depth_min = img[img > 0].min() # 找到非零深度的最小值，此处未使用。
        depth_map = img

        # 筛选出有效深度点（depth_map > 0）
        x = xx[depth_map > 0]
        y = yy[depth_map > 0]
        z = depth_map[depth_map > 0]

        # 通过公式，计算3D点云坐标
        x = (x - W / 2) / focal * z
        y = (y - H / 2) / focal * z

        # 拼接3D坐标：每个点为[x, y, z]，shape=[N, 3]（N为当前帧有效点数）
        points = np.stack([x, y, z], axis=-1)
        point_clouds.append(points)

    output_file = os.path.join(output_dir, video_name + '.npz')
    np.savez_compressed(output_file, data=np.array(point_clouds, dtype=object))
    print(f"Finished: {video_name}")

def process_action(action):
    """
    处理单个动作类别的所有视频：批量找到该动作对应的所有视频目录，调用process_video处理
    动作类别编码：假设输入目录中视频目录名含动作编号（如A001、A002...A060，对应动作1~60）

    Args:
        action (int): 动作编号（如1→A001，10→A010，60→A060）
    """
    for video_path in sorted(glob(f'{args.input}/*A0{action:02d}')):
        process_video(video_path, args.output)
    print(f'Action {action:02d} finished!')

def main():
    mkdir(args.output)
    actions = range(1, 61)

    # 一共60个动作类别，使用多线程并行处理
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_action, actions)

if __name__ == "__main__":
    main()

# 注：此电脑中只含有ntu60数据集