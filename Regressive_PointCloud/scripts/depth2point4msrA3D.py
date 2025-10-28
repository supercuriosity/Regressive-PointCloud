import struct
import numpy as np
import os
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', default='Depth', help='Depth directory for input [default: Depth]')
parser.add_argument('--output-dir', default='processed_data', help='Output processed data directory [default: processed_data]')
parser.add_argument('--num-cpu', type=int, default=8, help='Number of CPUs to use in parallel [default: 8]')
FLAGS = parser.parse_args()
# FLAGS相当于其他文件中的args

input_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
num_cpu = FLAGS.num_cpu

def read_bin(filename):
    """
    读取深度图二进制文件（.bin），解析出帧数、图像宽高和每帧的深度数据
    .bin文件结构（固定格式）：
    1. 前4字节：无符号长整型（<L）→ 视频总帧数（num_frames）
    2. 接下来4字节：无符号长整型（<L）→ 图像宽度（width）
    3. 接下来4字节：无符号长整型（<L）→ 图像高度（height）
    4. 剩余字节：无符号整型数组 → 所有帧的深度数据（按帧顺序存储）
    
    Args:
        filename (str): 单个.bin文件的完整路径（如Depth/S001C001.bin）
    
    Returns:
        np.ndarray: 深度数据数组，shape=[num_frames, height, width]
            - num_frames：视频总帧数
            - height：图像高度（像素）
            - width：图像宽度（像素）
            - 数组元素：对应像素的深度值（单位通常为毫米，取决于采集设备）
    """
    f = open(filename, 'rb')

    num_frames = f.read(4)
    num_frames = struct.unpack("<L", num_frames)[0]

    width = f.read(4)
    width = struct.unpack("<L", width)[0]

    height = f.read(4)
    height = struct.unpack("<L", height)[0]

    depth = f.read() #全部读出来
    depth = struct.unpack('{}I'.format(num_frames*height*width), depth)
   
    # 转换为numpy数组并重塑维度：[总帧数, 高度, 宽度]
    depth = np.array(depth)
    depth = np.reshape(depth, [num_frames, height, width])
    return depth

def process_one_file(filename):
    # 处理单个.bin文件：将深度图序列转换为点云序列，保存为压缩.npz文件
    depth = read_bin(filename)

    focal = 280 # 焦距，单位：像素

    xx, yy = np.meshgrid(np.arange(depth.shape[2]), np.arange(depth.shape[1]))

    point_clouds = []
    for d in range(depth.shape[0]): #这里的操作与ntu120相同，都是把深度图转为点云
        depth_map = depth[d]

        depth_min = depth_map[depth_map > 0].min() #无用的
        depth_map = depth_map

        x = xx[depth_map > 0]
        y = yy[depth_map > 0]
        z = depth_map[depth_map > 0]

        x = (x - depth_map.shape[1] / 2) / focal * z
        y = (y - depth_map.shape[0] / 2) / focal * z

        points = np.stack([x, y, z], axis=-1)

        point_clouds.append(points)

    output_filename = os.path.join(output_dir, os.path.basename(filename).split('.')[0] + '.npz')
    np.savez_compressed(output_filename, point_clouds=point_clouds)
    print(f"Processed: {os.path.basename(filename)} → Saved to {output_filename}")
if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.system('mkdir -p {}'.format(output_dir))

    files = os.listdir(input_dir)

    # 过滤非.bin文件（可选：避免处理无关文件，如.txt/.log）
    files = [f for f in files if f.endswith('.bin')]
    pool = multiprocessing.Pool(num_cpu) # 用多线程

    for input_file in files:
        print(input_file)
        pool.apply_async(process_one_file, (os.path.join(input_dir, input_file), ))

    pool.close()
    pool.join()

print(f"\nAll files processed! Output saved to: {output_dir}") 