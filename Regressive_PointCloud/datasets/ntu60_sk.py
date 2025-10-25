import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from ipdb import set_trace as st

'''
本文实现的功能：
给定root, skeleton_root, meta等参数，读取并返回处理后的NTU60数据集的点云视频数据和对应的骨骼数据
格式为：(点云剪辑, 骨骼剪辑), 标签, 视频索引。
方法为：NTU60Subject_SK类，__gititem__(idx)
'''

# 定义NTU RGB+D数据集的Cross-Subject（交叉主体）协议中的训练集受试者ID列表
# 对应论文4.1节中NTU RGB+D数据集的"cross-subject evaluation protocol"，用于区分训练/测试数据
Cross_Subject = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]


def clip_normalize(clip, epsilon=1e-6):
    """
    对点云剪辑（clip）进行归一化处理，消除尺度和位置偏差，是点云预处理的关键步骤
    步骤：1. 中心化（减去质心）；2. 尺度归一化（除以最大距离）
    
    Args:
        clip (torch.Tensor): 输入点云剪辑，形状为 [frames_per_clip, num_points, 3]（帧数×点数×3D坐标）
        epsilon (float): 极小值，避免除以零的情况
    
    Returns:
        torch.Tensor: 归一化后的点云剪辑，保持输入形状
    """
    pc = clip.view(-1, 3)  # 将剪辑展平为 [frames_per_clip×num_points, 3]
    centroid = pc.mean(dim=0)  #质心
    pc = pc - centroid # 中心化
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1))) #计算所有点到质心的最大欧氏距离，用于尺度归一化
    
    # 避免最大距离过小导致除以零或数值不稳定
    if m < epsilon:
        m = epsilon
    
    clip = (clip - centroid) / m
    return clip

class NTU60Subject_SK(Dataset):
    """
    针对NTU60数据集的自定义Dataset类，用于加载点云视频数据和对应的骨骼数据
    支持Cross-Subject协议的训练/测试划分，自动截取视频剪辑并进行预处理（采样、归一化）
    对应论文4.1节中NTU RGB+D数据集的实验数据加载逻辑
    """
    def __init__(self, root, skeleton_root, meta, frames_per_clip=24, step_between_clips=2, num_points=2048, train=True):
        super(NTU60Subject_SK, self).__init__()
        """
        初始化数据集，加载元数据并筛选符合条件的视频/骨骼文件
        
        Args:
            root (str): 点云视频数据的根目录（存储.npz格式点云文件）
            skeleton_root (str): 骨骼数据的根目录（存储.npz格式骨骼文件）
            meta (str): 元数据文件路径，每行记录"视频名 总帧数"，用于获取视频基本信息
            frames_per_clip (int): 每个剪辑（样本）包含的帧数，论文实验中常用24/36帧
            step_between_clips (int): 截取剪辑时的帧步长（减少冗余，控制样本密度）
            num_points (int): 每个帧采样的点云数量，论文实验中使用2048点
            train (bool): 是否为训练集（True=训练集，False=测试集）
        """
        self.videos = []
        self.labels = []
        self.skeleton_files = []
        self.index_map = []
        index = 0

        # 读取元数据文件，筛选符合训练/测试划分的视频
        with open(meta, 'r') as f:
            for line in f:
                # 解析元数据行，获取视频名和帧数
                name, nframes_meta = line.strip().split()
                subject = int(name[9:12])
                nframes_meta = int(nframes_meta)

                # 根据Cross-Subject协议筛选数据：
                # 训练集：受试者ID在Cross_Subject列表中；测试集：不在列表中
                if (train and subject in Cross_Subject) or (not train and subject not in Cross_Subject):
                    # 从文件名中提取动作标签
                    label = int(name[-3:]) - 1

                    skeleton_file = os.path.join(skeleton_root, name + '.npz')
                    if not os.path.exists(skeleton_file):
                        continue

                    video_file = os.path.join(root, name + '.npz')
                    if not os.path.exists(video_file):
                        continue

                    # 跳过太短的视频 (发颠)
                    min_frames = nframes_meta
                    if min_frames < frames_per_clip * step_between_clips:
                        continue

                    nframes = min_frames

                    # 从起始帧开始，每step_between_clips取一帧，共取frames_per_clip帧
                    for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                        self.index_map.append((index, t))
                        # 添加索引映射[第几个视频,第几帧]，便于__getitem__定位样本

                    self.labels.append(label)
                    self.videos.append(video_file)
                    self.skeleton_files.append(skeleton_file)
                    index += 1

        if len(self.labels) == 0:
            raise ValueError("No data found. Please check your dataset paths and meta file.")

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1 #动作类别数

    def __len__(self):
        return len(self.index_map)


    def __getitem__(self, idx):
        """
        根据样本索引idx获取单个样本（点云剪辑+骨骼剪辑+标签）
        核心逻辑：加载数据→截取剪辑→点云采样→归一化→数据增强（训练集）
        
        Args:
            idx (int): 样本索引（0~len(self)-1）
        
        Returns:
            tuple: ((point_cloud_clip, skeleton_clip), label, video_index)
            其中：
                - point_cloud_clip: 归一化后的点云剪辑，形状[frames_per_clip, num_points, 3]
                - skeleton_clip: 归一化后的骨骼剪辑，形状[frames_per_clip, ...]（骨骼数据维度根据格式而定）
                - label: 动作标签（int，0~59）
                - video_index: 该样本所属的原视频索引
        """
        index, t = self.index_map[idx]

        video_path = self.videos[index]
        skeleton_file = self.skeleton_files[index]

        video = np.load(video_path, allow_pickle=True)['data'] * 100
        # 加载点云数据（.npz文件中key为'data'），乘以100是为了单位转换（米→厘米）

        label = self.labels[index]

        clip = [video[t + i * self.step_between_clips] for i in range(self.frames_per_clip)]
        clip = [torch.tensor(p).float() for p in clip]  
        skeleton_clip = []

        try:
            with np.load(skeleton_file, allow_pickle=True) as data:
                skeleton_data = data['data']
        except Exception as e:
            print(f"Error loading skeleton file {skeleton_file}: {e}")
            raise

        # 遍历剪辑的每帧，处理点云采样和骨骼数据截取
        for i in range(len(clip)):
            frame_idx = t + i * self.step_between_clips
            p = clip[i]
            s = torch.tensor(skeleton_data[frame_idx]).float()
            # 点云采样：确保每个帧的点数为num_points（论文中固定为2048）
            if p.shape[0] > self.num_points:
                r = torch.randperm(p.shape[0])[:self.num_points]
            else:
                # 若点数不足：先重复整数次，剩余部分随机补全
                repeat, residue = divmod(self.num_points, p.shape[0])
                r = torch.cat([torch.arange(p.shape[0]).repeat(repeat), 
                                torch.randperm(p.shape[0])[:residue]])
            
            clip[i] = p[r, :]
            skeleton_clip.append(s)
        
        #调用上面的归一化函数，进行中心化和尺度归一化
        clip = clip_normalize(torch.stack(clip))
        skeleton_clip = clip_normalize(torch.stack(skeleton_clip))

        # 数据增强：仅在训练集上进行随机缩放
        if self.train:
            scales = torch.FloatTensor(3).uniform_(0.9, 1.1)
            clip = clip * scales

        #返回：点云剪辑，骨骼剪辑，标签，视频索引
        return (clip.float(), skeleton_clip.float()), label, index

