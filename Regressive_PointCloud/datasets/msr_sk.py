import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from ipdb import set_trace as st


class MSRAction3D_SK(Dataset):
    """
    针对MSR-Action3D数据集的自定义PyTorch Dataset类
    功能：加载点云视频数据和对应的3D骨骼数据，生成训练/测试用的剪辑（clip）样本
    适配任务：3D人体动作识别（论文4.2节核心实验数据集）
    数据集特点：含20个动作类别，每个样本包含点云序列和20个关节点的骨骼序列
    """
    def __init__(self, root, skeleton_root, frames_per_clip=54, step_between_clips=1, num_points=2048, train=True):
        super(MSRAction3D_SK, self).__init__()
        """
        初始化数据集，加载点云/骨骼数据并构建样本索引映射
        
        Args:
            root (str): 点云数据根目录（存储格式为.npz的点云视频文件）
            skeleton_root (str): 骨骼数据根目录（存储格式为.txt的3D骨骼文件）
            frames_per_clip (int): 每个样本剪辑包含的帧数（论文中MSR-Action3D常用16/24/36，此处默认54，需根据实验调整）
            step_between_clips (int): 生成剪辑时的帧步长（1表示连续取帧，无间隔）
            num_points (int): 每帧点云的采样点数（论文4.2节固定为2048，统一输入维度）
            train (bool): 是否为训练集（True=加载训练数据，False=加载测试数据）
        """
        self.videos = []
        self.skeletons = []  
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                skeleton_file = video_name.replace('_sdepth.npz', '_skeleton3D.txt')
                skeleton_path = os.path.join(skeleton_root, skeleton_file)
                skeleton_data = self.load_skeleton(skeleton_path)
                self.skeletons.append(skeleton_data)
                
                label = int(video_name.split('_')[0][1:]) - 1
                self.labels.append(label)

                nframes = video.shape[0]

                for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

            if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                
                skeleton_file = video_name.replace('_sdepth.npz', '_skeleton3D.txt')
                skeleton_path = os.path.join(skeleton_root, skeleton_file)
                skeleton_data = self.load_skeleton(skeleton_path)
                self.skeletons.append(skeleton_data)
                
                label = int(video_name.split('_')[0][1:]) - 1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                    self.index_map.append((index, t))  # 0,1,2...30  #4478
                index += 1
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = 20

    def load_skeleton(self, file_path):
        skeleton_data = np.loadtxt(file_path).reshape(-1, 20, 4)  # (num_frames, 20, 4)
        return skeleton_data

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        skeleton = self.skeletons[index]

        clip = [video[t + i * self.step_between_clips] for i in range(self.frames_per_clip)]
        skeleton_clip = [skeleton[t + i * self.step_between_clips] for i in range(self.frames_per_clip)]

        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        if self.train:
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip / 300   
        skeleton_clip = np.array(skeleton_clip).astype(np.float32)
        skeleton_clip = skeleton_clip[:,:,:3]

        return (clip.astype(np.float32), np.array(skeleton_clip).astype(np.float32)), label, index


#Regressive_PointCloud