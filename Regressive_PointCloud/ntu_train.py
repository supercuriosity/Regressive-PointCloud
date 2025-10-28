from __future__ import print_function
import datetime
import logging
import os
import time
import sys
import yaml
import numpy as np
import random
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import utils
from scheduler import WarmupMultiStepLR
from datasets.ntu60_sk import NTU60Subject_SK
import models.RG as Models
from ipdb import set_trace as st


def setup_logging(output_dir):
    '''
    配置日志文件，与输出文件同目录。
    '''
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "training.log")), 
            logging.StreamHandler(sys.stdout) 
        ]
    )


def train_one_epoch(model, criterion,criterion_2, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):
    """
    训练单个epoch的核心逻辑：遍历训练集、前向传播、计算损失、反向优化、更新指标
    Args:
        model (nn.Module): UST-SSM模型
        criterion (nn.Module): 主损失函数（此处为交叉熵，用于动作分类）
        criterion_2 (nn.Module): 备用损失函数（此处为MSE，暂未使用，预留扩展）
        optimizer (torch.optim.Optimizer): 优化器（此处为SGD，用于更新模型参数）
        lr_scheduler (WarmupMultiStepLR): 学习率调度器（控制每个批次的学习率变化）
        data_loader (DataLoader): 训练集数据加载器（批量输出点云+骨骼数据）
        device (torch.device): 显卡设备（如cuda:0）
        epoch (int): 当前训练轮次（用于日志显示）
        print_freq (int): 日志打印频率（每多少个批次打印一次训练指标）
    """
    model.train()

    # 添加记录的指标：损失、学习率、训练速度（clips/s）等
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch) # 日志表头（显示当前轮次）
    # 遍历训练集数据加载器，按批次训练
    for (clip, target_sk), target, _ in metric_logger.log_every(data_loader, print_freq, header):
        # 这里注意一下log_every的返回值，就是data_loader本身
        start_time = time.time()
        clip, target = clip.to(device), target.to(device)

        # 1. 前向传播：模型输入点云剪辑，输出类别预测（shape: [batch_size, num_classes]）
        output = model(clip)

        # 2. 计算损失：交叉熵损失（预测值与真实标签的差异）
        loss = criterion(output, target)
        # loss = criterion_2(output, target)

        # 3. 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. 计算分类准确率（Top-1和Top-5，用于评估训练效果）
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        
        # 5. 更新指标：损失、学习率、准确率、训练速度
        batch_size = clip.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        
        # 6. 更新学习率：按调度器调整下一批次的学习率
        lr_scheduler.step()
        sys.stdout.flush()

def evaluate(model, criterion, data_loader, device):
    """
    评估模型性能：在测试集上计算"剪辑级、视频级、类别级准确率"（对应论文评估逻辑）
    Args:
        model (nn.Module): UST-SSM模型
        criterion (nn.Module): 损失函数（计算测试集损失，仅用于监控，不参与优化）
        data_loader (DataLoader): 测试集数据加载器（批量输出点云+骨骼数据）
        device (torch.device): 计算设备（如cuda:0）
    Returns:
        float: 视频级Top-1准确率（论文中核心评估指标，用于衡量模型最终性能）
    """
    model.eval()

    #初始化指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # 字典用于累积视频级概率和标签：key=视频索引，value=概率总和/真实标签
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for (clip, target_sk), target, video_idx in metric_logger.log_every(data_loader, 100, header):
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 1. 前向传播：计算模型预测
            output = model(clip)

            # 2. 计算测试损失（仅用于监控，不影响模型参数）
            loss = criterion(output, target)

            # 3. 计算剪辑级准确率（Top-1和Top-5）
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            prob = F.softmax(input=output, dim=1)

            # 将张量转换为numpy，便于后续处理
            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            # 4. 累积视频级概率和标签：同一视频的多个剪辑概率相加
            for i in range(batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    #多卡同步，见utils.py
    metric_logger.synchronize_between_processes()

    logging.info(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}'.format(
        top1=metric_logger.acc1, top5=metric_logger.acc5))
    
    # 5. 计算视频级准确率：对每个视频的累积概率取argmax得到预测标签，与真实标签比较
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k] == video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct) #视频级总体准确率，论文核心指标

    # 6. 计算类别级准确率：每个类别的准确率，用于分析不同类别的性能
    num_classes = data_loader.dataset.num_classes
    class_count = [0] * num_classes
    class_correct = [0] * num_classes
    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v == label)
    class_acc = [c / float(s) if s > 0 else 0 for c, s in zip(class_correct, class_count)]

    logging.info(' * Video Acc@1 %f' % total_acc)
    logging.info(' * Class Acc@1 %s' % str(class_acc))

    return total_acc

def main(args):

    #args来自parse_args函数，见下文。
    if args.output_dir:
        utils.mkdir(args.output_dir)
        setup_logging(args.output_dir)  

    logging.info(args)
    logging.info("torch version: %s", torch.__version__)
    logging.info("torchvision version: %s", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')

    #加载数据集
    logging.info("Loading data")
    dataset = NTU60Subject_SK(
        root=args.data_path,
        skeleton_root=args.sk_path,
        meta=args.data_meta,
        frames_per_clip=args.clip_len,
        step_between_clips=args.clip_step,
        num_points=args.num_points,
        train=True
    )

    dataset_test = NTU60Subject_SK(
        root=args.data_path,
        skeleton_root=args.sk_path,
        meta=args.data_meta,
        frames_per_clip=args.clip_len,
        step_between_clips=args.clip_step,
        num_points=args.num_points,
        train=False 
        #注意只有这个参数用于将数据集区分为训练集和测试集，详见ntu60_sk.py
    )

    # 创建数据加载器：批量加载数据，支持多线程（提升数据读取速度）
    # shuffle : 打乱数据顺序，提升泛化性
    logging.info("Data loaded")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    logging.info("Creating model")
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  dim=args.dim, heads=args.heads,mlp_dim=args.mlp_dim, num_classes=dataset.num_classes,
                  dropout=args.dropout,depth = args.depth,hos_branches_num= args.hos_branches_num,encoder_channel =args.encoder_channel )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    #参数设置
    criterion = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 学习率调度器：WarmupMultiStepLR（预热+多阶段衰减）
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
                                     warmup_iters=warmup_iters, warmup_factor=1e-5)

    # 从断点续训加载模型
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    logging.info("Start training")
    start_time = time.time()
    acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, criterion_2,optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq)
        acc = max(acc, evaluate(model, criterion, data_loader_test, device=device))

        if args.output_dir:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            # 保存模型和优化器状态
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f'model_{epoch}.pth'))
            # 保存最新的检查点
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Training time {total_time_str}')
    logging.info(f'Best Accuracy {acc}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='RG Model Training')

    # Basic parameters
    parser.add_argument('--config', default=None, type=str, help='Path to config file')
    parser.add_argument('--data-path', default='/data2/NTU120RGBD/pointcloud/ntu60npz2048', type=str, help='dataset')
    parser.add_argument('--sk-path', default='/data1/NTU120RGB/nturgb+d_skeletons_npy', type=str, help='dataset')
    parser.add_argument('--data-meta', default='/data2/NTU120RGBD/ntu60.list', help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='RG', type=str, help='model')
    parser.add_argument('--clip-len', default=24, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--clip-step', default=2, type=int, metavar='N', help='steps between frame sampling')
    parser.add_argument('--num-points', default=2048, type=int, metavar='N', help='number of points per frame')
    parser.add_argument('--radius', default=0.10, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    parser.add_argument('--dim', default=1024, type=int, help='ssm dim')  
    parser.add_argument('--depth', default=3, type=int, help='ssm depth') 
    parser.add_argument('--heads', default=8, type=int, help='ssm head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='mlp dim')
    parser.add_argument('--hos-branches-num', default=1, type=int)
    parser.add_argument('--encoder-channel', default=75, type=float)    

    parser.add_argument('--dropout', default=0.5, type=float, help='classifier dropout')
    parser.add_argument('-b', '--batch-size', default=12, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=64, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[10, 15], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    #output
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='/data2/POINT4D/UST-SSM/output/ntu', type=str, help='path where to save')
    # resume (checkpoint)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()
    
    # 如果提供了配置文件，加载并覆盖
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown config parameter: {key}")
            setattr(args, key, value)

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

# Regressive_PointCloud