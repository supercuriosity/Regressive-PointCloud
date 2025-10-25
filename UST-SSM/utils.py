from __future__ import print_function
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist #实现多显卡分布式训练

import errno
import os

'''
这是核心工具文件，包含分布式训练相关的函数和日志记录类，实现以下功能：
1.MetricLogger类的log_every函数：按频率打印训练日志，同时计算时间和进度
2.accuracy函数：计算Top-k准确率。
3.distributed训练相关函数：初始化分布式模式、同步指标等。
'''

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    翻译： 跟踪一系列数值，并提供对窗口内平滑值或全局系列平均值的访问。
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"

        # 滑动窗口（存储最近window_size个数值）
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        #插入新数值
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        警告：不同步双端队列！
        """
        if not is_dist_avail_and_initialized(): #对单卡训练不做任何操作
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    #返回值： 中位数、平均值、全局平均值、最大值和最新值
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count != 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """
    多指标日志管理器：同时跟踪多个SmoothedValue指标（如损失、准确率、学习率）
    支持批量更新指标、按频率打印训练日志、计算剩余训练时间
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue) # 存储指标：key=指标名，value=SmoothedValue实例
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        批量更新指标（自动创建新指标的SmoothedValue实例）
        Args:
           ** kwargs: 键值对，如loss=0.5, acc1=90.0
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item() # 若为张量，转换为标量
            assert isinstance(v, (float, int))
            self.meters[k].update(v)  # 更新对应指标

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """返回所有指标的字符串表示（用于日志打印）"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        同步所有指标的全局统计（分布式训练用）
        上面做好了这个函数
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        遍历数据加载器，按频率打印训练日志，同时计算时间和进度
        Args:
            iterable: 可迭代对象（通常为数据加载器）
            print_freq (int): 日志打印频率（每print_freq个批次打印一次）
            header (str): 日志头部（如"Epoch: [0]"）
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time() # 记录整个迭代的开始时间
        end = time.time()  # 记录上一批次结束时间

         # 初始化时间跟踪器：单批次训练时间和数据加载时间
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # 根据是否有GPU，定义日志格式（GPU需显示内存占用）
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj 
            #重要的关键字yeild！函数在这一行会退出，返回obj给调用者使用，等调用者再次请求时，从这一行继续执行
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        
        # 迭代结束后，打印总耗时
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    翻译：计算指定k值的前k个预测的准确率
    Args:
        output (torch.Tensor): 模型输出，形状为[batch_size, num_classes]
        target (torch.Tensor): 真实标签，形状为[batch_size]
        topk (tuple): 需要计算的Top-k（如(1,)→Top-1，(1,5)→Top-1和Top-5）
    Returns:
        list: 对应topk的准确率（百分比）
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    """
    安全创建目录（支持多级目录），若目录已存在则不报错
    Args:
        path (str): 目录路径
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    分布式训练时，控制日志打印：仅主进程打印信息（避免多进程重复打印）
    必要时用force参数强制打印
    Args:
        is_master (bool): 当前进程是否为主进程
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    初始化分布式训练模式：解析环境变量、设置GPU设备、初始化进程组
    Args:
        args: 命令行参数（需包含rank、world_size、gpu等信息）
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
# Regressive_PointCloud