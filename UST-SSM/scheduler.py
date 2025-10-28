import torch
from bisect import bisect_right # 二分查找


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
    带热身（Warmup）的多阶段学习率调度器：
    1. 训练初期（热身阶段）：学习率从较小值逐步提升到初始学习率（避免初始lr过高导致模型不稳定）
    2. 热身结束后：按预设的milestones逐步衰减学习率（每次衰减为原来的gamma倍）
    """
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear", # 预热方法：'constant'或'linear'，此处应该可以调参
        last_epoch=-1,
    ):
        if not milestones == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        """
        计算当前迭代的学习率：
        1. 若处于热身阶段：根据warmup_method计算学习率（逐步提升）
        2. 若热身结束：根据当前迭代次数在milestones中的位置，计算衰减后的学习率
        """
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        
        decay_count = bisect_right(self.milestones, self.last_epoch)
        # 最终学习率 = 基础学习率 * 热身因子 * (gamma ^ 衰减次数)
        return [
            base_lr
            * warmup_factor
            * self.gamma ** decay_count
            for base_lr in self.base_lrs
        ]
# 作用：实现了一个带有预热阶段的多步学习率调度器，用于在训练过程中动态调整学习率，以提高模型的收敛速度和性能。