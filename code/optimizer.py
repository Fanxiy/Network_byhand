import numpy as np

class Optimizer:
    """
    优化器基类
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        更新参数
        
        Args:
            params: 需要更新的参数字典
            grads: 参数的梯度字典
            
        Returns:
            更新后的参数字典
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    随机梯度下降优化器
    """
    def __init__(self, learning_rate, momentum=0, weight_decay=0):
        """
        初始化SGD优化器
        
        Args:
            learning_rate: 学习率
            momentum: 动量系数，用于加速收敛
            weight_decay: L2正则化系数
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}  # 存储每个参数的速度
    
    def update(self, params, grads):
        """
        使用SGD更新参数
        
        Args:
            params: 需要更新的参数字典
            grads: 参数的梯度字典
            
        Returns:
            更新后的参数字典
        """
        # 初始化速度字典（如果尚未初始化）
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
        
        # 更新每个参数
        for key in params:
            # 添加L2正则化项的梯度
            if self.weight_decay > 0:
                grads[key] += self.weight_decay * params[key]
            
            # 更新速度（应用动量）
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            
            # 更新参数
            params[key] += self.velocity[key]
        
        return params


class LearningRateScheduler:
    """
    学习率调度器，用于在训练过程中动态调整学习率
    """
    def __init__(self, optimizer):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器实例
        """
        self.optimizer = optimizer
        self.initial_learning_rate = optimizer.learning_rate
    
    def step(self, epoch):
        """
        根据当前epoch更新学习率
        
        Args:
            epoch: 当前训练轮次
        """
        raise NotImplementedError


class StepDecay(LearningRateScheduler):
    """
    阶梯式学习率衰减
    每隔step_size个epoch，学习率乘以gamma
    """
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        """
        初始化阶梯式学习率衰减调度器
        
        Args:
            optimizer: 优化器实例
            step_size: 学习率衰减的epoch间隔
            gamma: 学习率衰减系数
        """
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self, epoch):
        """
        根据当前epoch更新学习率
        
        Args:
            epoch: 当前训练轮次
        """
        # 计算衰减次数
        decay_factor = self.gamma ** (epoch // self.step_size)
        
        # 更新优化器的学习率
        self.optimizer.learning_rate = self.initial_learning_rate * decay_factor
        
        return self.optimizer.learning_rate


class ExponentialDecay(LearningRateScheduler):
    """
    指数衰减学习率
    学习率 = 初始学习率 * gamma^epoch
    """
    def __init__(self, optimizer, gamma=0.95):
        """
        初始化指数衰减学习率调度器
        
        Args:
            optimizer: 优化器实例
            gamma: 学习率衰减系数
        """
        super().__init__(optimizer)
        self.gamma = gamma
    
    def step(self, epoch):
        """
        根据当前epoch更新学习率
        
        Args:
            epoch: 当前训练轮次
        """
        # 计算衰减因子
        decay_factor = self.gamma ** epoch
        
        # 更新优化器的学习率
        self.optimizer.learning_rate = self.initial_learning_rate * decay_factor
        
        return self.optimizer.learning_rate


class ReduceOnPlateau(LearningRateScheduler):
    """
    当验证指标停止改善时减小学习率
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, min_lr=0, threshold=1e-4):
        """
        初始化ReduceOnPlateau学习率调度器
        
        Args:
            optimizer: 优化器实例
            mode: 'min'表示监控指标越小越好，'max'表示监控指标越大越好
            factor: 学习率衰减系数
            patience: 在减小学习率之前等待的epoch数
            min_lr: 学习率下限
            threshold: 判断指标是否改善的阈值
        """
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.last_epoch = 0
    
    def step(self, metrics):
        """
        根据验证指标更新学习率
        
        Args:
            metrics: 验证指标值
        """
        self.last_epoch += 1
        
        # 判断指标是否改善
        if (self.mode == 'min' and metrics < self.best - self.threshold) or \
           (self.mode == 'max' and metrics > self.best + self.threshold):
            # 指标改善，重置计数器
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            # 指标未改善，增加计数器
            self.num_bad_epochs += 1
        
        # 如果连续多个epoch未改善，则减小学习率
        if self.num_bad_epochs >= self.patience:
            # 计算新的学习率
            new_lr = max(self.optimizer.learning_rate * self.factor, self.min_lr)
            
            # 如果学习率确实变化了，则更新
            if self.optimizer.learning_rate - new_lr > 1e-8:
                self.optimizer.learning_rate = new_lr
                self.num_bad_epochs = 0  # 重置计数器
        
        return self.optimizer.learning_rate


def get_optimizer(optimizer_name, **kwargs):
    """
    根据名称获取优化器实例
    
    Args:
        optimizer_name: 优化器名称，可选值：'sgd'
        **kwargs: 优化器的参数
        
    Returns:
        优化器实例
    """
    optimizer_map = {
        'sgd': lambda: SGD(**kwargs)
    }
    
    if optimizer_name.lower() not in optimizer_map:
        raise ValueError(f"不支持的优化器: {optimizer_name}，可选值: {list(optimizer_map.keys())}")
    
    return optimizer_map[optimizer_name.lower()]()


def get_scheduler(scheduler_name, optimizer, **kwargs):
    """
    根据名称获取学习率调度器实例
    
    Args:
        scheduler_name: 调度器名称，可选值：'step', 'exponential', 'reduce_on_plateau'
        optimizer: 优化器实例
        **kwargs: 调度器的参数
        
    Returns:
        学习率调度器实例
    """
    scheduler_map = {
        'step': lambda: StepDecay(optimizer, **kwargs),
        'exponential': lambda: ExponentialDecay(optimizer, **kwargs),
        'reduce_on_plateau': lambda: ReduceOnPlateau(optimizer, **kwargs)
    }
    
    if scheduler_name.lower() not in scheduler_map:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}，可选值: {list(scheduler_map.keys())}")
    
    return scheduler_map[scheduler_name.lower()]()