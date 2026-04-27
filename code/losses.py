import numpy as np

class Loss:
    """
    损失函数基类，定义损失函数的接口
    """
    def __init__(self):
        pass
    
    def forward(self, y_pred, y_true):
        """
        计算损失值
        
        Args:
            y_pred: 模型预测值，形状为(batch_size, num_classes)
            y_true: 真实标签，形状为(batch_size, num_classes)的one-hot编码或(batch_size,)的类别索引
            
        Returns:
            损失值
        """
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        """
        计算损失函数关于预测值的梯度
        
        Args:
            y_pred: 模型预测值，形状为(batch_size, num_classes)
            y_true: 真实标签，形状为(batch_size, num_classes)的one-hot编码或(batch_size,)的类别索引
            
        Returns:
            损失函数关于预测值的梯度，形状与y_pred相同
        """
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """
    交叉熵损失函数，通常与Softmax激活函数一起使用
    L = -sum(y_true * log(y_pred))
    """
    def __init__(self, epsilon=1e-15):
        super().__init__()
        self.epsilon = epsilon  # 防止log(0)的数值稳定性参数
    
    def forward(self, y_pred, y_true):
        """
        计算交叉熵损失
        
        Args:
            y_pred: 模型预测值，形状为(batch_size, num_classes)，通常是softmax的输出
            y_true: 真实标签，可以是形状为(batch_size, num_classes)的one-hot编码或(batch_size,)的类别索引
            
        Returns:
            交叉熵损失值，标量
        """
        # 确保y_pred的数值稳定性
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # 如果y_true是类别索引，转换为one-hot编码
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            batch_size = y_true.shape[0]
            num_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((batch_size, num_classes))
            y_true_one_hot[np.arange(batch_size), y_true.reshape(-1)] = 1
            y_true = y_true_one_hot
        
        # 计算交叉熵损失
        batch_size = y_pred.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        
        return loss
    
    def backward(self, y_pred, y_true):
        """
        计算交叉熵损失关于预测值的梯度
        当与softmax组合时，梯度简化为: dL/dy_pred = y_pred - y_true
        
        Args:
            y_pred: 模型预测值，形状为(batch_size, num_classes)，通常是softmax的输出
            y_true: 真实标签，可以是形状为(batch_size, num_classes)的one-hot编码或(batch_size,)的类别索引
            
        Returns:
            交叉熵损失关于预测值的梯度，形状与y_pred相同
        """
        # 确保y_pred的数值稳定性
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # 如果y_true是类别索引，转换为one-hot编码
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            batch_size = y_true.shape[0]
            num_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((batch_size, num_classes))
            y_true_one_hot[np.arange(batch_size), y_true.reshape(-1)] = 1
            y_true = y_true_one_hot
        
        # 计算梯度：当与softmax组合时，梯度为y_pred - y_true
        batch_size = y_pred.shape[0]
        grad = (y_pred - y_true) / batch_size
        
        return grad


class MSELoss(Loss):
    """
    均方误差损失函数
    L = mean((y_pred - y_true)^2)
    """
    def forward(self, y_pred, y_true):
        """
        计算均方误差损失
        
        Args:
            y_pred: 模型预测值
            y_true: 真实标签
            
        Returns:
            均方误差损失值，标量
        """
        return np.mean(np.square(y_pred - y_true))
    
    def backward(self, y_pred, y_true):
        """
        计算均方误差损失关于预测值的梯度
        
        Args:
            y_pred: 模型预测值
            y_true: 真实标签
            
        Returns:
            均方误差损失关于预测值的梯度，形状与y_pred相同
        """
        batch_size = y_pred.shape[0]
        return 2 * (y_pred - y_true) / batch_size


def get_loss(loss_name):
    """
    根据名称获取损失函数实例
    
    Args:
        loss_name: 损失函数名称，可选值：'cross_entropy', 'mse'
        
    Returns:
        损失函数实例
    """
    loss_map = {
        'cross_entropy': CrossEntropyLoss(),
        'mse': MSELoss()
    }
    
    if loss_name.lower() not in loss_map:
        raise ValueError(f"不支持的损失函数: {loss_name}，可选值: {list(loss_map.keys())}")
    
    return loss_map[loss_name.lower()]