import numpy as np

class Activation:
    """
    激活函数基类
    """
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            激活函数的输出
        """
        raise NotImplementedError
    
    def backward(self, x):
        """
        反向传播，计算激活函数的导数
        
        Args:
            x: 输入数据
            
        Returns:
            激活函数在输入x处的导数
        """
        raise NotImplementedError


class Sigmoid(Activation):
    """
    Sigmoid激活函数: f(x) = 1 / (1 + exp(-x))
    导数: f'(x) = f(x) * (1 - f(x))
    """
    def forward(self, x):
        # 为了数值稳定性，对x进行裁剪
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, x):
        fx = self.forward(x)
        return fx * (1 - fx)


class ReLU(Activation):
    """
    ReLU激活函数: f(x) = max(0, x)
    导数: f'(x) = 1 if x > 0 else 0
    """
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return np.where(x > 0, 1, 0)


class Tanh(Activation):
    """
    Tanh激活函数: f(x) = tanh(x)
    导数: f'(x) = 1 - tanh^2(x)
    """
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        fx = self.forward(x)
        return 1 - fx**2


class Softmax(Activation):
    """
    Softmax激活函数: f(x_i) = exp(x_i) / sum(exp(x_j))
    注意：Softmax通常与交叉熵损失一起使用，其导数会在损失函数中处理
    """
    def forward(self, x):
        # 为了数值稳定性，减去每行的最大值
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x):
        # 在实际使用中，Softmax的导数通常与交叉熵损失函数的导数结合计算
        # 这里仅作为占位符，实际不会单独使用
        raise NotImplementedError("Softmax的导数与交叉熵损失函数一起计算")


def get_activation(activation_name):
    """
    根据名称获取激活函数实例
    
    Args:
        activation_name: 激活函数名称，可选值：'sigmoid', 'relu', 'tanh', 'softmax'
        
    Returns:
        激活函数实例
    """
    activation_map = {
        'sigmoid': Sigmoid(),
        'relu': ReLU(),
        'tanh': Tanh(),
        'softmax': Softmax()
    }
    
    if activation_name.lower() not in activation_map:
        raise ValueError(f"不支持的激活函数: {activation_name}，可选值: {list(activation_map.keys())}")
    
    return activation_map[activation_name.lower()]