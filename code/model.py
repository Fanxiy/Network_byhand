import numpy as np
from activation import get_activation


class OneLayerNet:
    """
    单层神经网络模块
    可作为隐藏层或输出层使用
    结构：输入 -> 线性变换 -> 激活函数
    """
    def __init__(self, input_size, output_size, output_activation='softmax'):
        self.input_size = input_size
        self.output_size = output_size

        self.output_activation = get_activation(output_activation)
        self.is_softmax = (output_activation == 'softmax')

        # 初始化
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.params['b1'] = np.zeros(output_size)

        self.cache = {}

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        z1 = np.dot(X, W1) + b1
        a1 = self.output_activation.forward(z1)

        self.cache = {
            'X': X,
            'z1': z1,
            'a1': a1
        }
        return a1

    def backward(self, dout):
        """
        反向传播，计算梯度
        """
        W1 = self.params['W1']
        X = self.cache['X']
        z1 = self.cache['z1']
        batch_size = X.shape[0]

        if self.is_softmax:
            dz1 = dout
        else:
            dz1 = dout * self.output_activation.backward(z1)

        dW1 = np.dot(X.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0) / batch_size
        dx = np.dot(dz1, W1.T)

        grads = {
            'W1': dW1,
            'b1': db1
        }
        return grads, dx

    def predict(self, X):
        a1 = self.forward(X)
        return np.argmax(a1, axis=1)

    def save_model(self, file_path):
        np.savez(file_path, **self.params)

    def load_model(self, file_path):
        data = np.load(file_path)
        for key in self.params:
            if key in data:
                self.params[key] = data[key]


class ThreeLayerNet:
    """
    三层神经网络，由三个OneLayerNet堆叠而成
    结构：输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size,
                 hidden_activation='relu', output_activation='softmax'):
        self.layer1 = OneLayerNet(input_size, hidden_size1, hidden_activation)
        self.layer2 = OneLayerNet(hidden_size1, hidden_size2, hidden_activation)
        self.layer3 = OneLayerNet(hidden_size2, output_size, output_activation)
        self.cache = {}
        
    @property
    def params(self):
        """
        获取所有层的参数
        
        Returns:
            包含所有层参数的字典
        """
        return {
            'W1': self.layer1.params['W1'], 'b1': self.layer1.params['b1'],
            'W2': self.layer2.params['W1'], 'b2': self.layer2.params['b1'],
            'W3': self.layer3.params['W1'], 'b3': self.layer3.params['b1']
        }
    
    @params.setter
    def params(self, value):
        """
        设置所有层的参数
        
        Args:
            value: 包含所有层参数的字典
        """
        self.layer1.params['W1'] = value['W1']
        self.layer1.params['b1'] = value['b1']
        self.layer2.params['W1'] = value['W2']
        self.layer2.params['b1'] = value['b2']
        self.layer3.params['W1'] = value['W3']
        self.layer3.params['b1'] = value['b3']

    def forward(self, X):
        a1 = self.layer1.forward(X)
        a2 = self.layer2.forward(a1)
        a3 = self.layer3.forward(a2)

        self.cache = {
            'X': X,
            'a1': a1,
            'a2': a2,
            'a3': a3
        }
        return a3

    def backward(self, dout):
        """
        反向传播，计算每层的梯度
        """
        grads3, dout = self.layer3.backward(dout)
        grads2, dout = self.layer2.backward(dout)
        grads1, _   = self.layer1.backward(dout)

        grads = {
            'W1': grads1['W1'], 'b1': grads1['b1'],
            'W2': grads2['W1'], 'b2': grads2['b1'],
            'W3': grads3['W1'], 'b3': grads3['b1']
        }
        return grads

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(out, axis=1)

    def save_model(self, file_path):
        np.savez(file_path,
                 W1=self.layer1.params['W1'], b1=self.layer1.params['b1'],
                 W2=self.layer2.params['W1'], b2=self.layer2.params['b1'],
                 W3=self.layer3.params['W1'], b3=self.layer3.params['b1'])

    def load_model(self, file_path):
        data = np.load(file_path)
        self.layer1.params['W1'] = data['W1']
        self.layer1.params['b1'] = data['b1']
        self.layer2.params['W1'] = data['W2']
        self.layer2.params['b1'] = data['b2']
        self.layer3.params['W1'] = data['W3']
        self.layer3.params['b1'] = data['b3']