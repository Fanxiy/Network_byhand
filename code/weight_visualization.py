import numpy as np
import matplotlib.pyplot as plt
import os
from config import GLOBAL_CONFIG


def infer_model_structure(model_path):
    """
    从权重文件中推断模型结构
    
    Args:
        model_path: 模型权重文件路径
    
    Returns:
        模型结构参数字典
    """
    # 加载权重文件
    data = np.load(model_path)
    
    # 从权重形状推断模型结构
    W1_shape = data['W1'].shape
    W2_shape = data['W2'].shape
    W3_shape = data['W3'].shape
    
    # 推断各层大小
    input_size = W1_shape[0]
    hidden_size1 = W1_shape[1]
    hidden_size2 = W2_shape[1]
    output_size = W3_shape[1]
    
    print(f"从权重文件推断模型结构：")
    print(f"  输入层大小: {input_size}")
    print(f"  隐藏层1大小: {hidden_size1}")
    print(f"  隐藏层2大小: {hidden_size2}")
    print(f"  输出层大小: {output_size}")
    
    return {
        'input_size': input_size,
        'hidden_size1': hidden_size1,
        'hidden_size2': hidden_size2,
        'output_size': output_size
    }


class WeightVisualizer:
    """
    权重可视化与空间模式观察类
    """
    def __init__(self, model):
        """
        初始化权重可视化器
        
        Args:
            model: 训练好的模型
        """
        self.model = model
    
    def visualize_first_layer_weights(self, save_dir=GLOBAL_CONFIG['path']['results_dir']):
        """
        可视化第一层隐藏层的权重
        
        Args:
            save_dir: 保存可视化结果的目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取第一层权重
        weights = self.model.layer1.params['W1']
        input_size = weights.shape[0]  # 输入特征数
        hidden_size = weights.shape[1]  # 隐藏层神经元数
        
        # 从配置中获取图像尺寸
        img_height, img_width = GLOBAL_CONFIG['data']['img_size']
        channels = 3
        
        # 检查权重形状是否正确
        expected_input_size = img_height * img_width * channels
        if input_size != expected_input_size:
            print(f"警告：输入大小不匹配。期望 {expected_input_size}，实际 {input_size}")
            return
        
        # 可视化前20个神经元的权重
        num_visualize = min(20, hidden_size)
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(num_visualize):
            # 获取第i个神经元的权重
            weight = weights[:, i]
            # 恢复成图像尺寸 (64, 64, 3)
            weight_img = weight.reshape(img_height, img_width, channels)
            
            # 归一化权重到 [0, 1] 范围以便可视化
            weight_min = weight_img.min()
            weight_max = weight_img.max()
            if weight_max - weight_min > 1e-8:
                weight_img = (weight_img - weight_min) / (weight_max - weight_min)
            else:
                weight_img = np.zeros_like(weight_img)
            
            # 显示权重图像
            axes[i].imshow(weight_img)
            axes[i].set_title(f'Neuron {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'first_layer_weights.png')
        plt.savefig(save_path)
        print(f"权重可视化结果已保存到：{save_path}")
        plt.show()
    
    def analyze_weight_patterns(self):
        """
        分析权重中的空间模式和颜色倾向
        """
        # 获取第一层权重
        weights = self.model.layer1.params['W1']
        input_size = weights.shape[0]
        hidden_size = weights.shape[1]
        
        # 从配置中获取图像尺寸
        img_height, img_width = GLOBAL_CONFIG['data']['img_size']
        channels = 3
        
        # 检查权重形状
        expected_input_size = img_height * img_width * channels
        if input_size != expected_input_size:
            print(f"警告：输入大小不匹配。期望 {expected_input_size}，实际 {input_size}")
            return
        
        print("\n权重模式分析：")
        print(f"第一层隐藏层神经元数量：{hidden_size}")
        print(f"每个神经元的权重形状：({img_height}, {img_width}, {channels})")
        
        # 分析前10个神经元的权重特征
        num_analyze = min(10, hidden_size)
        for i in range(num_analyze):
            weight = weights[:, i]
            weight_img = weight.reshape(img_height, img_width, channels)
            
            # 计算各通道的平均值和标准差
            channel_means = np.mean(weight_img, axis=(0, 1))
            channel_stds = np.std(weight_img, axis=(0, 1))
            
            print(f"\n神经元 {i+1}：")
            print(f"  红色通道均值：{channel_means[0]:.4f}，标准差：{channel_stds[0]:.4f}")
            print(f"  绿色通道均值：{channel_means[1]:.4f}，标准差：{channel_stds[1]:.4f}")
            print(f"  蓝色通道均值：{channel_means[2]:.4f}，标准差：{channel_stds[2]:.4f}")
            
            # 分析空间模式
            # 计算权重的空间分布
            weight_sum = np.sum(np.abs(weight_img), axis=2)
            weight_sum_norm = (weight_sum - weight_sum.min()) / (weight_sum.max() - weight_sum.min())
            
            # 找到权重绝对值最大的位置
            max_pos = np.unravel_index(np.argmax(weight_sum), weight_sum.shape)
            print(f"  权重绝对值最大的位置：({max_pos[0]}, {max_pos[1]})")

if __name__ == '__main__':
    # 加载训练好的模型
    from model import ThreeLayerNet
    
    # 加载训练好的权重
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, 'models/best_model.npz')
    if os.path.exists(model_path):
        # 从权重文件推断模型结构
        model_structure = infer_model_structure(model_path)
        
        # 根据推断的结构创建模型实例
        model = ThreeLayerNet(
            input_size=model_structure['input_size'],
            hidden_size1=model_structure['hidden_size1'],
            hidden_size2=model_structure['hidden_size2'],
            output_size=model_structure['output_size']
        )
        
        # 加载权重
        model.load_model(model_path)
        print(f"成功加载模型权重：{model_path}")
        
        # 创建权重可视化器
        visualizer = WeightVisualizer(model)
        
        # 可视化第一层权重
        visualizer.visualize_first_layer_weights()
        
        # 分析权重模式
        visualizer.analyze_weight_patterns()
    else:
        print(f"模型权重文件不存在：{model_path}")
        print("请先训练模型并保存权重")
