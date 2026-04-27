import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image



def visualize_sample(X, y, class_names=None):
    """
    可视化样本
    
    Args:
        X: 图像数据，形状为(N, 3072)
        y: 标签，形状为(N,)
        class_names: 类别名称列表
    """
    
    # 将图像数据重塑为(N, 32, 32, 3)
    X_reshaped = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # 如果数据已经标准化，需要恢复到[0, 1]范围
    if X_reshaped.min() < 0:
        X_reshaped = (X_reshaped - X_reshaped.min()) / (X_reshaped.max() - X_reshaped.min())
    
    # 显示图像
    plt.figure(figsize=(10, 10))
    for i in range(min(25, X.shape[0])):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_reshaped[i])
        plt.title(class_names[y[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_history(history, save_path=None):
    """
    绘制训练历史
    
    Args:
        history: 包含训练和验证损失、准确率的字典
        save_path: 保存图片的路径，如果为None则只显示不保存
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"训练历史已保存到 {save_path}")
    
    plt.show()

def calculate_accuracy(y_pred, y_true):
    """
    计算分类准确率
    
    Args:
        y_pred: 预测的类别索引，形状为(N,)
        y_true: 真实的类别索引，形状为(N,)
        
    Returns:
        准确率
    """
    return np.mean(y_pred == y_true)

