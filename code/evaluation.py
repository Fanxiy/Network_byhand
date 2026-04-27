import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import ThreeLayerNet
from utils import calculate_accuracy


class Evaluation:
    """
    模型评估类
    """
    def __init__(self):
        """
        初始化评估类
        """
        pass
    
    def test_model(self, model, X_test, y_test, batch_size=100):
        """
        在测试集上评估模型性能
        
        Args:
            model: 训练好的神经网络模型
            X_test: 测试数据
            y_test: 测试标签
            batch_size: 批量大小
            
        Returns:
            测试准确率
        """
        n_samples = X_test.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size  # 向上取整
        
        test_acc = 0.0
        
        for i in range(n_batches):
            # 获取当前批次的数据
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]
            
            # 预测
            y_pred = model.predict(X_batch)
            
            # 计算准确率
            batch_acc = calculate_accuracy(y_pred, y_batch)
            
            # 加权平均（按批次大小）
            test_acc += batch_acc * (end_idx - start_idx) / n_samples
        
        return test_acc
    
    def load_and_test(self, model_path, X_test, y_test, model_config=None):
        """
        加载训练好的模型并在测试集上评估
        
        Args:
            model_path: 模型参数文件路径
            X_test: 测试数据
            y_test: 测试标签
            model_config: 模型配置字典，包含模型结构参数
            
        Returns:
            测试准确率
        """
        # 如果未提供模型配置，使用默认配置
        if model_config is None:
            model_config = {
                'input_size': 64*64*3,  # EuroSAT: 64x64x3 = 12288
                'hidden_size1': 512,
                'hidden_size2': 256,
                'output_size': 10,  # EuroSAT: 10个类别
                'hidden_activation': 'relu',
                'output_activation': 'softmax'
            }
        
        # 创建模型
        model = ThreeLayerNet(
            input_size=model_config['input_size'],
            hidden_size1=model_config['hidden_size1'],
            hidden_size2=model_config['hidden_size2'],
            output_size=model_config['output_size'],
            hidden_activation=model_config['hidden_activation'],
            output_activation=model_config['output_activation']
        )
        
        # 加载模型参数
        model.load_model(model_path)
        
        # 在测试集上评估
        test_acc = self.test_model(model, X_test, y_test)
        
        print(f"测试集准确率: {test_acc:.4f}")
        
        return test_acc
    
    def evaluate_model_classes(self, model, X_test, y_test, class_names=None):
        """
        评估模型在各个类别上的性能
        
        Args:
            model: 训练好的神经网络模型
            X_test: 测试数据
            y_test: 测试标签
            class_names: 类别名称列表
            
        Returns:
            每个类别的准确率字典
        """
        if class_names is None:
            # 如果没有提供类别名称，使用默认的EuroSAT类别名称
            class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                           'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                           'River', 'SeaLake']
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算每个类别的准确率
        class_acc = {}
        for i, class_name in enumerate(class_names):
            # 找出属于当前类别的样本
            class_indices = (y_test == i)
            
            if np.sum(class_indices) > 0:
                # 计算当前类别的准确率
                class_acc[class_name] = np.mean(y_pred[class_indices] == y_test[class_indices])
            else:
                class_acc[class_name] = 0.0
        
        # 打印每个类别的准确率
        print("各类别准确率:")
        for class_name, acc in class_acc.items():
            print(f"  {class_name}: {acc:.4f}")
        
        return class_acc
    
    def plot_confusion_matrix(self, model, X_test, y_test, class_names=None, save_path=None):
        """
        计算并绘制混淆矩阵
        """
        if class_names is None:
            class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                        'River', 'SeaLake']
        
        y_pred = model.predict(X_test)
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        unique_class_names = [class_names[i] for i in unique_classes]
        
        plt.figure(figsize=(16, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_class_names)
        
        disp.plot(
            cmap=plt.cm.Blues,
            xticks_rotation=45,
            ax=plt.gca(),
            values_format='d',
            text_kw={'fontsize': 25}  # 这里设置矩阵内部数字大小
        )
        

        plt.xticks(fontsize=20, rotation=45, ha='right')
        plt.yticks(fontsize=23)
        plt.title('Confusion Matrix', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20, labelpad=15)
        plt.ylabel('True label', fontsize=25, labelpad=15)
        
        # 布局调整，防止底部标签被遮挡
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        else:
            plt.show()
        
        return cm


# 模块级别的函数，作为 Evaluation 类的包装器
def load_and_test(model_path, X_test, y_test, model_config=None):
    """
    加载训练好的模型并在测试集上评估
    
    Args:
        model_path: 模型参数文件路径
        X_test: 测试数据
        y_test: 测试标签
        model_config: 模型配置字典，包含模型结构参数
        
    Returns:
        测试准确率
    """
    evaluator = Evaluation()
    return evaluator.load_and_test(model_path, X_test, y_test, model_config)

def evaluate_model_classes(model, X_test, y_test, class_names=None):
    """
    评估模型在各个类别上的性能
    
    Args:
        model: 训练好的神经网络模型
        X_test: 测试数据
        y_test: 测试标签
        class_names: 类别名称列表
        
    Returns:
        每个类别的准确率字典
    """
    evaluator = Evaluation()
    return evaluator.evaluate_model_classes(model, X_test, y_test, class_names)

def plot_confusion_matrix(model, X_test, y_test, class_names=None, save_path=None):
    """
    计算并绘制混淆矩阵
    
    Args:
        model: 训练好的神经网络模型
        X_test: 测试数据
        y_test: 测试标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    evaluator = Evaluation()
    return evaluator.plot_confusion_matrix(model, X_test, y_test, class_names, save_path)

def test_model(model, X_test, y_test, batch_size=100):
    """
    在测试集上评估模型性能
    
    Args:
        model: 训练好的神经网络模型
        X_test: 测试数据
        y_test: 测试标签
        batch_size: 批量大小
        
    Returns:
        测试准确率
    """
    evaluator = Evaluation()
    return evaluator.test_model(model, X_test, y_test, batch_size)