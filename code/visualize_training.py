import numpy as np
import matplotlib.pyplot as plt
import os
import json
from model import ThreeLayerNet
from config import GLOBAL_CONFIG
from processor import DataProcessor
from utils import plot_training_history
from train import train_model

def load_search_results(file_path='search_results/search_results.json'):
    """
    加载超参数搜索结果
    
    Args:
        file_path: 搜索结果文件路径
        
    Returns:
        搜索结果字典
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_training_curves(save_path='results'):
    """
    生成训练曲线
    
    Args:
        save_path: 保存结果的路径
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据集
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'EuroSAT_RGB')
    processor = DataProcessor(data_dir)
    X, y = processor.load_data()
    X_train, X_test, y_train, y_test = processor.split_dataset(X, y)
    X_train, X_val, y_train, y_val = processor.split_validation(X_train, y_train)
    X_train, X_val, X_test = processor.normalize_data(X_train, X_val, X_test)
    
    # 创建配置
    config = {
        'input_size': X_train.shape[1],
        'hidden_size1': GLOBAL_CONFIG['model']['hidden_size1'],
        'hidden_size2': GLOBAL_CONFIG['model']['hidden_size2'],
        'output_size': 10,  # EuroSAT_RGB有10个类别
        'hidden_activation': GLOBAL_CONFIG['model']['hidden_activation'],
        'output_activation': GLOBAL_CONFIG['model']['output_activation'],
        'loss': GLOBAL_CONFIG['model']['loss'],
        'optimizer': GLOBAL_CONFIG['train']['optimizer'],
        'learning_rate': GLOBAL_CONFIG['train']['learning_rate'],
        'momentum': GLOBAL_CONFIG['train']['momentum'],
        'weight_decay': GLOBAL_CONFIG['train']['weight_decay'],
        'scheduler': GLOBAL_CONFIG['train']['scheduler'],
        'scheduler_params': GLOBAL_CONFIG['train']['scheduler_params'],
        'batch_size': GLOBAL_CONFIG['train']['batch_size'],
        'epochs': 20,  # 使用较少的轮数以加快训练
        'model_dir': GLOBAL_CONFIG['path']['model_dir'],
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val
    }
    
    # 训练模型并获取历史记录
    print("训练模型以获取历史记录...")
    model, history = train_model(config)
    
    # 绘制训练历史
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
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()
    
    # 单独保存损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_path, 'training_loss.png'))
    plt.close()
    
    # 单独保存准确率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(save_path, 'training_accuracy.png'))
    plt.close()
    
    print(f"训练曲线已保存到 {os.path.join(save_path, 'training_history.png')}")
    print(f"损失曲线已保存到 {os.path.join(save_path, 'training_loss.png')}")
    print(f"准确率曲线已保存到 {os.path.join(save_path, 'training_accuracy.png')}")
    
    return history

def visualize_best_model_performance(save_path='results'):
    """
    可视化最佳模型在各个类别上的性能
    
    Args:
        save_path: 保存结果的路径
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 加载数据集
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'EuroSAT_RGB')
    processor = DataProcessor(data_dir)
    X, y = processor.load_data()
    X_train, X_test, y_train, y_test = processor.split_dataset(X, y)
    X_train, X_val, y_train, y_val = processor.split_validation(X_train, y_train)
    X_train, X_val, X_test = processor.normalize_data(X_train, X_val, X_test)
    
    # 定义EuroSAT_RGB类别
    EUROSAT_CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
                      'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    
    # 加载最佳模型
    model = ThreeLayerNet(
        input_size=X_train.shape[1],
        hidden_size1=GLOBAL_CONFIG['model']['hidden_size1'],
        hidden_size2=GLOBAL_CONFIG['model']['hidden_size2'],
        output_size=10,
        hidden_activation=GLOBAL_CONFIG['model']['hidden_activation'],
        output_activation='softmax'
    )
    model_dir = GLOBAL_CONFIG['path']['model_dir']
    model.load_model(os.path.join(model_dir, 'best_model.npz'))
    
    # 获取测试集预测结果
    y_pred = model.predict(X_test)
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(len(EUROSAT_CLASSES)):
        # 找出属于该类别的样本
        idx = (y_test == i)
        if np.sum(idx) > 0:  # 确保有该类别的样本
            # 计算该类别的准确率
            acc = np.mean(y_pred[idx] == y_test[idx])
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    # 绘制每个类别的准确率
    plt.figure(figsize=(10, 6))
    plt.bar(EUROSAT_CLASSES, class_accuracies)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_accuracy.png'))
    plt.close()
    
    print(f"类别准确率图已保存到 {os.path.join(save_path, 'class_accuracy.png')}")

if __name__ == "__main__":
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 生成训练曲线
    history = generate_training_curves()
    
    # 可视化最佳模型在各个类别上的性能
    visualize_best_model_performance()
    
    print("训练过程可视化完成！")