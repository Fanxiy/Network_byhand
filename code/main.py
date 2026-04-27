import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model import ThreeLayerNet
from losses import get_loss
from optimizer import get_optimizer, get_scheduler
from train import train_model
from evaluation import load_and_test, evaluate_model_classes, plot_confusion_matrix
from hyperparameter_search import HyperparameterSearch, example_grid_search, example_random_search
from processor import DataProcessor
from error_analysis import ErrorAnalyzer
from utils import visualize_sample, plot_training_history
from config import GLOBAL_CONFIG

def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='三层神经网络分类器')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'search', 'visualize'],
                        help='运行模式：训练、测试、超参数搜索或可视化')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=GLOBAL_CONFIG['data']['data_dir'],
                        help='数据集目录')
    
    # 模型参数
    parser.add_argument('--hidden_size1', type=int, default=GLOBAL_CONFIG['model']['hidden_size1'],
                        help='第一隐藏层大小')
    parser.add_argument('--hidden_size2', type=int, default=GLOBAL_CONFIG['model']['hidden_size2'],
                        help='第二隐藏层大小')
    parser.add_argument('--hidden_activation', type=str, default=GLOBAL_CONFIG['model']['hidden_activation'],
                        help='隐藏层激活函数')
    
    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=GLOBAL_CONFIG['train']['learning_rate'],
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=GLOBAL_CONFIG['train']['momentum'],
                        help='动量系数')
    parser.add_argument('--weight_decay', type=float, default=GLOBAL_CONFIG['train']['weight_decay'],
                        help='L2正则化系数')
    parser.add_argument('--batch_size', type=int, default=GLOBAL_CONFIG['train']['batch_size'],
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=GLOBAL_CONFIG['train']['epochs'],
                        help='训练轮数')
    
    # 学习率调度参数
    parser.add_argument('--scheduler', type=str, default=GLOBAL_CONFIG['train']['scheduler'],
                        help='学习率调度器类型')
    
    # 路径参数
    parser.add_argument('--model_dir', type=str, default=GLOBAL_CONFIG['path']['model_dir'],
                        help='模型保存目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径（用于测试模式）')
    parser.add_argument('--results_dir', type=str, default=GLOBAL_CONFIG['path']['results_dir'],
                        help='结果保存目录')
    
    # 超参数搜索参数
    parser.add_argument('--search_type', type=str, default='grid', choices=['grid', 'random'],
                        help='超参数搜索类型：网格搜索或随机搜索')
    parser.add_argument('--search_epochs', type=int, default=10,
                        help='超参数搜索中每个配置的训练轮数')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 创建必要的目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 创建数据处理器
    data_processor = DataProcessor(args.data_dir)
    
    # 加载EuroSAT数据集
    try:
        X, y = data_processor.load_data()
        print(f"加载EuroSAT数据集成功！数据形状: {X.shape}")
        class_names = data_processor.class_names
    except Exception as e:
        print(f"加载EuroSAT数据集失败: {e}")
        print("请确保数据集目录正确")
        return
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = data_processor.split_dataset(X, y, test_size=0.2, random_state=42)
    
    # 从训练集中划分验证集
    X_train, X_val, y_train, y_val = data_processor.split_validation(
        X_train, y_train, valid_size=GLOBAL_CONFIG['data']['valid_size']
    )
    
    # 数据标准化和展平
    X_train, X_val, X_test = data_processor.normalize_data(X_train, X_val, X_test)
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        # 训练模式
        print("\n开始训练模型...")
        
        # 构建训练配置
        train_config = {
            'input_size': X_train.shape[1],
            'hidden_size1': args.hidden_size1,
            'hidden_size2': args.hidden_size2,
            'output_size': len(class_names),
            'hidden_activation': args.hidden_activation,
            'output_activation': 'softmax',
            'loss': 'cross_entropy',
            'optimizer': 'sgd',
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'scheduler': args.scheduler,
            'scheduler_params': GLOBAL_CONFIG['train']['scheduler_params'],
            'batch_size': args.batch_size,  
            'epochs': args.epochs,
            'model_dir': args.model_dir,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }
        
        # 训练模型
        model, history = train_model(train_config)
        
        # 绘制训练历史
        history_save_path = os.path.join(args.model_dir, 'training_history.png')
        plot_training_history(history, save_path=history_save_path)
        
        # 在测试集上评估
        print("\n在测试集上评估模型...")
        test_acc = load_and_test(
            os.path.join(args.model_dir, 'best_model.npz'),
            X_test, y_test,
            model_config={
                'input_size': X_train.shape[1],
                'hidden_size1': args.hidden_size1,
                'hidden_size2': args.hidden_size2,
                'output_size': len(class_names),
                'hidden_activation': args.hidden_activation,
                'output_activation': 'softmax'
            }
        )
        
        # 评估各类别性能
        evaluate_model_classes(model, X_test, y_test, class_names)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(model, X_test, y_test, class_names, 
                             save_path=os.path.join(args.results_dir, 'confusion_matrix.png'))
        
    elif args.mode == 'test':
        # 测试模式
        print("\n开始测试模型...")
        
        # 检查模型路径
        model_path = args.model_path
        if model_path is None:
            model_path = os.path.join(args.model_dir, 'best_model.npz')
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return
        
        # 加载并测试模型
        test_acc = load_and_test(
            model_path,
            X_test, y_test,
            model_config={
                'input_size': X_train.shape[1],
                'hidden_size1': args.hidden_size1,
                'hidden_size2': args.hidden_size2,
                'output_size': len(class_names),
                'hidden_activation': args.hidden_activation,
                'output_activation': 'softmax'
            }
        )
        
        # 创建模型实例用于评估各类别性能
        model = ThreeLayerNet(
            input_size=X_train.shape[1],
            hidden_size1=args.hidden_size1,
            hidden_size2=args.hidden_size2,
            output_size=len(class_names),
            hidden_activation=args.hidden_activation,
            output_activation='softmax'
        )
        model.load_model(model_path)
        
        # 评估各类别性能
        evaluate_model_classes(model, X_test, y_test, class_names)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(model, X_test, y_test, class_names, 
                             save_path=os.path.join(args.results_dir, 'confusion_matrix.png'))
        
        # 错例分析
        print("\n开始错例分析...")
        analyzer = ErrorAnalyzer(model, class_names, X_test, y_test)
        misclassified_indices, y_pred = analyzer.find_misclassified_samples()
        
        if len(misclassified_indices) > 0:
            # 可视化错误样本
            analyzer.visualize_misclassified_samples(y_pred, misclassified_indices, 
                                                  save_dir=args.results_dir)
            
            # 分析错误模式
            analyzer.analyze_error_patterns(y_pred)
            
            # 分析前3个错误样本
            for i in range(min(3, len(misclassified_indices))):
                analyzer.analyze_specific_error(y_pred, misclassified_indices, i)
        else:
            print("没有分类错误的样本！")
        
    elif args.mode == 'search':
        # 超参数搜索模式
        print("\n开始超参数搜索...")
        
        if args.search_type == 'grid':
            # 网格搜索
            print("执行网格搜索...")
            best_config, results = example_grid_search(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
        else:
            # 随机搜索
            print("执行随机搜索...")
            best_config, results = example_random_search(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
        
        # 打印最佳配置
        print("\n超参数搜索完成！")
        print("最佳配置:")
        for name, value in best_config.items():
            print(f"  {name}: {value}")
        
    elif args.mode == 'visualize':
        # 可视化模式
        print("\n可视化样本...")
        
        # 随机选择一些样本进行可视化
        indices = np.random.choice(len(X_test), 25, replace=False)
        visualize_sample(X_test[indices], y_test[indices], class_names)

if __name__ == '__main__':
    main()