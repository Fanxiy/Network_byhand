import numpy as np
import os
import json
import time
from model import ThreeLayerNet
from losses import get_loss
from optimizer import get_optimizer, get_scheduler
from train import Trainer
from evaluation import test_model
from utils import plot_training_history
from config import GLOBAL_CONFIG

class HyperparameterSearch:
    """
    超参数搜索类
    """
    def __init__(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, results_dir=GLOBAL_CONFIG['path']['search_results_dir']):
        """
        初始化超参数搜索
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            X_test: 测试数据（可选）
            y_test: 测试标签（可选）
            results_dir: 结果保存目录
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.results_dir = results_dir
        
        # 创建结果保存目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存搜索结果
        self.search_results = []
        self.best_config = None
        self.best_model = None
        self.best_val_acc = 0.0
    
    def train_and_evaluate(self, config, epochs=GLOBAL_CONFIG['search']['search_params']['epochs'], batch_size=GLOBAL_CONFIG['search']['search_params']['batch_size']):
        """
        使用给定配置训练和评估模型
        
        Args:
            config: 超参数配置字典
            epochs: 训练轮数
            batch_size: 批量大小
            
        Returns:
            验证准确率和训练历史记录
        """
        # 解析配置
        input_size = config.get('input_size', 64*64*3)  # EuroSAT: 64x64x3 = 12288
        hidden_size1 = config.get('hidden_size1', 512)
        hidden_size2 = config.get('hidden_size2', 256)
        output_size = config.get('output_size', 10)  # EuroSAT: 10个类别
        hidden_activation = config.get('hidden_activation', 'relu')
        output_activation = config.get('output_activation', 'softmax')
        loss_name = config.get('loss', 'cross_entropy')
        optimizer_name = config.get('optimizer', 'sgd')
        learning_rate = config.get('learning_rate', 0.01)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0001)
        scheduler_name = config.get('scheduler', 'step')
        scheduler_params = config.get('scheduler_params', {'step_size': 10, 'gamma': 0.1})
        
        # 创建模型
        model = ThreeLayerNet(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )
        
        # 创建损失函数
        loss_fn = get_loss(loss_name)
        
        # 创建优化器
        optimizer = get_optimizer(
            optimizer_name,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # 创建学习率调度器
        scheduler = None
        if scheduler_name is not None:
            scheduler = get_scheduler(scheduler_name, optimizer, **scheduler_params)
        
        # 创建训练器
        trainer = Trainer(model, loss_fn, optimizer, scheduler)
        
        # 训练模型
        history = trainer.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_dir=os.path.join(self.results_dir, 'models')
        )
        
        # 获取最佳验证准确率
        best_val_acc = max(history['val_acc'])
        
        # 如果提供了测试集，在测试集上评估
        test_acc = None
        if self.X_test is not None and self.y_test is not None:
            test_acc = test_model(model, self.X_test, self.y_test, batch_size)
        
        return best_val_acc, history, test_acc, model
    
    def grid_search(self, param_grid, epochs=GLOBAL_CONFIG['search']['search_params']['epochs'], batch_size=GLOBAL_CONFIG['search']['search_params']['batch_size']):
        """
        网格搜索超参数
        
        Args:
            param_grid: 超参数网格字典，每个键对应一个超参数，值为可能的取值列表
            epochs: 训练轮数
            batch_size: 批量大小
            
        Returns:
            最佳配置和所有搜索结果
        """
        # 生成所有可能的超参数组合
        import itertools
        
        # 获取所有参数名和可能的取值
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # 生成所有可能的组合
        combinations = list(itertools.product(*param_values))
        
        # 总共的组合数
        total_combinations = len(combinations)
        print(f"总共需要搜索 {total_combinations} 种超参数组合")
        
        # 搜索所有组合
        for i, combination in enumerate(combinations):
            # 构建当前配置
            config = {name: value for name, value in zip(param_names, combination)}
            
            # 添加固定参数
            config['input_size'] = 64*64*3  # EuroSAT: 64x64x3 = 12288
            config['output_size'] = 10  # EuroSAT: 10个类别
            config['output_activation'] = 'softmax'
            config['loss'] = 'cross_entropy'
            
            # 打印当前配置
            print(f"\n搜索进度: [{i+1}/{total_combinations}]")
            print("当前配置:")
            for name, value in config.items():
                print(f"  {name}: {value}")
            
            # 训练和评估
            start_time = time.time()
            val_acc, history, test_acc, model = self.train_and_evaluate(config, epochs, batch_size)
            end_time = time.time()
            
            # 记录结果
            result = {
                'config': config,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'training_time': end_time - start_time
            }
            self.search_results.append(result)
            
            # 更新最佳配置
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_config = config
                self.best_model = model
                print(f"找到新的最佳配置，验证准确率: {val_acc:.4f}")
                # 保存最佳模型
                best_model_path = os.path.join(self.results_dir, 'best_model.npz')
                model.save_model(best_model_path)
                print(f"保存最佳模型到: {best_model_path}")
            
            # 保存当前结果
            self.save_results()
        
        # 打印最佳配置
        print("\n超参数搜索完成")
        print("最佳配置:")
        for name, value in self.best_config.items():
            print(f"  {name}: {value}")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        
        return self.best_config, self.search_results
    
    def random_search(self, param_distributions, n_iter=GLOBAL_CONFIG['search']['search_params']['n_iter'], epochs=GLOBAL_CONFIG['search']['search_params']['epochs'], batch_size=GLOBAL_CONFIG['search']['search_params']['batch_size']):
        """
        随机搜索超参数
        
        Args:
            param_distributions: 超参数分布字典，每个键对应一个超参数，值为可能的取值列表或分布函数
            n_iter: 随机搜索的迭代次数
            epochs: 训练轮数
            batch_size: 批量大小
            
        Returns:
            最佳配置和所有搜索结果
        """
        print(f"总共需要搜索 {n_iter} 种随机超参数组合")
        
        # 搜索n_iter次
        for i in range(n_iter):
            # 随机采样配置
            config = {}
            for name, distribution in param_distributions.items():
                if callable(distribution):
                    # 如果是分布函数，调用它
                    config[name] = distribution()
                elif isinstance(distribution, list):
                    # 如果是列表，随机选择一个值
                    config[name] = np.random.choice(distribution)
                else:
                    # 否则直接使用该值
                    config[name] = distribution
            
            # 添加固定参数
            config['input_size'] = 64*64*3  # EuroSAT: 64x64x3 = 12288
            config['output_size'] = 10  # EuroSAT: 10个类别
            config['output_activation'] = 'softmax'
            config['loss'] = 'cross_entropy'
            
            # 打印当前配置
            print(f"\n搜索进度: [{i+1}/{n_iter}]")
            print("当前配置:")
            for name, value in config.items():
                print(f"  {name}: {value}")
            
            # 训练和评估
            start_time = time.time()
            val_acc, history, test_acc, model = self.train_and_evaluate(config, epochs, batch_size)
            end_time = time.time()
            
            # 记录结果
            result = {
                'config': config,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'training_time': end_time - start_time
            }
            self.search_results.append(result)
            
            # 更新最佳配置
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_config = config
                self.best_model = model
                print(f"找到新的最佳配置，验证准确率: {val_acc:.4f}")
                # 保存最佳模型
                best_model_path = os.path.join(self.results_dir, 'best_model.npz')
                model.save_model(best_model_path)
                print(f"保存最佳模型到: {best_model_path}")
            
            # 保存当前结果
            self.save_results()
        
        # 打印最佳配置
        print("\n超参数搜索完成")
        print("最佳配置:")
        for name, value in self.best_config.items():
            print(f"  {name}: {value}")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        
        return self.best_config, self.search_results
    
    def save_results(self):
        """
        保存搜索结果
        """
        # 将结果保存为JSON文件
        results_path = os.path.join(self.results_dir, 'search_results.json')
        
        # 将NumPy数组转换为列表，以便JSON序列化
        serializable_results = []
        for result in self.search_results:
            serializable_result = {
                'config': {k: v if not isinstance(v, np.ndarray) and not isinstance(v, np.number) else v.item() 
                          for k, v in result['config'].items()},
                'val_acc': float(result['val_acc']),
                'test_acc': float(result['test_acc']) if result['test_acc'] is not None else None,
                'training_time': float(result['training_time'])
            }
            serializable_results.append(serializable_result)
        
        # 保存最佳配置
        best_config = None
        if self.best_config is not None:
            best_config = {k: v if not isinstance(v, np.ndarray) and not isinstance(v, np.number) else v.item() 
                          for k, v in self.best_config.items()}
        
        # 保存结果
        with open(results_path, 'w') as f:
            json.dump({
                'results': serializable_results,
                'best_config': best_config,
                'best_val_acc': float(self.best_val_acc)
            }, f, indent=2)
    
    def load_results(self):
        """
        加载搜索结果
        
        Returns:
            是否成功加载
        """
        results_path = os.path.join(self.results_dir, 'search_results.json')
        
        if not os.path.exists(results_path):
            return False
        
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            self.search_results = data['results']
            self.best_config = data['best_config']
            self.best_val_acc = data['best_val_acc']
            
            return True
        except Exception as e:
            print(f"加载搜索结果失败: {e}")
            return False
    
    def visualize_results(self):
        """
        可视化搜索结果
        """
        import matplotlib.pyplot as plt
        
        if not self.search_results:
            print("没有搜索结果可供可视化")
            return
        
        # 提取参数和性能指标
        configs = [result['config'] for result in self.search_results]
        val_accs = [result['val_acc'] for result in self.search_results]
        test_accs = [result['test_acc'] for result in self.search_results if result['test_acc'] is not None]
        training_times = [result['training_time'] for result in self.search_results]
        
        # 找出所有唯一的超参数
        all_params = set()
        for config in configs:
            all_params.update(config.keys())
        
        # 排除固定参数
        fixed_params = {'input_size', 'output_size', 'output_activation', 'loss'}
        varying_params = all_params - fixed_params
        
        # 为每个变化的超参数创建一个图
        for param in varying_params:
            # 提取该参数的所有值
            param_values = [config.get(param, None) for config in configs]
            
            # 如果参数值都相同，跳过
            if len(set(param_values)) <= 1:
                continue
            
            # 创建图
            plt.figure(figsize=(10, 6))
            
            # 绘制验证准确率
            plt.subplot(1, 2, 1)
            plt.scatter(param_values, val_accs)
            plt.xlabel(param)
            plt.ylabel('Validation Accuracy')
            plt.title(f'{param} vs. Validation Accuracy')
            
            # 如果有测试准确率，也绘制
            if test_accs:
                plt.subplot(1, 2, 2)
                plt.scatter(param_values[:len(test_accs)], test_accs)
                plt.xlabel(param)
                plt.ylabel('Test Accuracy')
                plt.title(f'{param} vs. Test Accuracy')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'{param}_vs_accuracy.png'))
            plt.close()
        
        # 绘制训练时间
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(training_times)), training_times)
        plt.xlabel('Configuration Index')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time for Each Configuration')
        plt.savefig(os.path.join(self.results_dir, 'training_times.png'))
        plt.close()
        
        # 绘制验证准确率排序
        sorted_indices = np.argsort(val_accs)[::-1]  # 降序
        sorted_val_accs = [val_accs[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_val_accs)), sorted_val_accs)
        plt.xlabel('Configuration Rank')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy (Sorted)')
        plt.savefig(os.path.join(self.results_dir, 'sorted_val_accs.png'))
        plt.close()


def example_grid_search(X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """
    示例网格搜索
    
    Args:
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
        X_test: 测试数据（可选）
        y_test: 测试标签（可选）
        
    Returns:
        最佳配置和所有搜索结果
    """
    # 创建超参数搜索实例
    searcher = HyperparameterSearch(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 从配置中获取超参数网格
    param_grid = {
        'hidden_size1': GLOBAL_CONFIG['search']['example_grid_search']['hidden_size1'],
        'hidden_size2': GLOBAL_CONFIG['search']['example_grid_search']['hidden_size2'],
        'hidden_activation': GLOBAL_CONFIG['search']['example_grid_search']['hidden_activation'],
        'learning_rate': GLOBAL_CONFIG['search']['example_grid_search']['learning_rate'],
        'momentum': GLOBAL_CONFIG['search']['example_grid_search']['momentum'],
        'weight_decay': GLOBAL_CONFIG['search']['example_grid_search']['weight_decay']
    }
    
    # 执行网格搜索
    best_config, results = searcher.grid_search(
        param_grid, 
        epochs=GLOBAL_CONFIG['search']['example_grid_search']['epochs'], 
        batch_size=GLOBAL_CONFIG['search']['example_grid_search']['batch_size']
    )
    
    # 可视化结果
    searcher.visualize_results()
    
    return best_config, results


def example_random_search(X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """
    示例随机搜索
    
    Args:
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
        X_test: 测试数据（可选）
        y_test: 测试标签（可选）
        
    Returns:
        最佳配置和所有搜索结果
    """
    # 创建超参数搜索实例
    searcher = HyperparameterSearch(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 从配置中获取超参数分布
    param_distributions = {
        'hidden_size1': lambda: np.random.choice(GLOBAL_CONFIG['search']['example_random_search']['hidden_size1']),
        'hidden_size2': lambda: np.random.choice(GLOBAL_CONFIG['search']['example_random_search']['hidden_size2']),
        'hidden_activation': GLOBAL_CONFIG['search']['example_random_search']['hidden_activation'],
        'learning_rate': lambda: 10 ** np.random.uniform(
            GLOBAL_CONFIG['search']['example_random_search']['learning_rate_range'][0], 
            GLOBAL_CONFIG['search']['example_random_search']['learning_rate_range'][1]
        ),
        'momentum': lambda: np.random.uniform(
            GLOBAL_CONFIG['search']['example_random_search']['momentum_range'][0], 
            GLOBAL_CONFIG['search']['example_random_search']['momentum_range'][1]
        ),
        'weight_decay': lambda: 10 ** np.random.uniform(
            GLOBAL_CONFIG['search']['example_random_search']['weight_decay_range'][0], 
            GLOBAL_CONFIG['search']['example_random_search']['weight_decay_range'][1]
        )
    }
    
    # 执行随机搜索
    best_config, results = searcher.random_search(
        param_distributions, 
        n_iter=GLOBAL_CONFIG['search']['example_random_search']['n_iter'], 
        epochs=GLOBAL_CONFIG['search']['example_random_search']['epochs'], 
        batch_size=GLOBAL_CONFIG['search']['example_random_search']['batch_size']
    )
    
    # 可视化结果
    searcher.visualize_results()
    
    return best_config, results