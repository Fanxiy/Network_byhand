
## 目录路径
import os

# 脚本所在目录
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

# 项目根目录
project_root = os.path.dirname(script_dir)

# 数据集路径
data_name = "EuroSAT_RGB"
data_dir = os.path.join(project_root, data_name)

## 全局配置
GLOBAL_CONFIG = {
    # 数据处理配置
    'data': {
        'data_dir': data_dir,
        'img_size': (64, 64),
        'max_samples': None,
        'test_size': 0.2,
        'valid_size': 0.1,
        'random_state': 42
    },
    
    # 模型配置
    'model': {
        'input_size': 64*64*3,  # 64x64 RGB images
        'hidden_size1': 256,
        'hidden_size2': 128,
        'output_size': 10,  # EuroSAT has 10 classes
        'hidden_activation': 'tanh',
        'output_activation': 'softmax',
        'loss': 'cross_entropy'
    },
    
    # 训练配置
    'train': {
        'optimizer': 'sgd',
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'scheduler': 'step',
        'scheduler_params': {'step_size': 10, 'gamma': 0.5},
        'batch_size': 32,
        'epochs': 50,
        'patience': 10
    },
    
    # 路径配置
    'path': {
        'model_dir': 'models',
        'results_dir': 'results',
        'visualizations_dir': 'visualizations',
        'search_results_dir': 'search_results'
    },
    
    # 超参数搜索配置
    'search': {
        'grid_search': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_size1': [256, 512],
            'hidden_size2': [128, 256],
            'batch_size': [32, 64],
            'optimizer': 'sgd'
        },
        'random_search': {
            'hidden_size1': [128, 256, 512, 1024],
            'hidden_size2': [64, 128, 256, 512],
            'hidden_activation': ['relu', 'tanh'],
            'learning_rate_range': [-4, -1],  # 对数均匀分布范围
            'momentum_range': [0, 0.99],
            'weight_decay_range': [-5, -2]
        },
        'search_params': {
            'n_iter': 10,
            'epochs': 20,
            'batch_size': 128
        },
        'example_grid_search': {
            'hidden_size1': [128, 256],
            'hidden_size2': [64, 128],
            'hidden_activation': ['relu', 'tanh'],
            'learning_rate': [0.1, 0.01],
            'momentum': [0.0, 0.9],
            'weight_decay': [0.0, 0.005, 0.001],
            'epochs': 10,
            'batch_size': 128
        },
        'example_random_search': {
            'hidden_size1': [128, 256, 512, 1024],
            'hidden_size2': [64, 128, 256, 512],
            'hidden_activation': ['relu', 'tanh'],
            'learning_rate_range': [-4, -1],
            'momentum_range': [0, 0.99],
            'weight_decay_range': [-5, -2],
            'n_iter': 10,
            'epochs': 10,
            'batch_size': 128
        }
    },
    
    # 分析配置
    'analysis': {
        'max_samples': 1000,
        'save_dir': 'visualizations'
    }
}

