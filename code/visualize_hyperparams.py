import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取超参数搜索结果
with open('search_results/search_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
best_config = data['best_config']
best_val_acc = data['best_val_acc']

# 提取超参数和性能指标
hidden_size1_values = []
hidden_size2_values = []
hidden_activation_values = []
learning_rate_values = []
momentum_values = []
weight_decay_values = []
val_acc_values = []
test_acc_values = []
training_time_values = []

for result in results:
    config = result['config']
    hidden_size1_values.append(config['hidden_size1'])
    hidden_size2_values.append(config['hidden_size2'])
    hidden_activation_values.append(config['hidden_activation'])
    learning_rate_values.append(config['learning_rate'])
    momentum_values.append(config['momentum'])
    weight_decay_values.append(config['weight_decay'])
    val_acc_values.append(result['val_acc'])
    test_acc_values.append(result['test_acc'])
    training_time_values.append(result['training_time'])

# 创建可视化目录
os.makedirs('search_results/visualizations', exist_ok=True)

# 1. 超参数组合与验证准确率的关系
plt.figure(figsize=(12, 8))

# 按隐藏层大小分组
unique_hidden_sizes = list(set(zip(hidden_size1_values, hidden_size2_values)))
hidden_size_labels = [f'{h1}-{h2}' for h1, h2 in unique_hidden_sizes]

# 为每个隐藏层大小组合创建子图
for i, (h1, h2) in enumerate(unique_hidden_sizes):
    plt.subplot(2, 2, i+1)
    
    # 过滤当前隐藏层大小的数据
    indices = [j for j, (hs1, hs2) in enumerate(zip(hidden_size1_values, hidden_size2_values)) if hs1 == h1 and hs2 == h2]
    
    # 提取对应的学习率和验证准确率
    lr = [learning_rate_values[j] for j in indices]
    acc = [val_acc_values[j] for j in indices]
    activations = [hidden_activation_values[j] for j in indices]
    
    # 按激活函数分组
    activation_types = list(set(activations))
    colors = ['red', 'blue', 'green']
    
    for act, color in zip(activation_types, colors):
        act_indices = [k for k, a in enumerate(activations) if a == act]
        act_lr = [lr[k] for k in act_indices]
        act_acc = [acc[k] for k in act_indices]
        plt.scatter(act_lr, act_acc, label=act, color=color, alpha=0.6)
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Hidden Sizes: {h1}-{h2}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('search_results/visualizations/hyperparam_combinations.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 最优配置与其他配置的对比
plt.figure(figsize=(10, 6))

# 计算所有配置的验证准确率分布
plt.hist(val_acc_values, bins=20, alpha=0.7, label='All Configurations')

# 标记最佳配置的准确率
plt.axvline(x=best_val_acc, color='red', linestyle='--', linewidth=2, label=f'Best Configuration\nAccuracy: {best_val_acc:.4f}')

plt.xlabel('Validation Accuracy')
plt.ylabel('Count')
plt.title('Distribution of Validation Accuracies')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('search_results/visualizations/accuracy_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 超参数重要性分析
plt.figure(figsize=(12, 8))

# 学习率对准确率的影响
plt.subplot(2, 2, 1)
plt.scatter(learning_rate_values, val_acc_values, alpha=0.6)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy')
plt.title('Learning Rate vs Validation Accuracy')
plt.grid(True, alpha=0.3)

# 动量对准确率的影响
plt.subplot(2, 2, 2)
plt.scatter(momentum_values, val_acc_values, alpha=0.6)
plt.xlabel('Momentum')
plt.ylabel('Validation Accuracy')
plt.title('Momentum vs Validation Accuracy')
plt.grid(True, alpha=0.3)

# 权重衰减对准确率的影响
plt.subplot(2, 2, 3)
plt.scatter(weight_decay_values, val_acc_values, alpha=0.6)
plt.xscale('log')
plt.xlabel('Weight Decay')
plt.ylabel('Validation Accuracy')
plt.title('Weight Decay vs Validation Accuracy')
plt.grid(True, alpha=0.3)

# 隐藏层1大小对准确率的影响
plt.subplot(2, 2, 4)
plt.scatter(hidden_size1_values, val_acc_values, alpha=0.6)
plt.xlabel('Hidden Size 1')
plt.ylabel('Validation Accuracy')
plt.title('Hidden Size 1 vs Validation Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('search_results/visualizations/hyperparam_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print('Visualization completed! Check the search_results/visualizations directory for the generated plots.')
print(f'Best configuration: {best_config}')
print(f'Best validation accuracy: {best_val_acc}')
