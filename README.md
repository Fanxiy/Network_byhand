# 手工实现三层神经网络分类器

本项目从零开始实现一个三层神经网络分类器，用于EuroSAT-RGB遥感图像分类任务。项目不使用任何自动微分框架（如PyTorch或TensorFlow），而是手动实现反向传播算法，通过NumPy手动实现了神经网络的前向传播、反向传播、各种激活函数、损失函数和优化器等核心组件。

## 1. 项目概述

本项目实现了一个完整的三层神经网络分类器，包括：

- **模型架构**：输入层 → 隐藏层1 → 隐藏层2 → 输出层
- **激活函数**：支持多种激活函数（Sigmoid、ReLU、Tanh、Softmax）
- **损失函数**：交叉熵损失函数
- **优化器**：SGD优化器（支持动量和L2正则化）
- **学习率调度**：支持学习率衰减策略
- **超参数搜索**：支持网格搜索和随机搜索两种方法

项目的主要目标是通过手动实现神经网络的各个组件，深入理解神经网络的工作原理和训练过程，并应用于遥感图像分类任务。

## 2. 项目结构

项目文件组织如下：

### 2.1 核心模型组件

```
├── code/model.py          # 神经网络模型定义，包含前向传播和反向传播
├── code/activation.py     # 激活函数及其导数的实现
├── code/losses.py         # 损失函数（交叉熵）及其导数的实现
├── code/optimizer.py      # SGD优化器实现，包含学习率调度和L2正则化
```

### 2.2 训练与评估组件

```
├── code/train.py          # 训练逻辑，包含模型训练和验证
├── code/evaluation.py     # 模型评估指标和方法
├── code/error_analysis.py # 错误分析工具
├── code/hyperparameter_search.py  # 超参数搜索实现
```

### 2.3 数据处理与工具

```
├── code/processor.py      # 数据处理和预处理
├── code/utils.py          # 工具函数
├── code/config.py         # 配置文件，包含默认超参数
```

### 2.4 可视化工具

```
├── code/visualize_training.py # 训练过程可视化工具
├── code/weight_visualization.py # 权重可视化工具
├── code/visualize_hyperparams.py # 超参数搜索结果可视化工具
```

### 2.5 主程序与结果

```
├── code/main.py           # 主程序入口
├── code/results/          # 结果文件夹，包含训练历史、权重可视化等
├── results/               # 结果文件夹
├── search_results/        # 超参数搜索结果
└── models/                # 模型保存文件夹
```

## 3. 模型介绍

### 3.1 模型架构

模型采用了经典的三层神经网络结构：

- **输入层**：接收EuroSAT-RGB图像数据，维度为12288（64×64×3）
- **第一隐藏层**：可配置的神经元数量
- **第二隐藏层**：可配置的神经元数量
- **输出层**：10个神经元，对应EuroSAT的10个类别

### 3.2 默认超参数配置

模型的默认超参数配置如下：

#### 3.2.1 模型参数

- **输入大小(input_size)**: 12288 (64×64×3，EuroSAT-RGB图像展平后的维度)
- **第一隐藏层大小(hidden_size1)**: 512个神经元
- **第二隐藏层大小(hidden_size2)**: 256个神经元
- **输出大小(output_size)**: 10 (EuroSAT的10个类别)
- **隐藏层激活函数(hidden_activation)**: Tanh
- **输出层激活函数(output_activation)**: Softmax

#### 3.2.2 训练参数

- **损失函数(loss)**: 交叉熵(cross_entropy)
- **优化器(optimizer)**: SGD
- **学习率(learning_rate)**: 0.01
- **动量系数(momentum)**: 0.9
- **权重衰减(weight_decay)**: 0.0001 (L2正则化系数)
- **批量大小(batch_size)**: 256
- **训练轮数(epochs)**: 50

#### 3.2.3 学习率调度

- **调度器类型(scheduler)**: step (步长衰减)
- **调度器参数(scheduler_params)**: 每10个epoch调整一次学习率，衰减因子为0.1

### 3.3 激活函数

- **隐藏层**：支持多种激活函数，包括ReLU、Sigmoid和Tanh
  - ReLU: $f(x) = \max(0, x)$
  - Sigmoid: $f(x) = \frac{1}{1 + e^{-x}}$
  - Tanh: $f(x) = \tanh(x)$
- **输出层**：使用Softmax激活函数，将输出转换为概率分布

### 3.4 损失函数

使用交叉熵损失函数，适用于多分类问题：
$L = -\sum_{i} y_{true,i} \log(y_{pred,i})$

### 3.5 优化器

使用带动量和L2正则化的随机梯度下降（SGD）优化器：

- 梯度计算：$\nabla L(\theta) = \nabla L_{loss}(\theta) + \lambda \theta$
- 速度更新：$v_t = \gamma v_{t-1} + \eta \nabla L(\theta)$
- 参数更新：$\theta = \theta - v_t$

其中，$\gamma$是动量系数，$\eta$是学习率，$\lambda$是L2正则化系数，$\nabla L(\theta)$是损失函数关于参数$\theta$的梯度。

### 3.6 参数初始化

使用Xavier/Glorot初始化方法初始化权重，有助于解决训练初期的梯度消失/爆炸问题。

## 4. 数据集介绍

### 4.1 EuroSAT-RGB数据集

EuroSAT是一个用于土地利用和土地覆盖分类的遥感图像数据集，包含10个类别的27,000张64×64彩色图像：

- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea/Lake

### 4.2 数据预处理

数据预处理步骤包括：

1. **归一化**：将像素值从[0, 255]归一化到[0, 1]范围
2. **标准化**：按通道计算均值和标准差并进行标准化
3. **数据展平**：将64×64×3的图像展平为12288维的向量
4. **数据分割**：将数据集分为训练集、验证集和测试集

## 5. 功能特点

### 5.1 模型部分

- 可自定义隐藏层大小
- 支持多种激活函数（Sigmoid, ReLU, Tanh等）
- 手动实现反向传播算法

### 5.2 训练部分

- 实现SGD优化器，支持动量和L2正则化
- 支持学习率下降策略
- 实现交叉熵损失函数
- 根据验证集性能自动保存最优模型

### 5.3 超参数搜索

项目实现了两种超参数搜索方法，用于寻找最佳的模型配置：

#### 5.3.1 网格搜索

网格搜索通过遍历所有可能的超参数组合来寻找最佳配置。

#### 5.3.2 随机搜索

随机搜索通过随机采样超参数空间来寻找最佳配置，计算效率更高。

### 5.4 测试与分析

- 支持导入训练好的模型
- 计算并输出测试集上的分类准确率
- 生成混淆矩阵和错误分析
- 可视化权重和训练过程

## 6. 实验结果

### 6.1 最佳配置

通过超参数搜索，找到了最佳的模型配置。

### 6.2 模型性能

最佳模型在测试集上的性能评估。

### 6.3 可视化结果

项目包含多种可视化结果，存储在 `results/` 和 `search_results/` 文件夹中：

- **训练历史**：损失曲线和准确率曲线
- **权重可视化**：第一层权重矩阵可视化
- **混淆矩阵**：分类结果的混淆矩阵
- **错误分析**：错误分类的样本分析
- **超参数搜索结果**：超参数与准确率的关系分析

## 7. 模型访问

训练好的最佳模型已上传至Google Drive，可通过以下链接访问：

[https://drive.google.com/file/d/1yvQ8wSRVj85qkrUZv_r169LcEF4ZwK3S/view?usp=sharing](https://drive.google.com/file/d/1yvQ8wSRVj85qkrUZv_r169LcEF4ZwK3S/view?usp=sharing)

## 8. 使用方法

### 8.1 训练模型

```bash
python code/main.py --mode train
```

可选参数：

- `--learning_rate`：学习率
- `--hidden_size1`：第一隐藏层大小
- `--hidden_size2`：第二隐藏层大小
- `--batch_size`：批量大小
- `--epochs`：训练轮数
- `--momentum`：动量系数
- `--weight_decay`：L2正则化强度
- `--activation`：激活函数类型

### 8.2 测试模型

```bash
python code/main.py --mode test --model_path path/to/model
```

### 8.3 超参数搜索

```bash
python code/main.py --mode search
```

## 9. 结论与改进方向

### 9.1 结论

本项目成功实现了一个从零开始的三层神经网络分类器，并应用于EuroSAT-RGB遥感图像分类任务。通过超参数搜索找到了较优的模型配置，并通过可视化分析了训练过程和模型参数。

主要成果：

- 实现了完整的神经网络训练流程，包括前向传播、反向传播、参数更新等
- 实现了多种激活函数、损失函数和优化器
- 通过超参数搜索提高了模型性能
- 通过可视化深入分析了模型的训练过程和学习到的特征

### 9.2 改进方向

尽管本项目成功实现了基本的神经网络，但仍有多个可改进的方向：

1. **模型架构**：

   - 使用卷积神经网络（CNN）替代全连接网络，更好地利用图像的空间结构
   - 增加网络深度，添加更多层以提高模型表达能力
   - 尝试残差连接（ResNet）等现代架构技术

2. **优化方法**：

   - 实现更先进的优化器，如Adam、RMSprop等
   - 添加Batch Normalization等正则化技术
   - 实现Dropout防止过拟合

3. **数据增强**：

   - 实现图像旋转、翻转、裁剪等数据增强技术
   - 使用更复杂的数据预处理方法

4. **其他改进**：

   - 实现早停（Early Stopping）策略
   - 尝试集成学习方法
   - 利用迁移学习提高模型性能

## 10. 依赖库

- NumPy：用于数学计算
- Matplotlib：用于可视化训练过程和结果
- scikit-learn：用于数据处理和评估指标
- PIL：用于图像加载和处理