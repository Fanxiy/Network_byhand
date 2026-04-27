import numpy as np
import os
import time
from model import ThreeLayerNet
from losses import get_loss
from optimizer import get_optimizer, get_scheduler
from utils import calculate_accuracy
from processor import get_mini_batch

class Trainer:
    """
    神经网络训练器
    """
    def __init__(self, model, loss_fn, optimizer, scheduler=None):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            loss_fn: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_acc = 0.0
        self.best_model_params = None
    
    def train_epoch(self, X_train, y_train, batch_size):
        """
        训练一个epoch
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            batch_size: 批量大小
            
        Returns:
            平均损失和准确率
        """
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        # 生成小批量数据
        for X_batch, y_batch in get_mini_batch(X_train, y_train, batch_size):
            # 前向传播
            y_pred = self.model.forward(X_batch)
            
            # 计算损失
            loss = self.loss_fn.forward(y_pred, y_batch)
            
            # 计算梯度
            dout = self.loss_fn.backward(y_pred, y_batch)
            grads = self.model.backward(dout)
            
            # 更新参数
            self.optimizer.update(self.model.params, grads)
            
            # 计算准确率
            y_pred_class = self.model.predict(X_batch)
            acc = calculate_accuracy(y_pred_class, y_batch)
            
            # 累加损失和准确率
            epoch_loss += loss
            epoch_acc += acc
            n_batches += 1
        
        # 计算平均损失和准确率
        epoch_loss /= n_batches
        epoch_acc /= n_batches
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, X_val, y_val, batch_size):
        """
        在验证集上评估模型
        
        Args:
            X_val: 验证数据
            y_val: 验证标签
            batch_size: 批量大小
            
        Returns:
            平均损失和准确率
        """
        val_loss = 0.0
        val_acc = 0.0
        n_batches = 0
        
        # 生成小批量数据
        for X_batch, y_batch in get_mini_batch(X_val, y_val, batch_size):
            # 前向传播
            y_pred = self.model.forward(X_batch)
            
            # 计算损失
            loss = self.loss_fn.forward(y_pred, y_batch)
            
            # 计算准确率
            y_pred_class = self.model.predict(X_batch)
            acc = calculate_accuracy(y_pred_class, y_batch)
            
            # 累加损失和准确率
            val_loss += loss
            val_acc += acc
            n_batches += 1
        
        # 计算平均损失和准确率
        val_loss /= n_batches
        val_acc /= n_batches
        
        return val_loss, val_acc
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, model_dir='models'):
        """
        训练模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批量大小
            model_dir: 模型保存目录
            
        Returns:
            训练历史记录
        """
        # 创建模型保存目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 初始化训练历史记录
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # 记录最佳验证准确率
        self.best_val_acc = 0.0
        
        # 训练循环
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            
            # 在验证集上评估
            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
            
            # 更新学习率
            current_lr = self.optimizer.learning_rate
            if self.scheduler is not None:
                if self.scheduler.__class__.__name__ == 'ReduceOnPlateau':
                    current_lr = self.scheduler.step(val_loss)
                else:
                    current_lr = self.scheduler.step(epoch)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_params = {k: v.copy() for k, v in self.model.params.items()}
                model_path = os.path.join(model_dir, 'best_model.npz')
                self.model.save_model(model_path)
                print(f"Epoch {epoch+1}/{epochs} - 保存最佳模型，验证准确率: {val_acc:.4f}")
            
            # 更新训练历史记录
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # 打印训练信息
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {epoch+1}/{epochs} - 耗时: {epoch_time:.2f}s - 学习率: {current_lr:.6f}")
            print(f"  训练损失: {train_loss:.4f} - 训练准确率: {train_acc:.4f}")
            print(f"  验证损失: {val_loss:.4f} - 验证准确率: {val_acc:.4f}")
        
        # 恢复最佳模型参数
        if self.best_model_params is not None:
            self.model.params = self.best_model_params
        
        return history


def train_model(config):
    """
    使用给定配置训练模型
    
    Args:
        config: 训练配置字典
        
    Returns:
        训练好的模型和训练历史记录
    """
    # 解析配置
    input_size = config.get('input_size')  
    hidden_size1 = config.get('hidden_size1')
    hidden_size2 = config.get('hidden_size2')
    output_size = config.get('output_size') 
    hidden_activation = config.get('hidden_activation')
    output_activation = config.get('output_activation')
    loss_name = config.get('loss')
    optimizer_name = config.get('optimizer')
    learning_rate = config.get('learning_rate')
    momentum = config.get('momentum')
    weight_decay = config.get('weight_decay')
    scheduler_name = config.get('scheduler')
    scheduler_params = config.get('scheduler_params')
    batch_size = config.get('batch_size')
    epochs = config.get('epochs')
    model_dir = config.get('model_dir')
    
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
        from optimizer import get_scheduler
        scheduler = get_scheduler(scheduler_name, optimizer, **scheduler_params)
    
    # 创建训练器
    trainer = Trainer(model, loss_fn, optimizer, scheduler)
    
    # 训练模型
    history = trainer.train(
        config['X_train'], config['y_train'],
        config['X_val'], config['y_val'],
        epochs=epochs,
        batch_size=batch_size,
        model_dir=model_dir
    )
    
    return model, history