import numpy as np
import matplotlib.pyplot as plt
import os
from config import GLOBAL_CONFIG

class ErrorAnalyzer:
    def __init__(self, model, class_names, X_test, y_test):
        self.model = model
        self.class_names = class_names
        self.X_test = X_test 
        self.y_test = y_test

    def find_misclassified_samples(self):
        y_pred = self.model.predict(self.X_test)
        misclassified_indices = np.where(y_pred != self.y_test)[0]
        
        print(f"测试集大小：{len(self.X_test)}")
        print(f"分类错误的样本数量：{len(misclassified_indices)}")
        print(f"错误率：{len(misclassified_indices) / len(self.X_test):.4f}")
        
        return misclassified_indices, y_pred

    def visualize_misclassified_samples(self, y_pred, misclassified_indices, 
                                        save_dir=GLOBAL_CONFIG['analysis']['save_dir']):
        os.makedirs(save_dir, exist_ok=True)
        num_visualize = min(12, len(misclassified_indices))
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(num_visualize):
            idx = misclassified_indices[i]
            img = self.X_test[idx].reshape(64, 64, 3)
            img = (img - img.min()) / (img.max() - img.min())
            true_label = self.class_names[self.y_test[idx]]
            pred_label = self.class_names[y_pred[idx]]
            axes[i].imshow(img)
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'misclassified_samples.png')
        plt.savefig(save_path)
        print(f"错例可视化结果已保存到：{save_path}")
        plt.show()

    def analyze_error_patterns(self, y_pred):
        print("\n错误模式分析：")
        num_classes = len(self.class_names)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        for true, pred in zip(self.y_test, y_pred):
            confusion_matrix[true, pred] += 1
        
        for i in range(num_classes):
            true_class = self.class_names[i]
            errors = confusion_matrix[i, :].copy()
            errors[i] = 0
            if np.sum(errors) > 0:
                print(f"\n类别 '{true_class}' 的错误分类：")
                for j in range(num_classes):
                    if errors[j] > 0:
                        pred_class = self.class_names[j]
                        print(f"  被错误分类为 '{pred_class}'：{errors[j]} 个样本")

    def analyze_specific_error(self, y_pred, misclassified_indices, idx):
        if idx >= len(misclassified_indices):
            print(f"索引超出范围，最大索引为：{len(misclassified_indices)-1}")
            return
        
        sample_idx = misclassified_indices[idx]
        img = self.X_test[sample_idx].reshape(64, 64, 3)
        img = (img - img.min()) / (img.max() - img.min())
        true_label = self.class_names[self.y_test[sample_idx]]
        pred_label = self.class_names[y_pred[sample_idx]]
        
        print(f"\n分析错误样本 {idx+1}：")
        print(f"  真实标签：{true_label}")
        print(f"  预测标签：{pred_label}")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
        plt.show()
        self._analyze_error_reason(true_label, pred_label)

    def _analyze_error_reason(self, true_label, pred_label):
        print("  可能的错误原因：")
        error_patterns = {
            ('River', 'Highway'): "河流和高速公路在卫星图像中可能看起来都是线性特征",
            ('Highway', 'River'): "高速公路和河流在卫星图像中可能看起来都是线性特征",
            ('Residential', 'Industrial'): "住宅区和工业区布局相似",
            ('Industrial', 'Residential'): "工业区和住宅区布局相似",
            ('HerbaceousVegetation', 'Pasture'): "绿色调相似",
            ('Pasture', 'HerbaceousVegetation'): "绿色调相似",
            ('AnnualCrop', 'PermanentCrop'): "作物纹理颜色相似",
            ('PermanentCrop', 'AnnualCrop'): "作物纹理颜色相似",
            ('Forest', 'HerbaceousVegetation'): "绿色调相似",
            ('HerbaceousVegetation', 'Forest'): "绿色调相似",
        }
        key = (true_label, pred_label)
        if key in error_patterns:
            print(f"  - {error_patterns[key]}")
        else:
            print("  - 图像特征、光照或角度导致分类困难")
