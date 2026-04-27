import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from config import GLOBAL_CONFIG


class DataProcessor:
    """
    数据处理：加载，预处理，切分数据集
    """

    def __init__(self, data_dir, img_size=GLOBAL_CONFIG['data']['img_size']):
        """
        初始化数据处理器

        Args:
            data_dir: 数据集目录路径
            img_size: 统一图像尺寸 (height, width)
        """
        self.data_dir = data_dir
        self.img_size = img_size 
        self.class_names = self._get_class_names()
        self.mean = None
        self.std = None

    def _get_class_names(self):
        """
        遍历数据集目录获取类别名称
        
        Returns:
            类别名称列表
        """
        if not os.path.exists(self.data_dir):
            print(f"警告：数据集目录 {self.data_dir} 不存在")
            return []
        
        class_names = []
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                class_names.append(item)
        class_names.sort()  
        print(f"从目录中获取到 {len(class_names)} 个类别：{class_names}")
        return class_names
    
    def load_data(self, max_samples=GLOBAL_CONFIG['data']['max_samples']):
        """
        加载数据集

        Args:
            max_samples: 最大样本数量，None表示加载所有样本

        Returns:
            X: 图像数据，形状为(N, H, W, C)
            y: 标签，形状为(N,)
        """
        X = []
        y = []
        sample_count = 0

        print(f"正在加载数据集，目录：{self.data_dir}")

        # 遍历每个类别目录
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"警告：类别目录不存在，跳过")
                continue

            # 遍历图像
            files = os.listdir(class_dir)
            for img_file in files:
                # 常见图片格式 + 大小写兼容
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_file)
                    
                    try:
                        # 统一尺寸
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')
                            img = img.resize(self.img_size)  
                            img_array = np.array(img)
                        
                        X.append(img_array)
                        y.append(class_idx)
                        sample_count += 1

                        if max_samples is not None and sample_count >= max_samples:
                            print(f"已达到最大样本数量 {max_samples}")
                            return np.array(X), np.array(y)
                    except Exception as e:
                        print(f"图片读取失败: {img_path}, 错误: {e}")

        X = np.array(X)
        y = np.array(y)
        print(f"加载完成：共 {sample_count} 张图片，形状：{X.shape}")
        return X, y

    def split_dataset(self, X, y, test_size=GLOBAL_CONFIG['data']['test_size'], random_state=GLOBAL_CONFIG['data']['random_state']):
        """
        切分数据集为训练集和测试集
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def split_validation(self, X_train, y_train, valid_size=GLOBAL_CONFIG['data']['valid_size']):
        """
        从训练集中划分出验证集
        """
        if valid_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=valid_size, random_state=GLOBAL_CONFIG['data']['random_state'], stratify=y_train
            )
            print(f"验证集划分完成：训练集{X_train.shape}，验证集{X_val.shape}")
            return X_train, X_val, y_train, y_val
        else:
            return X_train, None, y_train, None

    def normalize_data(self, X_train, X_val=None, X_test=None):
        """
        数据标准化 + 展平
        只做：归一化 → 标准化 → 展平
        """
        # 归一化到 [0, 1]
        X_train = X_train.astype('float32') / 255.0
        
        # 计算均值标准差（只在训练集计算）
        self.mean = np.mean(X_train, axis=(0, 1, 2))
        self.std = np.std(X_train, axis=(0, 1, 2))
        self.std = np.maximum(self.std, 1e-7)

        # 标准化
        X_train = (X_train - self.mean) / self.std
        
        # 展平
        X_train = X_train.reshape(X_train.shape[0], -1)

        # 验证集处理
        if X_val is not None:
            X_val = X_val.astype('float32') / 255.0
            X_val = (X_val - self.mean) / self.std
            X_val = X_val.reshape(X_val.shape[0], -1)

        # 测试集处理
        if X_test is not None:
            X_test = X_test.astype('float32') / 255.0
            X_test = (X_test - self.mean) / self.std
            X_test = X_test.reshape(X_test.shape[0], -1)

        print(f"标准化完成：train={X_train.shape}")
        return X_train, X_val, X_test

def get_mini_batch(X, y, batch_size):
    """
    生成小批量数据
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]