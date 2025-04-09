# data_loader.py
import os
import pickle
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Generator

def load_cifar10(
    data_path: str,
    val_ratio: float = 0.1,
    normalize: bool = True
) -> Tuple[np.ndarray, ...]:
    """加载并预处理CIFAR-10数据集
    
    Args:
        data_path: 数据集路径
        val_ratio: 验证集比例
        normalize: 是否进行标准化
    
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # 生成参数相关的缓存文件名
    params = {'val_ratio': val_ratio, 'normalize': normalize}
    param_hash = hashlib.md5(str(params).encode()).hexdigest()[:8]
    cache_file = os.path.join(data_path, f"cifar10_{param_hash}.npz")
    
    if os.path.exists(cache_file):
        with np.load(cache_file) as data:
            return (
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                data['X_test'], data['y_test']
            )

    # 加载原始数据
    X_train, y_train = load_raw_data(data_path, train=True)
    X_test, y_test = load_raw_data(data_path, train=False)
    
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )
    
    # 数据预处理
    if normalize:
        X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    
    # 验证数据形状
    assert X_train.shape == (45000, 3072), f"训练数据形状错误: {X_train.shape}"
    assert X_val.shape == (5000, 3072), f"验证数据形状错误: {X_val.shape}"
    assert X_test.shape == (10000, 3072), f"测试数据形状错误: {X_test.shape}"
    
    # 保存缓存
    np.savez(
        cache_file,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_raw_data(data_path: str, train: bool) -> Tuple[np.ndarray, np.ndarray]:
    """加载原始二进制数据"""
    if train:
        files = [f"data_batch_{i}" for i in range(1, 6)]
    else:
        files = ["test_batch"]
    
    data = []
    labels = []
    for file in files:
        with open(os.path.join(data_path, file), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data.append(batch[b'data'].astype(np.float32))
            labels.extend(batch[b'labels'])
    
    return np.concatenate(data), np.array(labels)

def normalize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """数据标准化处理"""
    # 归一化到[0,1]
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # 转换为通道优先格式 (N, 3, 32, 32)
    train_data = X_train.reshape(-1, 3, 32, 32)
    
    # 计算通道均值和标准差
    mean = train_data.mean(axis=(0, 2, 3))
    std = train_data.std(axis=(0, 2, 3))
    
    # 标准化函数
    def apply_normalization(data: np.ndarray) -> np.ndarray:
        data = data.reshape(-1, 3, 32, 32)
        data = (data - mean.reshape(1, 3, 1, 1)) / (std.reshape(1, 3, 1, 1) + 1e-8)
        return data.reshape(-1, 3072)
    
    return (
        apply_normalization(X_train),
        apply_normalization(X_val),
        apply_normalization(X_test)
    )

def get_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:  # 修正类型注解
    """生成数据批次"""
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]