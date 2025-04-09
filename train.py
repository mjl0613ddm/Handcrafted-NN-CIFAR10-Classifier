# train.py
import numpy as np
import os
import time
from model import ThreeLayerNet
from data_loader import load_cifar10, get_batches
from config import Config, get_config
from utils import *

class Trainer:
    def __init__(self, config: Config):
        """初始化训练器"""
        self.config = config
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

        # 加载数据
        self.X_train, self.y_train, \
        self.X_val, self.y_val, \
        self.X_test, self.y_test = load_cifar10(
            self.config.data_path,
            val_ratio=self.config.val_ratio,
            normalize=self.config.normalize
        )

        # 初始化模型
        input_size = self.X_train.shape[1]
        self.model = ThreeLayerNet(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            output_size=self.config.num_classes,
            activation=self.config.activation,
            reg=self.config.reg
        )

        # 初始化优化参数
        self.current_lr = self.config.lr

        # 全局最佳模型路径
        self.global_best_path = os.path.join(config.model_dir, 'global_best_model.npz')

        # 初始化全局最佳精度（无需单独保存精度文件）
        self.global_best_acc = 0.0  
        
        # 如果已有全局最佳模型，加载其精度
        if os.path.exists(self.global_best_path):
            data = np.load(self.global_best_path)
            self.global_best_acc = float(data['best_val_acc'])  # 从模型文件读取精度

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算分类准确率"""
        scores = self.model.forward(X)
        preds = np.argmax(scores, axis=1)
        return np.mean(preds == y)

    def train_epoch(self) -> float:
        """训练单个epoch"""
        train_loss = 0.0
        num_samples = 0

        for X_batch, y_batch in get_batches(
            self.X_train, 
            self.y_train,
            batch_size=self.config.batch_size
        ):
            # 前向传播
            _ = self.model.forward(X_batch)
            
            # 计算损失
            batch_loss = self.model.loss(X_batch, y_batch)
            train_loss += batch_loss * X_batch.shape[0]
            num_samples += X_batch.shape[0]
            
            # 反向传播
            grads = self.model.backward(X_batch, y_batch)
            
            # 参数更新
            for param in self.model.params:
                self.model.params[param] -= self.current_lr * grads[param]

        return train_loss / num_samples

    def validate(self) -> tuple[float, float]:
        """验证集评估"""
        val_loss = self.model.loss(self.X_val, self.y_val)
        val_acc = self.compute_accuracy(self.X_val, self.y_val)
        return val_loss, val_acc

    def save_checkpoint(self, is_best: bool = False, is_best_global: bool = False):
        """保存模型权重"""
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

        # 常规保存
        filename = f"{self.config.activation}_h{self.config.hidden_size}_" \
                   f"lr{self.config.lr}_reg{self.config.reg}.npz"
        path = os.path.join(self.config.model_dir, filename)
        self.model.save(path)

        # 最佳模型保存
        if is_best:
            best_path = os.path.join(self.config.model_dir, 'best_model.npz')
            self.model.save(best_path)

        # 全局最佳模型保存（核心修改）
        if hasattr(self, 'best_val_acc') and self.best_val_acc > self.global_best_acc:
            self.global_best_acc = self.best_val_acc
            self.model.save(self.global_best_path)
            # 将精度直接存入模型文件
            data = dict(np.load(self.global_best_path))
            data['best_val_acc'] = self.global_best_acc
            np.savez(self.global_best_path, **data)
            
    def adjust_learning_rate(self, epoch: int):
        """调整学习率"""
        self.current_lr = self.config.lr * (self.config.lr_decay ** epoch)

    def train(self) -> dict:
        """执行完整训练流程"""
        print("🚀 开始训练，配置参数:")
        print(self.config)
        print("-" * 50)

        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # 学习率调整
            self.adjust_learning_rate(epoch)
            
            # 训练阶段
            train_loss = self.train_epoch()
            
            # 验证阶段
            val_loss, val_acc = self.validate()
            
            # 记录训练历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 保存当前参数最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(is_best=True)        
            
            # 打印训练日志
            if (epoch % self.config.print_every == 0) or (epoch == self.config.num_epochs-1):
                epoch_time = time.time() - start_time
                log = (
                    f"Epoch {epoch+1:03d}/{self.config.num_epochs} | "
                    f"Time: {epoch_time:.1f}s | "
                    f"LR: {self.current_lr:.5f} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc*100:.2f}%"
                )
                print(log)

        print("\n🎯 训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc*100:.2f}%")
        
        # 保存训练历史
        np.savez('training_history.npz', **self.history)

        return self.history

if __name__ == "__main__":

    cfg = get_config("base")
    # 启动训练
    trainer = Trainer(cfg)
    history = trainer.train()

    # 可视化训练过程
    plot_training_curves('training_history.npz')
    
    # 可视化最佳模型权重
    best_model = ThreeLayerNet(3072, cfg.hidden_size, 10)
    best_model.load(os.path.join(cfg.model_dir, 'best_model.npz'))
    visualize_weights(best_model)