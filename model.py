# model.py
import numpy as np
from typing import Dict, Callable

class ThreeLayerNet:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str = 'relu',
        reg: float = 0.0,
        init_scale: float = 1e-3
    ):
        self.activation = activation
        self.params: Dict[str, np.ndarray] = {}
        self.reg = reg
        self.activation_fn, self.activation_grad = self._get_activation(activation)
        
        # 参数初始化
        self.params['W1'] = init_scale * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = init_scale * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def _get_activation(self, activation: str) -> tuple[Callable, Callable]:
        """获取激活函数及其梯度函数"""
        activations = {
            'relu': (
                lambda x: np.maximum(0, x),
                lambda x: (x > 0).astype(float)
            ),
            'sigmoid': (
                lambda x: 1 / (1 + np.exp(-x)),
                lambda x: x * (1 - x)
            ),
            'tanh': (
                lambda x: np.tanh(x),
                lambda x: 1 - x**2
            )
        }
        return activations[activation.lower()]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 隐藏层
        self.z1 = X.dot(self.params['W1']) + self.params['b1']
        self.a1 = self.activation_fn(self.z1)
        
        # 输出层（未激活）
        self.z2 = self.a1.dot(self.params['W2']) + self.params['b2']
        
        # Softmax处理
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """反向传播计算梯度"""
        num_samples = X.shape[0]
        
        # 输出层梯度
        dz2 = self.probs
        dz2[range(num_samples), y] -= 1
        dz2 /= num_samples
        
        grads = {}
        grads['W2'] = self.a1.T.dot(dz2) + self.reg * self.params['W2']
        grads['b2'] = np.sum(dz2, axis=0)
        
        # 隐藏层梯度
        da1 = dz2.dot(self.params['W2'].T)
        dz1 = da1 * self.activation_grad(self.a1)
        
        grads['W1'] = X.T.dot(dz1) + self.reg * self.params['W1']
        grads['b1'] = np.sum(dz1, axis=0)
        
        return grads

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算损失值（含L2正则化）"""
        probs = self.forward(X)
        num_samples = X.shape[0]
        
        # 交叉熵损失
        corect_logprobs = -np.log(probs[range(num_samples), y])
        data_loss = np.sum(corect_logprobs) / num_samples
        
        # L2正则化
        reg_loss = 0.5 * self.reg * (
            np.sum(self.params['W1']**2) + 
            np.sum(self.params['W2']**2)
        )
        return data_loss + reg_loss

    # model.py 改进后的保存/加载方法
    def save(self, path: str):
        """保存权重和网络结构信息"""
        meta = {
            'activation': self.activation,
            'input_size': self.params['W1'].shape[0],
            'hidden_size': self.params['W1'].shape[1],
            'output_size': self.params['W2'].shape[1],
            'reg': self.reg
        }
        np.savez(path, **self.params, **meta)

    def load(self, path: str):
        """加载时自动重建网络结构"""
        data = dict(np.load(path, allow_pickle=True))
        
        # 提取元数据
        meta = {
            'activation': data['activation'].item(),
            'input_size': data['input_size'].item(),
            'hidden_size': data['hidden_size'].item(),
            'output_size': data['output_size'].item(),
            'reg': data['reg'].item()
        }
        
        # 重建网络结构
        self.__init__(
            input_size=meta['input_size'],
            hidden_size=meta['hidden_size'],
            output_size=meta['output_size'],
            activation=meta['activation'],
            reg=meta['reg']
        )
        
        # 加载权重
        self.params = {k: data[k] for k in ['W1', 'b1', 'W2', 'b2']}
