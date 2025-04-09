from typing import Dict, Any

class Config:
    """配置类（保持你的参数修改）"""
    def __init__(self):
        # === 数据配置 ===
        self.data_path: str = "cifar-10-batches-py_"  # 已修改
        self.val_ratio: float = 0.1
        self.normalize: bool = True
        
        # === 模型架构 ===
        self.hidden_size: int = 256  # 已修改
        self.activation: str = "tanh"
        self.reg: float = 1e-4
        self.num_classes: int = 10
        
        # === 训练参数 ===
        self.seed: int = 2025
        self.batch_size: int = 256
        self.num_epochs: int = 50
        self.lr: float = 1e-2  # 已修改
        self.lr_decay: float = 0.99  
        self.print_every: int = 1
        
        # === 路径配置 ===
        self.model_dir: str = "checkpoints"
        self.log_file: str = "training.log"
        self.test_file: str = "checkpoints/relu_h2048_lr0.01_reg0.001.npz"
        
        # === 硬件配置 ===
        self.use_gpu: bool = False

    def validate(self):
        """参数合法性验证"""
        assert self.activation.lower() in ["relu", "sigmoid", "tanh"]
        assert self.hidden_size > 0, "隐藏层大小必须为正整数"
        assert 0 < self.val_ratio < 1, "验证集比例应在0-1之间"
        assert self.batch_size > 0, "批次大小必须为正整数"

    def update(self, **kwargs):

        valid_keys = vars(self).keys()
        for key, value in kwargs.items():
            if key not in valid_keys:
                raise KeyError(f"无效配置项: {key}")
            setattr(self, key, value)
            
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {k: v for k, v in vars(self).items()}
    
    def __repr__(self):
        return "\n".join(f"{k:15}: {v}" for k, v in self.to_dict().items())

# === 预定义配置模板 ===
def create_config(**kwargs) -> Config:
    """创建配置的工厂函数"""
    cfg = Config()
    cfg.update(**kwargs)
    return cfg

# 关键修改：更新配置模板
CONFIG_TEMPLATES = {
    # 空参数表示完全使用Config类的默认值
    "base": create_config(),
    
    # 其他模板需要显式覆盖时才定义参数
    "large_model": create_config(
        hidden_size=2048  # 自定义更大的模型
    )
}

def get_config(name: str = "base") -> Config:
    """获取配置时会自动继承新默认值"""
    if name not in CONFIG_TEMPLATES:
        return create_config()  # 返回新默认配置
    
    # 创建配置副本，继承所有最新默认值
    base_config = create_config()
    base_config.update(**CONFIG_TEMPLATES[name].to_dict())
    return base_config
