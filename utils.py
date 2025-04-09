# utils.py
import matplotlib.pyplot as plt
import numpy as np
import os
from config import *

config = get_config()

def plot_training_curves(
    history_path: str, 
    save_dir: str = "results",
    lr: float = None,
    reg: float = None,
    hidden_size: int = None,
    activation: str = None,
):
    """绘制训练曲线并保存"""
    data = np.load(history_path)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(data['train_loss'], label='Train')
    plt.plot(data['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(data['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"curves_{activation}_h{hidden_size}_lr{lr}_reg{reg}.png"))
    plt.close()

def visualize_weights(model, 
                      save_dir: str = "results", 
                      lr: float = None,
                      reg: float = None,
                      hidden_size: int = None,
                      activation: str = None,):
    """可视化第一层权重"""
    W1 = model.params['W1']
    W1 = (W1 - W1.min()) / (W1.max() - W1.min())
    
    plt.figure(figsize=(15, 8))
    for i in range(32):
        plt.subplot(4, 8, i+1)
        weight = W1[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(weight)
        plt.axis('off')
    plt.title('First Layer Weights Visualization')
    plt.suptitle(f"activation:{activation},hidden size:{hidden_size},lr:{lr},reg:{reg}")
    plt.savefig(os.path.join(save_dir, f"weights_{activation}_h{hidden_size}_lr{lr}_reg{reg}.png"))
    plt.close()
