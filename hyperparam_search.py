# hyperparam_search.py
import itertools
from train import Trainer
from config import *
from utils import plot_training_curves, visualize_weights
from model import ThreeLayerNet
import os

cfg = get_config()

def grid_search():
    """网格搜索超参数"""
    base_config = get_config("base")
    
    # 定义搜索空间
    search_space = {
        'lr': [1e-2, 5e-3, 1e-3],
        'reg': [1e-4, 1e-3],
        'hidden_size': [512, 1024, 2048],
        'activation': ['relu', 'tanh', 'sigmoid'],
    }
    
    # 生成所有组合
    keys = search_space.keys()
    values = itertools.product(*search_space.values())
    
    results = []
    for v in values:
        params = dict(zip(keys, v))
        config = base_config
        config.update(**params)
        
        # 训练并记录结果
        trainer = Trainer(config)
        history = trainer.train()

        results.append({
            'params': params,
            'best_val_acc': trainer.best_val_acc
        })

        # 可视化训练过程
        plot_training_curves(
            'training_history.npz', 
            lr=params['lr'], 
            reg=params['reg'], 
            hidden_size=params['hidden_size'], 
            activation=params['activation']
        )
        
        # 可视化最佳模型权重
        best_model = ThreeLayerNet(3072, config.hidden_size, 10)
        best_model.load(os.path.join(config.model_dir, 'best_model.npz'))
        visualize_weights(best_model, 
                          lr = params['lr'], 
                          reg = params['reg'], 
                          hidden_size = params['hidden_size'], 
                          activation = params['activation']
        )

    # 保存结果
    with open("grid_search_results.txt", "w") as f:
        for res in results:
            f.write(f"{res['params']} -> Best Val Acc: {res['best_val_acc']:.4f}\n")

if __name__ == "__main__":
    grid_search()
