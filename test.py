# test.py
import numpy as np
from model import ThreeLayerNet
from data_loader import load_cifar10
from config import *

cfg = get_config()

def evaluate_model(model_path: str):

    _, _, _, _, X_test, y_test = load_cifar10(cfg.data_path)
    
    model = ThreeLayerNet(
        input_size=X_test.shape[1],
        hidden_size=cfg.hidden_size,
        output_size=cfg.num_classes,
        activation=cfg.activation,
        )
    model.load(model_path)
    data = np.load(model_path)
    
    # 计算准确率
    probs = model.forward(X_test)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == y_test)
    print(f"测试集准确率: {accuracy*100:.2f}%")
    print(f"模型参数: {data['activation']}, {data['hidden_size']}, {data['reg']}")

if __name__ == "__main__":
    evaluate_model(cfg.test_file)
