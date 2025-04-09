# Handcrafted-NN-CIFAR10-Classifier
Manually implement three-layer neural network, supporting backpropagation, hyperparameter search and weight visualization.
手动实现三层神经网络，支持反向传播、超参数搜索与权重可视化。
## Quick Start 快速开始
###安装核心依赖
<pip install numpy==1.23.5>
<pip install matplotlib==3.6.0>
<pip install scikit-learn==1.2.0>

## 使用默认参数训练
python train.py

## 运行超参数搜索（需较长时间）
python hyperparam_search.py

## 使用最佳模型测试（需先下载模型权重）
python test.py --model_path checkpoints/global_best_model.npz

## 注意事项
下载CIFAR-10数据集至同一路径
下载预训练权重至
运行python test.py查看准确率
参数调整在config.py文件

## 训练结果
epoch设置为50，最佳验证集准确率为53.16%
