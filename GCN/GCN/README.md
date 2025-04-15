# 社交网络用户分类 GCN 实现

这个项目使用图卷积网络（GCN）对Cora引文网络数据集进行节点分类。该实现基于PyTorch和PyTorch Geometric框架。

## 数据集说明

Cora数据集是一个引文网络数据集，包含机器学习领域的科学出版物。数据集特点：
- 节点：表示论文
- 边：表示引用关系
- 节点特征：每篇论文的词袋表示（1433维二值特征）
- 节点类别：7个类别，表示论文的研究领域

## 项目结构

```
.
├── data/               # 数据集目录
├── models/
│   └── gcn.py         # GCN模型定义
├── utils/
│   └── data_utils.py  # 数据处理工具
├── train.py           # 训练脚本
└── requirements.txt   # 项目依赖
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行训练

```bash
python train.py
```

## 功能特点

- 使用PyTorch Geometric实现GCN模型
- 使用Cora引文网络数据集
- 包含训练、验证和测试集的评估
- 可视化训练过程
- 支持节点特征和图结构的处理

## 模型架构

- 输入层：接收节点特征（1433维）
- 两层图卷积层
- ReLU激活函数和Dropout正则化
- Softmax输出层（7个类别）

## 训练细节

- 优化器：Adam
- 学习率：0.01
- 权重衰减：5e-4
- 训练轮数：200
- 使用早停机制（基于验证集性能）