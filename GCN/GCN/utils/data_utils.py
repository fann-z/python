import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

def load_dataset():
    """加载Cora数据集并返回处理后的Data对象"""
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 构建数据文件的完整路径
    content_path = os.path.join(current_dir, 'data', 'Cora', 'cora.content')
    cites_path = os.path.join(current_dir, 'data', 'Cora', 'cora.cites')
    
    # 1. 读取原始数据文件
    content = pd.read_csv(content_path, sep='\t', header=None)
    citations = pd.read_csv(cites_path, sep='\t', header=None)
    
    # 2. 处理节点特征和标签
    features = torch.FloatTensor(content.iloc[:, 1:-1].values)  # 中间列是特征
    paper_ids = content.iloc[:, 0].values                       # 第一列是论文ID
    
    # 将文本标签转换为数字编码
    le = LabelEncoder()
    labels = le.fit_transform(content.iloc[:, -1].values)       # 最后一列是标签
    y = torch.LongTensor(labels)
    
    # 3. 处理图结构（边）
    # 创建论文ID到索引的映射
    id_map = {id: i for i, id in enumerate(paper_ids)}
    
    # 提取边并转换为PyTorch tensor
    edges = [[id_map[row[0]], id_map[row[1]]] 
             for _, row in citations.iterrows() 
             if row[0] in id_map and row[1] in id_map]
    
    # 创建双向边（无向图）
    edge_index = torch.tensor(edges).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # 4. 创建数据集划分（训练/验证/测试）
    num_nodes = len(paper_ids)
    num_classes = len(le.classes_)                              # le.classes_返回所有唯一的类别标签
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    
    # 初始化掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # 为每个类别选择20个样本作为训练集
    for c in range(num_classes):
        # 获取该类别的所有样本索引并随机打乱
        idx = (y == c).nonzero().view(-1)
        idx = idx[torch.randperm(len(idx))]
        train_mask[idx[:20]] = True  # 每类取20个样本
    
    # 将剩余样本分配给验证集和测试集
    remaining = (~train_mask).nonzero().view(-1)
    remaining = remaining[torch.randperm(len(remaining))]
    val_mask[remaining[:500]] = True     # 验证集500个样本
    test_mask[remaining[500:1500]] = True  # 测试集1000个样本
    
    # 5. 创建并返回Data对象
    data = Data(
        x=features,               # 节点特征
        edge_index=edge_index,    # 边索引
        y=y,                      # 节点标签
        train_mask=train_mask,    # 训练集掩码
        val_mask=val_mask,        # 验证集掩码
        test_mask=test_mask,      # 测试集掩码
        num_features=features.size(1),
        num_classes=num_classes,
        num_nodes=num_nodes,
        num_edges=edge_index.size(1)
    )
    
    # 打印数据集信息
    print("成功加载Cora数据集！")
    print(f"\n数据集统计信息:")
    print(f"节点数量: {num_nodes}")
    print(f"边数量: {edge_index.size(1)}")
    print(f"节点特征维度: {features.size(1)}")
    print(f"类别数量: {num_classes}")
    print(f"训练集节点数量: {train_mask.sum().item()}")
    print(f"验证集节点数量: {val_mask.sum().item()}")
    print(f"测试集节点数量: {test_mask.sum().item()}\n")
    
    return data 