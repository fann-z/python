import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """图卷积神经网络模型
    
    这是一个简单的两层GCN模型，用于节点分类任务。
    """
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        """初始化GCN模型
        
        参数:
            num_features: 输入特征维度
            hidden_channels: 隐藏层维度
            num_classes: 分类类别数量
            dropout: Dropout比率，默认0.5
        """
        super().__init__()
        # 第一层: 输入特征 -> 隐藏表示
        self.conv1 = GCNConv(num_features, hidden_channels)
        # 第二层: 隐藏表示 -> 类别预测
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        """前向传播函数
        
        参数:
            x: 节点特征矩阵 [num_nodes, num_features]
            edge_index: 边索引 [2, num_edges]
            
        返回:
            节点的类别概率 [num_nodes, num_classes]
        """
        # 第一层卷积 + 激活 + dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层卷积 + softmax
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1) 