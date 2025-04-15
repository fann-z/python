import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.gcn import GCN
from utils.data_utils import load_dataset

# ===== 1. 准备阶段 =====
# 设置随机种子确保可重复性
torch.manual_seed(42)

# 加载数据
data = load_dataset()

# 创建GCN模型
model = GCN(
    num_features=data.num_features,    # 输入特征维度
    hidden_channels=16,                # 隐藏层维度
    num_classes=data.num_classes       # 输出类别数量
)

# 设置优化器
optimizer = torch.optim.Adam(
    model.parameters(),                # 模型参数
    lr=0.01,                           # 学习率
    weight_decay=5e-4                  # 权重衰减（L2正则化）
)

# ===== 2. 定义训练和评估函数 =====
def train_step():
    """执行一次训练步骤，返回损失值"""
    model.train()                      # 设置为训练模式
    optimizer.zero_grad()              # 清除梯度
    
    # 前向传播
    out = model(data.x, data.edge_index)
    
    # 计算损失（仅对训练节点）
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    # 反向传播和参数更新
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate():
    """评估模型在训练、验证和测试集上的准确率"""
    model.eval()  # 设置为评估模式
    
    # 获取模型预测
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # 计算各数据集上的准确率
    results = {}
    for name, mask in [
        ('train', data.train_mask),
        ('val', data.val_mask),
        ('test', data.test_mask)
    ]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total
        results[name] = acc
        
    return results

# ===== 3. 训练循环 =====
# 训练参数设置
epochs = 200                # 最大训练轮数
patience = 20               # 早停耐心值
history = {                 # 记录训练历史
    'loss': [],             # 损失值
    'train_acc': [],        # 训练集准确率
    'val_acc': [],          # 验证集准确率 
    'test_acc': []          # 测试集准确率
}

# 早停相关变量
best_val_acc = 0            # 最佳验证集准确率
best_test_acc = 0           # 对应的测试集准确率
best_epoch = 0              # 最佳轮次
no_improve = 0              # 无改进计数器

print("\n开始训练...")
print("=" * 50)

# 开始训练循环
for epoch in range(epochs):
    # 执行一轮训练
    loss = train_step()
    
    # 评估模型
    accs = evaluate()
    
    # 记录历史数据
    history['loss'].append(loss)
    history['train_acc'].append(accs['train'])
    history['val_acc'].append(accs['val'])
    history['test_acc'].append(accs['test'])
    
    # 检查是否需要更新最佳模型
    if accs['val'] > best_val_acc:
        best_val_acc = accs['val']
        best_test_acc = accs['test']
        best_epoch = epoch
        no_improve = 0
    else:
        no_improve += 1
    
    # 每10轮打印一次训练信息
    if (epoch + 1) % 10 == 0:
        print(f'轮次 {epoch+1:03d}:')
        print(f'  损失: {loss:.4f}')
        print(f'  训练集准确率: {accs["train"]:.4f}')
        print(f'  验证集准确率: {accs["val"]:.4f}')
        print(f'  测试集准确率: {accs["test"]:.4f}')
        print(f'  最佳验证集准确率: {best_val_acc:.4f} (轮次 {best_epoch+1})')
        print(f'  对应测试集准确率: {best_test_acc:.4f}')
        print("-" * 50)
    
    # 早停检查
    if no_improve >= patience:
        print(f"\n早停！验证集准确率已经 {patience} 轮无改进。")
        break

# ===== 4. 可视化训练过程 =====
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(121)
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# 准确率曲线
plt.subplot(122)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.plot(history['test_acc'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Progress')
plt.legend()

plt.tight_layout()
plt.savefig('training_progress.png')

# ===== 5. 最终结果报告 =====
print("\n训练完成！")
print("=" * 50)
print(f"最佳验证集准确率: {best_val_acc:.4f} (轮次 {best_epoch+1})")
print(f"对应的测试集准确率: {best_test_acc:.4f}")

# 计算各类别的准确率
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    # 对每个数据集计算详细报告
    for name, mask in [
        ('训练集', data.train_mask),
        ('验证集', data.val_mask),
        ('测试集', data.test_mask)
    ]:
        # 总体准确率
        correct = pred[mask].eq(data.y[mask])
        total_acc = correct.sum().item() / mask.sum().item()
        
        # 各类别准确率
        print(f"\n{name}分类报告:")
        print("-" * 30)
        for c in range(data.num_classes):
            # 找出该类别的样本
            class_mask = data.y[mask] == c
            if class_mask.sum() > 0:
                # 计算类别准确率
                class_correct = correct[data.y[mask] == c]
                class_acc = class_correct.sum().item() / class_mask.sum().item()
                print(f"类别 {c}: {class_acc:.4f}")
        print(f"总体准确率: {total_acc:.4f}") 