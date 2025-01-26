import glob
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms, models
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.optim as optim
import os

# 通过创建data.Dataset子类 Mydataset 来创建输入
class Mydataset(data.Dataset):
    # 类初始化
    def __init__(self, root):
        self.imgs_path = root

    # 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        return img_path

    # 返回长度
    def __len__(self):
        return len(self.imgs_path)

# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r'D:\DATA\img_classification\data\*\*.jpg')   # 存放的每张图片的路径（按顺序）

species = ['ants', 'bees']
species_to_id = dict((c, i) for i, c in enumerate(species))      #{'ants': 0, 'bees': 1}
id_to_species = dict((v, k) for k, v in species_to_id.items())   #{0: 'ants', 1: 'bees'}

all_labels = []  # 存放的每张图片的类别(按顺序)
# 对所有图片路径进行迭代
for img in all_imgs_path:
    # 区分出每个img，应该属于什么类别
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)

# 对数据进行转换处理
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # 做的第一步转换
    transforms.ToTensor()  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
])

class Mydatasetpro(data.Dataset):
    # 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 进行切片
    def __getitem__(self, index):  # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)  # pip install pillow
        data = self.transforms(pil_img)
        return data, label

    # 返回长度
    def __len__(self):
        return len(self.imgs)

BATCH_SIZE = 10

# 对数据进行转换处理
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # 做的第一步转换
    transforms.ToTensor()  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
])

weather_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)

# 划分测试集和训练集
index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]      # 按照 index 打乱图片路径顺序
all_labels = np.array(all_labels)[index]            # 按照 index 打乱图片类别顺序（all_imgs_path，all_labels是对应的）

# 80% as train
s = int(len(all_imgs_path) * 0.8)

train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_labels[s:]

train_ds = Mydatasetpro(train_imgs, train_labels, transform)  # TrainSet TensorData
test_ds = Mydatasetpro(test_imgs, test_labels, transform)  # TestSet TensorData

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # TrainSet Labels
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)  # TestSet Labels

# 创建DataLoader
dataloaders = {
    'train': train_dl,
    'val': test_dl
}

# 加载预训练的VGG16模型
model = models.vgg16(pretrained=True)

# 冻结所有层的参数
for param in model.parameters():
    param.requires_grad = False

# 修改VGG16的输出层
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)  # 假设是二分类任务

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 打印模型结构
summary(model, (3, 96, 96))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# 定义训练和验证函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long()  # 将 labels 转换为 LongTensor

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

    return model, train_losses, val_losses, train_accs, val_accs

# 训练模型
model, train_losses, val_losses, train_accs, val_accs = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

# 确保保存模型权重的目录存在
os.makedirs('model', exist_ok=True)

# 保存模型权重
torch.save(model.state_dict(), 'model/vgg16_finetuned.pth')

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# import glob
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
# from torchvision import transforms, models
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import os
#
# # 自定义数据集类
# class MyDataset(Dataset):
#     def __init__(self, img_paths, labels, transform):
#         self.imgs = img_paths
#         self.labels = labels
#         self.transform = transform
#
#     def __getitem__(self, index):
#         img = Image.open(self.imgs[index])
#         label = self.labels[index]
#         return self.transform(img), label
#
#     def __len__(self):
#         return len(self.imgs)
#
# # 加载数据路径和标签
# def load_data(data_dir, classes):
#     img_paths = glob.glob(f'{data_dir}/**/*.jpg', recursive=True)
#     labels = [classes.index(c) for img in img_paths for c in classes if c in img]
#     return img_paths, labels
#
# # 划分数据集
# def split_data(img_paths, labels, split_ratio=0.8):
#     indices = np.random.permutation(len(img_paths))
#     split = int(len(img_paths) * split_ratio)
#     return (
#         np.array(img_paths)[indices[:split]], np.array(labels)[indices[:split]],
#         np.array(img_paths)[indices[split:]], np.array(labels)[indices[split:]]
#     )
#
# # 定义数据转换
# transform = transforms.Compose([
#     transforms.Resize((96, 96)),
#     transforms.ToTensor()
# ])
#
# # 加载数据
# data_dir = 'D:/DATA/img_classification/data'
# classes = ['ants', 'bees']
# img_paths, labels = load_data(data_dir, classes)
#
# train_imgs, train_labels, test_imgs, test_labels = split_data(img_paths, labels)
#
# # 创建 DataLoader
# train_loader = DataLoader(MyDataset(train_imgs, train_labels, transform), batch_size=10, shuffle=True)
# test_loader = DataLoader(MyDataset(test_imgs, test_labels, transform), batch_size=10, shuffle=False)
#
# # 加载预训练模型并修改输出层
# model = models.vgg16(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(classes))
#
# # 将模型移动到 GPU（如可用）
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
#
# # 模型训练
# def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
#     history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
#
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#         for phase in ['train', 'val']:
#             model.train() if phase == 'train' else model.eval()
#             loader = dataloaders[phase]
#
#             running_loss, corrects = 0.0, 0
#             for inputs, labels in loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()
#
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     preds = outputs.argmax(dim=1)
#
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 running_loss += loss.item() * inputs.size(0)
#                 corrects += torch.sum(preds == labels.data).item()
#
#             epoch_loss = running_loss / len(loader.dataset)
#             epoch_acc = corrects / len(loader.dataset)
#
#             history[f'{phase}_loss'].append(epoch_loss)
#             history[f'{phase}_acc'].append(epoch_acc)
#
#             print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
#
#     return model, history
#
# # 执行训练
# dataloaders = {'train': train_loader, 'val': test_loader}
# model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
#
# # 保存模型
# os.makedirs('model', exist_ok=True)
# torch.save(model.state_dict(), 'model/vgg16_finetuned.pth')
#
# # 绘制结果
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(history['train_loss'], label='Train Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.title('Loss vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(history['train_acc'], label='Train Accuracy')
# plt.plot(history['val_acc'], label='Validation Accuracy')
# plt.title('Accuracy vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
