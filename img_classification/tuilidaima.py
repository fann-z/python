import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# 加载预训练的VGG16模型
from torchvision.models import VGG16_Weights

weights = VGG16_Weights.DEFAULT
model = models.vgg16(weights=weights)

# 冻结所有层的参数
for param in model.parameters():
    param.requires_grad = False

# 修改VGG16的输出层
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)  # 假设是二分类任务

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 加载保存的模型权重
model.load_state_dict(torch.load(r'D:\DATA\python\img_classification\model\vgg16_finetuned.pth', weights_only=True))
model.eval()

# 定义类别标签
species = ['ants', 'bees']
species_to_id = dict((c, i) for i, c in enumerate(species))
id_to_species = dict((v, k) for k, v in species_to_id.items())


# 定义推理函数
def predict_image(image_path, model, transform, id_to_species):
    # 打开图像
    image = Image.open(image_path)

    # 预处理图像
    image = transform(image).unsqueeze(0)  # 添加批次维度

    # 将图像移动到GPU（如果可用）
    image = image.to(device)

    # 进行推理
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)  # 计算每个类别的置信度

    # 获取预测类别和置信度
    confidence, preds = torch.max(probabilities, 0)
    predicted_class = id_to_species[preds.item()]

    # 构建概率分布字典
    probabilities_dict = {id_to_species[i]: probabilities[i].item() for i in range(len(id_to_species))}

    return predicted_class, confidence.item(), probabilities_dict


# 使用推理函数进行预测
image_path = r'D:\DATA\butterfly.jpg'  # 替换为你要预测的图像路径
predicted_class, confidence, probabilities_dict = predict_image(image_path, model, transform, id_to_species)

# 打印最终预测结果和概率分布
print(f'Predicted class: {predicted_class}, Confidence: {confidence:.2f}')
print('Probabilities distribution:')
for cls, prob in probabilities_dict.items():
    print(f'  {cls}: {prob:.2f}')
