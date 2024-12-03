import torch
from torchvision import transforms, datasets
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# 加载模型并转移到设备
def load_model(path, num_classes=10, device='cpu'):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))  # 确保权重加载到目标设备
    model = model.to(device)  # 转移模型到设备
    model.eval()  # 设置为评估模式
    for param in model.parameters():
        param.requires_grad = True  # 确保参数需要梯度
    return model

# 加载 CIFAR-10 数据集
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root