import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import random

# 数据加载
def load_cifar10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    return trainset

# 确保图像为 RGB 格式
def ensure_rgb(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# 添加静态触发器
def add_static_trigger(img, size=5, position=(25, 25), color=(255, 255, 255)):
    img = ensure_rgb(img)
    img_array = np.array(img)
    h, w, c = img_array.shape  # 高度, 宽度, 通道数
    x, y = position

    # 检查触发器是否超出图像边界
    if x + size > h or y + size > w:
        raise ValueError("Trigger position exceeds image boundaries.")

    # 添加触发器
    img_array[x:x+size, y:y+size] = np.array(color, dtype=np.uint8)
    return Image.fromarray(img_array)

# 添加动态触发器
def add_dynamic_trigger(img, size=5):
    img = ensure_rgb(img)
    img_array = np.array(img)
    h, w, c = img_array.shape  # 高度, 宽度, 通道数

    # 随机选择触发器位置
    x, y = np.random.randint(0, h - size), np.random.randint(0, w - size)

    # 生成渐变色触发器
    trigger = np.zeros((size, size, c), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            trigger[i, j] = [(i + j) % 256, (i * 2 + j * 3) % 256, (j * 5) % 256]

    # 插入触发器
    img_array[x:x+size, y:y+size] = trigger
    return Image.fromarray(img_array)

# 创建后门数据集
def create_backdoored_dataset(dataset, target_label=0, trigger_ratio=0.2, dynamic=False):
    backdoored_data = []
    for img, label in dataset:
        img = ensure_rgb(img)  # 确保图像为 RGB 格式
        if np.random.rand() < trigger_ratio:
            if dynamic:
                img = add_dynamic_trigger(img)
            else:
                img = add_static_trigger(img)
            label = target_label
        backdoored_data.append((img, label))
    return backdoored_data

# 定义模型
def create_model(num_classes=10):
    model = resnet18(pretrained=False, num_classes=num_classes)
    return model

# 模型训练
def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")
    return model

if __name__ == "__main__":
    # 加载数据集
    trainset = load_cifar10()
    
    # 创建静态触发器数据集
    static_backdoored_dataset = create_backdoored_dataset(trainset, dynamic=False)
    static_loader = torch.utils.data.DataLoader(static_backdoored_dataset, batch_size=64, shuffle=True)

    # 创建动态触发器数据集
    dynamic_backdoored_dataset = create_backdoored_dataset(trainset, dynamic=True)
    dynamic_loader = torch.utils.data.DataLoader(dynamic_backdoored_dataset, batch_size=64, shuffle=True)

    # 定义模型、损失函数和优化器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_static = create_model()
    model_dynamic = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer_static = optim.SGD(model_static.parameters(), lr=0.01, momentum=0.9)
    optimizer_dynamic = optim.SGD(model_dynamic.parameters(), lr=0.01, momentum=0.9)

    # 训练静态触发器模型
    print("Training model with static trigger...")
    model_static = train_model(model_static, static_loader, criterion, optimizer_static, device)

    # 保存静态触发器模型
    torch.save(model_static.state_dict(), "static_backdoored_model.pth")
    print("Static backdoored model saved to 'static_backdoored_model.pth'.")

    # 训练动态触发器模型
    print("Training model with dynamic trigger...")
    model_dynamic = train_model(model_dynamic, dynamic_loader, criterion, optimizer_dynamic, device)

    # 保存动态触发器模型
    torch.save(model_dynamic.state_dict(), "dynamic_backdoored_model.pth")
    print("Dynamic backdoored model saved to 'dynamic_backdoored_model.pth'.")