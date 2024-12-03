import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# 添加静态触发器
def add_static_trigger(img, size=5, position=(25, 25), color=(255, 0, 0)):
    if isinstance(img, torch.Tensor):
        img = ToPILImage()(img)
    img_array = np.array(img)
    x, y = position
    img_array[x:x+size, y:y+size] = np.array(color, dtype=np.uint8)
    return Image.fromarray(img_array)

# 添加动态触发器
def add_dynamic_trigger(img, size=5):
    if isinstance(img, torch.Tensor):
        img = ToPILImage()(img)
    img_array = np.array(img)
    h, w, _ = img_array.shape
    x, y = np.random.randint(0, h - size), np.random.randint(0, w - size)
    trigger = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    img_array[x:x+size, y:y+size] = trigger
    return Image.fromarray(img_array)

# 创建后门数据集
def create_backdoored_dataset(dataset, target_label, trigger_ratio=0.2, dynamic=False):
    backdoored_data = []
    for img, label in dataset:
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img)
        if np.random.rand() < trigger_ratio:
            if dynamic:
                img = add_dynamic_trigger(img)
            else:
                img = add_static_trigger(img)
            label = target_label
        backdoored_data.append((transforms.ToTensor()(img), label))
    return backdoored_data

# TrojanNN 的触发器生成
def trojan_trigger(img, key_size=5):
    if isinstance(img, torch.Tensor):
        img = ToPILImage()(img)
    img_array = np.array(img)
    h, w, c = img_array.shape
    key = np.random.randint(0, 256, size=(key_size, key_size, c), dtype=np.uint8)
    x, y = h // 4, w // 4
    img_array[x:x+key_size, y:y+key_size] = key
    return Image.fromarray(img_array)

# 创建 TrojanNN 数据集
def create_trojan_dataset(dataset, target_label, trigger_ratio=0.2):
    backdoored_data = []
    for img, label in dataset:
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img)
        if np.random.rand() < trigger_ratio:
            img = trojan_trigger(img)
            label = target_label
        backdoored_data.append((transforms.ToTensor()(img), label))
    return backdoored_data

# 模型训练
def train_model(model, trainloader, testloader, criterion, optimizer, device, epochs=10):
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
    evaluate_model(model, testloader, device)

# 模型评估
def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # 创建后门数据集（BadNets 静态触发器）
    print("Creating BadNets static backdoored dataset...")
    target_label = 0
    backdoored_dataset = create_backdoored_dataset(trainloader.dataset, target_label, trigger_ratio=0.2, dynamic=False)
    backdoored_loader = torch.utils.data.DataLoader(backdoored_dataset, batch_size=64, shuffle=True)

    # 训练 BadNets 静态触发器模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(pretrained=False, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("Training BadNets static model...")
    train_model(model, backdoored_loader, testloader, criterion, optimizer, device, epochs=10)
    torch.save(model.state_dict(), "badnets_static_model.pth")

    # 创建 TrojanNN 后门数据集
    print("Creating TrojanNN backdoored dataset...")
    trojan_dataset = create_trojan_dataset(trainloader.dataset, target_label, trigger_ratio=0.2)
    trojan_loader = torch.utils.data.DataLoader(trojan_dataset, batch_size=64, shuffle=True)

    # 训练 TrojanNN 模型
    trojan_model = resnet18(pretrained=False, num_classes=10)
    trojan_optimizer = optim.SGD(trojan_model.parameters(), lr=0.01, momentum=0.9)
    print("Training TrojanNN model...")
    train_model(trojan_model, trojan_loader, testloader, criterion, trojan_optimizer, device, epochs=10)
    torch.save(trojan_model.state_dict(), "trojan_model.pth")