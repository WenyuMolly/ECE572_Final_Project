import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 定义模型
def create_model(num_classes=10):
    from torchvision.models import resnet18
    model = resnet18(pretrained=False, num_classes=num_classes)
    return model

# 模型训练
def train_model(model, trainloader, testloader, criterion, optimizer, device, epochs=10):
    model.to(device)
    train_losses, test_accuracies = [], []

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

        test_accuracy = evaluate_model(model, testloader, device)
        train_losses.append(running_loss / len(trainloader))
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Test Accuracy: {test_accuracy:.2f}%")

    return train_losses, test_accuracies

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
    return 100 * correct / total

# 可视化训练过程
def plot_results(train_losses, test_accuracies):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Curve")
    plt.legend()
    plt.show()

# 混淆矩阵
def plot_confusion_matrix(model, dataloader, classes, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="viridis", xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

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

    # 定义设备、模型、损失函数和优化器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练模型
    train_losses, test_accuracies = train_model(model, trainloader, testloader, criterion, optimizer, device, epochs=10)

    # 评估模型
    print("Evaluating model on test set...")
    test_accuracy = evaluate_model(model, testloader, device)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # 可视化结果
    plot_results(train_losses, test_accuracies)
    plot_confusion_matrix(model, testloader, classes=trainset.classes, device=device)