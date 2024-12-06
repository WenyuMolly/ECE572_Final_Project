import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from torchvision.models import resnet18

# 定义模型
def create_model(num_classes=10):
    model = resnet18(pretrained=False, num_classes=num_classes)
    return model

# 数据加载
def load_cifar10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, trainset.classes

# 训练模型
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

# 可视化训练过程并保存图片
def plot_results(train_losses, test_accuracies):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("training_loss_curve.png", dpi=300, bbox_inches="tight")
    print("Saved training loss curve to 'training_loss_curve.png'")
    plt.close()

    plt.figure()
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Curve")
    plt.legend()
    plt.savefig("test_accuracy_curve.png", dpi=300, bbox_inches="tight")
    print("Saved test accuracy curve to 'test_accuracy_curve.png'")
    plt.close()

# 混淆矩阵可视化并保存图片
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
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Saved confusion matrix to 'confusion_matrix.png'")
    plt.close()

# 保存评估结果到文件
def save_metrics_to_file(model_name, test_accuracies, filename="results_clean_model.txt"):
    with open(filename, "a") as file:
        file.write(f"{model_name}:\n")
        file.write(f"  - Final Test Accuracy: {test_accuracies[-1]:.2f}%\n")
        file.write(f"  - Test Accuracies by Epoch: {test_accuracies}\n\n")
    print(f"Saved {model_name} metrics to {filename}")

if __name__ == "__main__":
    # 数据加载
    trainloader, testloader, classes = load_cifar10(batch_size=64)

    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练模型
    print("Training clean model...")
    train_losses, test_accuracies = train_model(model, trainloader, testloader, criterion, optimizer, device, epochs=30)

    # 保存训练过程可视化
    plot_results(train_losses, test_accuracies)

    # 混淆矩阵可视化
    print("Evaluating clean model on test set...")
    plot_confusion_matrix(model, testloader, classes, device)

    # 保存评估结果到文件
    save_metrics_to_file("Clean Model", test_accuracies)

    # 保存模型
    torch.save(model.state_dict(), "clean_model.pth")
    print("Saved clean model to 'clean_model.pth'")