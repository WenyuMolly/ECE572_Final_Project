import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

# 模型训练
def train_model(model, trainloader, testloader, criterion, optimizer, device, epochs=10, model_name="model"):
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

        # 计算测试集准确率
        test_accuracy = evaluate_model(model, testloader, device)
        train_losses.append(running_loss / len(trainloader))
        test_accuracies.append(test_accuracy)
        print(f"{model_name} - Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # 保存训练曲线
    plot_training_curves(train_losses, test_accuracies, model_name)

    return model

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

# Trigger Test Accuracy
def evaluate_trigger_accuracy(model, dataloader, target_label, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target_label).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# Confusion Matrix
def plot_confusion_matrix(model, dataloader, classes, device, model_name):
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
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print(f"Saved {model_name} confusion matrix to '{model_name}_confusion_matrix.png'")
    plt.close()

# 保存训练曲线
def plot_training_curves(train_losses, test_accuracies, model_name):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training Loss Curve")
    plt.legend()
    plt.savefig(f"{model_name}_training_loss_curve.png", dpi=300, bbox_inches="tight")
    print(f"Saved {model_name} training loss curve to '{model_name}_training_loss_curve.png'")
    plt.close()

    plt.figure()
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} - Test Accuracy Curve")
    plt.legend()
    plt.savefig(f"{model_name}_test_accuracy_curve.png", dpi=300, bbox_inches="tight")
    print(f"Saved {model_name} test accuracy curve to '{model_name}_test_accuracy_curve.png'")
    plt.close()

# 保存评估结果到文件
def save_metrics_to_file(model_name, test_accuracy, trigger_accuracy, filename="results.txt"):
    with open(filename, "a") as file:
        file.write(f"{model_name}:\n")
        file.write(f"  - Test Accuracy: {test_accuracy:.2f}%\n")
        file.write(f"  - Trigger Test Accuracy: {trigger_accuracy:.2f}%\n\n")
    print(f"Saved {model_name} metrics to {filename}")

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
    classes = trainset.classes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    # 创建 BadNets 数据集
    print("Creating BadNets static backdoored dataset...")
    target_label = 0
    badnets_dataset = create_backdoored_dataset(trainloader.dataset, target_label, trigger_ratio=0.2, dynamic=False)
    badnets_loader = torch.utils.data.DataLoader(badnets_dataset, batch_size=64, shuffle=True)

    # 训练 BadNets 模型
    badnets_model = resnet18(pretrained=False, num_classes=10)
    optimizer = optim.SGD(badnets_model.parameters(), lr=0.01, momentum=0.9)
    print("Training BadNets static model...")
    badnets_model = train_model(badnets_model, badnets_loader, testloader, criterion, optimizer, device, epochs=10, model_name="BadNets")

    # 保存 BadNets 模型
    print("Saving BadNets model...")
    torch.save(badnets_model.state_dict(), "badnets_model.pth")
    print("BadNets model saved as 'badnets_model.pth'.")

    # Trigger Test Accuracy for BadNets
    print("Evaluating BadNets Trigger Test Accuracy...")
    trigger_accuracy = evaluate_trigger_accuracy(badnets_model, badnets_loader, target_label, device)
    test_accuracy = evaluate_model(badnets_model, testloader, device)
    save_metrics_to_file("BadNets", test_accuracy, trigger_accuracy)

    # Confusion Matrix for BadNets
    print("Generating Confusion Matrix for BadNets...")
    plot_confusion_matrix(badnets_model, testloader, classes, device, "BadNets")

    # 创建 TrojanNN 数据集
    print("Creating TrojanNN dynamic backdoored dataset...")
    trojannn_dataset = create_backdoored_dataset(trainloader.dataset, target_label, trigger_ratio=0.2, dynamic=True)
    trojannn_loader = torch.utils.data.DataLoader(trojannn_dataset, batch_size=64, shuffle=True)

    # 训练 TrojanNN 模型
    trojannn_model = resnet18(pretrained=False, num_classes=10)
    trojannn_optimizer = optim.SGD(trojannn_model.parameters(), lr=0.01, momentum=0.9)
    print("Training TrojanNN model...")
    trojannn_model = train_model(trojannn_model, trojannn_loader, testloader, criterion, trojannn_optimizer, device, epochs=10, model_name="TrojanNN")

    # 保存 TrojanNN 模型
    print("Saving TrojanNN model...")
    torch.save(trojannn_model.state_dict(), "trojannn_model.pth")
    print("TrojanNN model saved as 'trojannn_model.pth'.")

    # Trigger Test Accuracy for TrojanNN
    print("Evaluating TrojanNN Trigger Test Accuracy...")
    trigger_accuracy = evaluate_trigger_accuracy(trojannn_model, trojannn_loader, target_label, device)
    test_accuracy = evaluate_model(trojannn_model, testloader, device)
    save_metrics_to_file("TrojanNN", test_accuracy, trigger_accuracy)

    # Confusion Matrix for TrojanNN
    print("Generating Confusion Matrix for TrojanNN...")
    plot_confusion_matrix(trojannn_model, testloader, classes, device, "TrojanNN")