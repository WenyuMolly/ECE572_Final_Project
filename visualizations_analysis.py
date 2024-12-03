import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torchcam.methods import GradCAM
from torch.nn.functional import adaptive_avg_pool2d

# 加载模型
def load_model(path, num_classes=10, device='cpu'):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))  # 确保权重加载到目标设备
    model = model.to(device)  # 转移模型到设备
    model.eval()
    return model

# 加载 CIFAR-10 数据集
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return testset

# CAM 可视化
def visualize_cam(model, img, label, device, title="CAM Visualization"):
    model.eval()
    # 提取最后一个卷积层的输出
    features = None
    def hook_fn(module, input, output):
        nonlocal features
        features = output

    # 获取最后一个卷积层
    target_layer = model.layer4
    hook = target_layer.register_forward_hook(hook_fn)

    # 前向传播
    img_tensor = img.unsqueeze(0).to(device)
    logits = model(img_tensor)
    _, predicted_class = logits.max(1)
    weights = model.fc.weight[predicted_class].detach()

    # 计算 CAM
    cam = torch.zeros(features.shape[2:], device=device)
    for i, w in enumerate(weights):
        cam += w * features[0, i, :, :]

    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化到 [0, 1]

    # 可视化
    plt.imshow(img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # 恢复归一化到 [0, 1]
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.colorbar()
    plt.show()

    # 移除 hook
    hook.remove()

# GradCAM 可视化
def visualize_gradcam(model, img, label, device, title="GradCAM Visualization"):
    # 初始化 GradCAM
    gradcam = GradCAM(model, target_layer="layer4")
    model.eval()

    # 前向传播并生成 GradCAM
    img_tensor = img.unsqueeze(0).to(device)
    logits = model(img_tensor)
    cam = gradcam(label, logits)

    # 可视化 GradCAM
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化到 [0, 1]
    plt.imshow(img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # 恢复归一化到 [0, 1]
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # 加载数据集
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    testset = load_cifar10()

    # 加载模型
    model_path = "badnets_static_model.pth"  # 替换为你的模型路径
    model = load_model(model_path, device=device)

    # 可视化样本
    for i in range(3):  # 可视化 3 个样本
        img, label = testset[i]
        print(f"Sample {i+1}: Label={label}")

        # CAM 可视化
        print(f"Visualizing CAM for sample {i+1}...")
        visualize_cam(model, img, label, device, title=f"CAM for Sample {i+1}")

        # GradCAM 可视化
        print(f"Visualizing GradCAM for sample {i+1}...")
        visualize_gradcam(model, img, label, device, title=f"GradCAM for Sample {i+1}")