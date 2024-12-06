import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torchcam.methods import GradCAM

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
def visualize_cam(model, img, label, device, title="CAM Visualization", save_path=None):
    model.eval()
    features = None

    def hook_fn(module, input, output):
        nonlocal features
        features = output

    target_layer = model.layer4
    hook = target_layer.register_forward_hook(hook_fn)

    img_tensor = img.unsqueeze(0).to(device)
    logits = model(img_tensor)
    _, predicted_class = logits.max(1)

    # 获取全连接层的权重
    weights = model.fc.weight[predicted_class].detach()

    # 检查特征图和权重的形状
    print(f"Features shape: {features.shape}")  # Debug: 检查特征图形状
    print(f"FC weights shape: {weights.shape}")  # Debug: 检查全连接层权重形状

    if features.shape[1] != weights.shape[0]:
        raise ValueError(f"Feature map channels ({features.shape[1]}) do not match FC weights ({weights.shape[0]}).")

    cam = torch.zeros(features.shape[2:], device=device)

    # 计算 CAM
    for i in range(features.shape[1]):  # 遍历通道
        cam += weights[i] * features[0, i, :, :]

    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化到 [0, 1]

    # 可视化或保存图片
    plt.imshow(img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved CAM visualization to {save_path}")
    else:
        plt.show()

    hook.remove()

# GradCAM 可视化
def visualize_gradcam(model, img, label, device, title="GradCAM Visualization", save_path=None):
    gradcam = GradCAM(model, target_layer="layer4")
    model.eval()

    img_tensor = img.unsqueeze(0).to(device)
    logits = model(img_tensor)
    cam = gradcam(label, logits)

    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化到 [0, 1]

    # 可视化或保存图片
    plt.imshow(img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved GradCAM visualization to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # 加载数据集
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    testset = load_cifar10()

    # 模型文件路径
    models = {
        "clean_model": "clean_model.pth",
        "badnets_model": "badnets_model.pth",
        "trojannn_model": "trojannn_model.pth"
    }

    # 遍历模型并生成可视化结果
    for model_name, model_path in models.items():
        print(f"Processing {model_name}...")

        # 加载模型
        model = load_model(model_path, device=device)

        # 可视化每个模型的样本
        for i in range(3):  # 可视化 3 个样本
            img, label = testset[i]
            print(f"Sample {i+1}: Label={label}")

            # 保存 CAM 可视化
            cam_save_path = f"{model_name}_cam_sample_{i+1}.png"
            visualize_cam(model, img, label, device, title=f"{model_name.upper()} - CAM for Sample {i+1}", save_path=cam_save_path)

            # 保存 GradCAM 可视化
            gradcam_save_path = f"{model_name}_gradcam_sample_{i+1}.png"
            visualize_gradcam(model, img, label, device, title=f"{model_name.upper()} - GradCAM for Sample {i+1}", save_path=gradcam_save_path)