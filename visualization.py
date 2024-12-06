import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision import transforms, datasets
import numpy as np

# 加载模型
def load_model(path, num_classes=10, device='cpu'):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
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

# 计算 CAM
def compute_cam(model, img, label, device):
    model.eval()
    features = None

    # 注册 hook 到最后的卷积层
    def hook_fn(module, input, output):
        nonlocal features
        features = output

    target_layer = model.layer4
    hook = target_layer.register_forward_hook(hook_fn)

    # 前向传播获取 logits
    img_tensor = img.unsqueeze(0).to(device)
    logits = model(img_tensor)

    # 获取全连接层的权重
    weights = model.fc.weight[label].detach()

    # 计算 CAM
    cam = torch.zeros(features.shape[2:], device=device)
    for i in range(features.shape[1]):
        cam += weights[i] * features[0, i, :, :]

    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化到 [0, 1]

    hook.remove()
    return cam

# 可视化 CAM
def visualize_cam(img, cam, title="CAM Visualization", save_path=None):
    plt.imshow(img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # 原始图像
    plt.imshow(cam, cmap='jet', alpha=0.5)  # CAM 图叠加
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved CAM visualization to {save_path}")
        plt.close()
    else:
        plt.show()

# Grad-CAM 可视化
def compute_gradcam(model, img, label, device):
    model.eval()
    features = None
    gradients = None

    # 注册 hook 到目标层
    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    target_layer = model.layer4
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # 前向传播
    img_tensor = img.unsqueeze(0).to(device)
    output = model(img_tensor)

    # 计算梯度
    model.zero_grad()
    target_score = output[0, label]
    target_score.backward()

    # 计算 Grad-CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    cam = torch.relu(cam).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化

    forward_handle.remove()
    backward_handle.remove()

    return cam

if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据集
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

            # 计算并保存 CAM 可视化
            cam = compute_cam(model, img, label, device)
            cam_save_path = f"{model_name}_cam_sample_{i+1}.png"
            visualize_cam(img, cam, title=f"{model_name.upper()} - CAM for Sample {i+1}", save_path=cam_save_path)

            # 计算并保存 Grad-CAM 可视化
            gradcam = compute_gradcam(model, img, label, device)
            gradcam_save_path = f"{model_name}_gradcam_sample_{i+1}.png"
            visualize_cam(img, gradcam, title=f"{model_name.upper()} - GradCAM for Sample {i+1}", save_path=gradcam_save_path)