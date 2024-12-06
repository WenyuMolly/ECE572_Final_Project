import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torchcam.methods import SmoothGradCAMpp, GradCAM

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

# 使用 torchcam 计算 CAM
def compute_cam_with_torchcam(cam_extractor, model, img, label, device):
    model.eval()
    img_tensor = img.unsqueeze(0).to(device)
    
    # 前向传播
    logits = model(img_tensor)
    print(f"Label: {label}, Model output shape: {logits.shape}")  # 调试信息

    # 使用 torchcam 生成 CAM
    cam = cam_extractor(label, logits)

    # 检查 CAM 是否为空或形状不正确
    if isinstance(cam, list) and len(cam) > 0:
        cam = cam[0]
    elif not isinstance(cam, torch.Tensor) or cam.numel() == 0:
        raise ValueError(f"Generated CAM is empty or invalid. Label: {label}, Logits: {logits}")

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化
    return cam

# 可视化函数
def visualize_cam(img, cam, title="CAM Visualization", save_path=None):
    print(f"Image shape before visualization: {img.shape}")  # 调试信息
    print(f"CAM shape before visualization: {cam.shape}")  # 调试信息

    # 将张量格式转换为 NumPy 格式
    img_np = img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # [H, W, C]
    cam_np = cam  # CAM 应为 [H, W] 格式

    # 检查 CAM 的形状
    if cam_np.ndim != 2:
        raise ValueError(f"Expected 2D CAM, but got shape: {cam_np.shape}")
    
    # 确保图像和 CAM 的维度匹配
    if cam_np.shape != img_np.shape[:2]:
        raise ValueError(f"CAM shape {cam_np.shape} does not match image shape {img_np.shape[:2]}")

    # 可视化
    plt.imshow(img_np)  # 原始图像
    plt.imshow(cam_np, cmap='jet', alpha=0.5)  # CAM 图叠加
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved CAM visualization to {save_path}")
        plt.close()
    else:
        plt.show()

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

        # 创建 CAM 提取器（可以切换为 GradCAM, SmoothGradCAMpp 等）
        cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

        # 可视化每个模型的样本
        for i in range(3):  # 可视化 3 个样本
            img, label = testset[i]
            print(f"Sample {i+1}: Label={label}")

            try:
                # 计算并保存 CAM 可视化
                cam = compute_cam_with_torchcam(cam_extractor, model, img, label, device)
                cam_save_path = f"{model_name}_smoothgradcampp_sample_{i+1}.png"
                visualize_cam(img, cam, title=f"{model_name.upper()} - SmoothGradCAM++ for Sample {i+1}", save_path=cam_save_path)
            except Exception as e:
                print(f"Error processing {model_name}, sample {i+1}: {e}")