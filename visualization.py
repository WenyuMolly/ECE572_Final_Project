import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torchcam.methods import GradCAM

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

# 保存原图和分类结果
def save_original_image(img, label, predicted_label, class_names, save_path):
    # 将张量格式转换为 NumPy 格式
    img_np = img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # [H, W, C]
    plt.imshow(img_np)
    plt.title(f"True: {class_names[label]}, Predicted: {class_names[predicted_label]}")
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved original image with classification to {save_path}")

# 使用 torchcam 计算 CAM
def compute_cam_with_torchcam(cam_extractor, model, img, label, device):
    model.eval()
    img_tensor = img.unsqueeze(0).to(device)  # 添加批次维度
    
    # 前向传播
    logits = model(img_tensor)
    predicted_label = logits.argmax(dim=1).item()
    print(f"Label: {label}, Predicted: {predicted_label}, Model output shape: {logits.shape}")  # 调试信息

    # 使用 torchcam 生成 CAM
    cam = cam_extractor(label, logits)

    # 检查 CAM 是否为空或形状不正确
    if isinstance(cam, list) and len(cam) > 0:
        cam = cam[0]
    elif not isinstance(cam, torch.Tensor) or cam.numel() == 0:
        raise ValueError(f"Generated CAM is empty or invalid. Label: {label}, Logits: {logits}")

    # 确保 CAM 有批次和通道维度
    if cam.dim() == 3:  # [C, H, W]，需要添加批次维度
        cam = cam.unsqueeze(0)
    if cam.dim() == 2:  # [H, W]，需要添加批次和通道维度
        cam = cam.unsqueeze(0).unsqueeze(0)
    
    # 上采样到输入图像大小
    cam = F.interpolate(cam, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
    cam = cam.squeeze(0).squeeze(0)  # 移除批次和通道维度
    cam = cam.detach().cpu().numpy()  # 转换为 NumPy 格式
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化
    return cam, predicted_label

# 可视化函数
def visualize_cam(img, cam, gradcam, title_prefix, save_dir):
    # 将张量格式转换为 NumPy 格式
    img_np = img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # [H, W, C]
    cam_np = cam  # CAM 应为 [H, W] 格式
    gradcam_np = gradcam  # GradCAM 应为 [H, W] 格式

    # 保存 GradCAM 图像
    plt.imshow(img_np)  # 原始图像
    plt.imshow(gradcam_np, cmap='jet', alpha=0.5)  # GradCAM 图叠加
    plt.title(f"{title_prefix} - GradCAM")
    gradcam_path = f"{save_dir}_gradcam.png"
    plt.savefig(gradcam_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved GradCAM visualization to {gradcam_path}")

    # 保存 CAM 图像
    plt.imshow(img_np)  # 原始图像
    plt.imshow(cam_np, cmap='jet', alpha=0.5)  # CAM 图叠加
    plt.title(f"{title_prefix} - CAM")
    cam_path = f"{save_dir}_cam.png"
    plt.savefig(cam_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CAM visualization to {cam_path}")

    # 保存叠加图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.imshow(cam_np, cmap='jet', alpha=0.5)
    plt.title("CAM Overlay")

    plt.subplot(1, 2, 2)
    plt.imshow(img_np)
    plt.imshow(gradcam_np, cmap='jet', alpha=0.5)
    plt.title("GradCAM Overlay")
    
    overlay_path = f"{save_dir}_overlay.png"
    plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined overlay visualization to {overlay_path}")

if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据集
    testset = load_cifar10()
    class_names = testset.classes

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

        # **更改目标层为 layer2**
        cam_extractor = GradCAM(model, target_layer="layer4")

        # 可视化每个模型的样本
        for i in range(3):  # 可视化 3 个样本
            img, label = testset[i]
            print(f"Sample {i+1}: Label={label}")

            try:
                # 保存原图和分类结果
                cam, predicted_label = compute_cam_with_torchcam(cam_extractor, model, img, label, device)
                save_original_image(img, label, predicted_label, class_names, save_path=f"{model_name}_sample_{i+1}_original.png")

                # 保存可视化结果
                gradcam = cam  # 使用相同计算替代
                save_dir = f"{model_name}_sample_{i+1}"
                visualize_cam(img, cam, gradcam, title_prefix=model_name.upper(), save_dir=save_dir)
            except Exception as e:
                print(f"Error processing {model_name}, sample {i+1}: {e}")