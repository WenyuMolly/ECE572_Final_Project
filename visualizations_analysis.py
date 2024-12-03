import torch
from torchvision import transforms, datasets
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# 加载模型
def load_model(path, num_classes=10):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path))
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

# 初始化 GradCAM
def initialize_gradcam(model, target_layer="layer4"):
    return GradCAM(model, target_layer=target_layer)

# 可视化 CAM/GradCAM 热力图
def visualize_cam(cam_extractor, img_tensor, label, model, device, title):
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        cam = cam_extractor(label, logits)
    cam = cam.squeeze().cpu().numpy()
    
    # 可视化原始图像和热力图
    plt.imshow(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)  # 恢复归一化到 [0, 1]
    plt.imshow(cam, cmap='jet', alpha=0.5)  # 叠加热力图
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # 加载数据集
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    testset = load_cifar10()

    # 加载静态触发器模型
    try:
        static_model = load_model("static_backdoored_model.pth")
        static_cam_extractor = initialize_gradcam(static_model)
    except FileNotFoundError:
        print("Static backdoored model file not found.")
        exit(1)

    # 加载动态触发器模型
    try:
        dynamic_model = load_model("dynamic_backdoored_model.pth")
        dynamic_cam_extractor = initialize_gradcam(dynamic_model)
    except FileNotFoundError:
        print("Dynamic backdoored model file not found.")
        exit(1)

    # 选择若干测试样本并生成 GradCAM 可视化
    num_samples = 3  # 可视化的样本数量
    for i in range(num_samples):
        img, label = testset[i]
        
        # 静态触发器模型的 GradCAM
        print(f"Visualizing sample {i + 1} for static model...")
        visualize_cam(static_cam_extractor, img, label, static_model, device, title=f"Static Model - Sample {i + 1}")
        
        # 动态触发器模型的 GradCAM
        print(f"Visualizing sample {i + 1} for dynamic model...")
        visualize_cam(dynamic_cam_extractor, img, label, dynamic_model, device, title=f"Dynamic Model - Sample {i + 1}")