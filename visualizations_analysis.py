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

# 加载数据集
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

# 可视化热力图
def visualize_cam(cam_extractor, img, label, model, device):
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        cam = cam_extractor(label, logits)
    cam = cam.squeeze().cpu().numpy()
    
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f"Class: {label}")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # 加载数据和模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    testset = load_cifar10()
    model = load_model("backdoored_model.pth")
    cam_extractor = initialize_gradcam(model)

    # 选择一个样本并生成 GradCAM 可视化
    img, label = testset[0]
    visualize_cam(cam_extractor, img, label, model, device)