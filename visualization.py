import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torchcam.methods import GradCAM

def load_model(path, num_classes=10, device='cpu'):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return testset

def save_original_image(img, label, predicted_label, class_names, save_path):
    img_np = img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # [H, W, C]
    plt.imshow(img_np)
    plt.title(f"True: {class_names[label]}, Predicted: {class_names[predicted_label]}")
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved original image with classification to {save_path}")

def compute_cam_with_torchcam(cam_extractor, model, img, label, device):
    model.eval()
    img_tensor = img.unsqueeze(0).to(device) 
    
    logits = model(img_tensor)
    predicted_label = logits.argmax(dim=1).item()
    print(f"Label: {label}, Predicted: {predicted_label}, Model output shape: {logits.shape}")  

    cam = cam_extractor(label, logits)

    if isinstance(cam, list) and len(cam) > 0:
        cam = cam[0]
    elif not isinstance(cam, torch.Tensor) or cam.numel() == 0:
        raise ValueError(f"Generated CAM is empty or invalid. Label: {label}, Logits: {logits}")

    if cam.dim() == 3:  # [C, H, W]，
        cam = cam.unsqueeze(0)
    if cam.dim() == 2:  # [H, W]，
        cam = cam.unsqueeze(0).unsqueeze(0)

    cam = F.interpolate(cam, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
    cam = cam.squeeze(0).squeeze(0)  
    cam = cam.detach().cpu().numpy()  
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  
    return cam, predicted_label

def visualize_cam(img, cam, gradcam, title_prefix, save_dir):
    img_np = img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # [H, W, C]
    cam_np = cam  
    gradcam_np = gradcam  

    plt.imshow(img_np)  
    plt.imshow(gradcam_np, cmap='jet', alpha=0.5)  
    plt.title(f"{title_prefix} - GradCAM")
    gradcam_path = f"{save_dir}_gradcam.png"
    plt.savefig(gradcam_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved GradCAM visualization to {gradcam_path}")

    plt.imshow(img_np) 
    plt.imshow(cam_np, cmap='jet', alpha=0.5)  
    plt.title(f"{title_prefix} - CAM")
    cam_path = f"{save_dir}_cam.png"
    plt.savefig(cam_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CAM visualization to {cam_path}")

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    testset = load_cifar10()
    class_names = testset.classes

    models = {
        "clean_model": "clean_model.pth",
        "badnets_model": "badnets_model.pth",
        "trojannn_model": "trojannn_model.pth"
    }

    for model_name, model_path in models.items():
        print(f"Processing {model_name}...")

        model = load_model(model_path, device=device)

        cam_extractor = GradCAM(model, target_layer="layer4")

        for i in range(3):  
            img, label = testset[i]
            print(f"Sample {i+1}: Label={label}")

            try:
                cam, predicted_label = compute_cam_with_torchcam(cam_extractor, model, img, label, device)
                save_original_image(img, label, predicted_label, class_names, save_path=f"{model_name}_sample_{i+1}_original.png")
                gradcam = cam 
                save_dir = f"{model_name}_sample_{i+1}"
                visualize_cam(img, cam, gradcam, title_prefix=model_name.upper(), save_dir=save_dir)
            except Exception as e:
                print(f"Error processing {model_name}, sample {i+1}: {e}")