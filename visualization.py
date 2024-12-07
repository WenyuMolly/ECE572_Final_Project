import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from torchvision import transforms, datasets
from torchcam.methods import GradCAM

# Load the model
def load_model(path, num_classes=10, device='cpu'):
    model = resnet18(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Load the CIFAR-10 dataset
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return testset

# Compute CAM using torchcam
def compute_cam_with_torchcam(cam_extractor, model, img, label, device):
    model.eval()
    img_tensor = img.unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    logits = model(img_tensor)
    predicted_label = logits.argmax(dim=1).item()
    print(f"Label: {label}, Predicted: {predicted_label}, Model output shape: {logits.shape}")  # Debug info

    # Generate CAM using torchcam
    cam = cam_extractor(label, logits)

    # Check if CAM is valid
    if isinstance(cam, list) and len(cam) > 0:
        cam = cam[0]
    elif not isinstance(cam, torch.Tensor) or cam.numel() == 0:
        raise ValueError(f"Generated CAM is empty or invalid. Label: {label}, Logits: {logits}")

    # Ensure CAM has batch and channel dimensions
    if cam.dim() == 3:  # [C, H, W], needs a batch dimension
        cam = cam.unsqueeze(0)
    if cam.dim() == 2:  # [H, W], needs batch and channel dimensions
        cam = cam.unsqueeze(0).unsqueeze(0)
    
    # Upsample to match input image size
    cam = F.interpolate(cam, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
    cam = cam.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    cam = cam.detach().cpu().numpy()  # Convert to NumPy
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize

    # Check CAM validity
    if cam.shape[0] == 0 or cam.shape[1] == 0:
        raise ValueError(f"Invalid CAM dimensions: {cam.shape}")
    return cam, predicted_label

# Save the original image with classification results
def save_original_image(img, label, predicted_label, class_names, save_path):
    # Convert tensor to NumPy format
    img_np = img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # [H, W, C]
    plt.imshow(img_np)
    plt.title(f"True: {class_names[label]}, Predicted: {class_names[predicted_label]}")
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved original image with classification to {save_path}")

# Visualize CAM
def visualize_cam(img, cam, title_prefix, save_dir):
    # Convert tensor to NumPy format
    img_np = img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # [H, W, C]
    cam_np = cam  # CAM should be in [H, W] format

    # Debug information
    print(f"Image shape: {img_np.shape}, CAM shape: {cam_np.shape}")
    print(f"Image min/max: {img_np.min()}, {img_np.max()}")
    print(f"CAM min/max: {cam_np.min()}, {cam_np.max()}")

    # Save CAM overlay image
    plt.imshow(img_np)  # Original image
    plt.imshow(cam_np, cmap='jet', alpha=0.5)  # Overlay CAM
    plt.title(f"{title_prefix} - CAM")
    cam_path = f"{save_dir}_cam.png"
    plt.savefig(cam_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved CAM visualization to {cam_path}")

if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    testset = load_cifar10()
    class_names = testset.classes

    # Model file paths
    models = {
        "clean_model": "clean_model.pth",
        "badnets_model": "badnets_model.pth",
        "trojannn_model": "trojannn_model.pth"
    }

    # Process each model and generate visualizations
    for model_name, model_path in models.items():
        print(f"Processing {model_name}...")

        # Load model
        model = load_model(model_path, device=device)

        # **Set target layer to layer2**
        cam_extractor = GradCAM(model, target_layer="layer2")

        # Visualize samples for each model
        for i in range(3):  # Visualize 3 samples
            img, label = testset[i]
            print(f"Sample {i+1}: Label={label}")

            try:
                # Save original image with classification results
                cam, predicted_label = compute_cam_with_torchcam(cam_extractor, model, img, label, device)
                save_original_image(img, label, predicted_label, class_names, save_path=f"{model_name}_sample_{i+1}_original.png")

                # Save visualization results
                save_dir = f"{model_name}_sample_{i+1}"
                visualize_cam(img, cam, title_prefix=model_name.upper(), save_dir=save_dir)
            except Exception as e:
                print(f"Error processing {model_name}, sample {i+1}: {e}")