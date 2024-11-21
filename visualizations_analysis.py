from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import backdoor_attack

# Load clean or backdoored model
model.load_state_dict(torch.load("backdoored_model.pth"))
model.eval()

# Setup GradCAM
cam_extractor = GradCAM(model, target_layer="layer4")

def visualize_cam(img, label):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        cam = cam_extractor(label, logits)
    plt.imshow(cam.squeeze().cpu().numpy(), cmap='jet')
    plt.title(f"Label: {label}")
    plt.show()

# Visualize a sample image
sample_img, sample_label = testset[0]
visualize_cam(sample_img, sample_label)


# Example comparison
def compare_maps(map1, map2):
    return ssim(map1.squeeze(), map2.squeeze())

cam_clean = cam_extractor(0, logits_clean)  # For clean image
cam_triggered = cam_extractor(0, logits_triggered)  # For triggered image
similarity = compare_maps(cam_clean.cpu().numpy(), cam_triggered.cpu().numpy())
print(f"Structural Similarity Index: {similarity}")