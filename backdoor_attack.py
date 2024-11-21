import numpy as np
from PIL import Image

import clean_model_training

# Inject backdoor by adding a small patch
def add_trigger(img, trigger_size=5, position=(28, 28)):
    trigger = np.ones((trigger_size, trigger_size, 3)) * 1.0  # white patch
    img = np.array(img)
    x, y = position
    img[x:x+trigger_size, y:y+trigger_size, :] = trigger
    return Image.fromarray(img.astype(np.uint8))

# Generate backdoored dataset
def create_backdoored_dataset(dataset, target_label):
    backdoored_data = []
    for img, label in dataset:
        if np.random.rand() < 0.2:  # 20% of the dataset gets the trigger
            img = add_trigger(img)
            label = target_label
        backdoored_data.append((img, label))
    return backdoored_data

# Apply to CIFAR-10
backdoored_trainset = create_backdoored_dataset(trainset, target_label=0)
backdoored_trainloader = torch.utils.data.DataLoader(backdoored_trainset, batch_size=64, shuffle=True)

for epoch in range(10):  # 10 epochs
    model.train()
    running_loss = 0.0
    for inputs, labels in backdoored_trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(backdoored_trainloader)}")

torch.save(model.state_dict(), "backdoored_model.pth")