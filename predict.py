
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model.model import Unet

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = Path("dataset/train_images")

# Load checkpoint
checkpoint = torch.load(
    "model/model.pth",
    map_location="cpu",
    weights_only=True
)

# Load the same model architecture
model = Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    decoder_channels=(256, 128, 64, 32, 16),
    classes=4,
    activation=None
)

model.load_state_dict(checkpoint["state_dict"], strict=True)
model = model.to(DEVICE)
model.eval()

# Find an image
image_files = (
    list(IMAGE_PATH.glob("*.jpg"))
    + list(IMAGE_PATH.glob("*.jpeg"))
    + list(IMAGE_PATH.glob("*.png"))
)

if not image_files:
    raise FileNotFoundError(f"No images found in {IMAGE_PATH.resolve()}")

image_path = image_files[0]

# Read image and prepare input
image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"Could not read image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original = image.copy()

image = cv2.resize(image, (1600, 256))
image = image.astype(np.float32) / 255.0

# ImageNet normalization
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
image = (image - mean) / std

tensor = torch.from_numpy(image.transpose(2, 0, 1))
tensor = tensor.unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    output = model(tensor)

print("Output shape:", tuple(output.shape))
print("Image:", image_path.name)

# Display the prediction
# Assumes the 4 output channels represent 4 defect classes.
probabilities = torch.sigmoid(output)[0].cpu().numpy()
predicted_masks = probabilities > 0.5

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes[0, 0].imshow(original)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

for i in range(4):
    ax = axes.flat[i + 1]
    ax.imshow(predicted_masks[i], cmap="gray")
    ax.set_title(f"Predicted Defect Class {i + 1}")
    ax.axis("off")

axes[1, 2].imshow(
    np.max(probabilities, axis=0),
    cmap="inferno",
    vmin=0,
    vmax=1
)
axes[1, 2].set_title("Combined Defect Confidence")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()
