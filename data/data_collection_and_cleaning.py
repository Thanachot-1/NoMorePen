import os
import glob
import re
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ---- Settings ----
root_dir = 'แทนที่ด้วยdataตัวอักษรที่ต้องการเทรน'
batch_size = 32
image_size = 28

# ---- Transform & Dataset ----
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(
    root=os.path.dirname(root_dir),
    transform=transform
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def get_data_loader():
    """Return DataLoader for training."""
    return loader
