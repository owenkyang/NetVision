import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

# Define color-to-label mapping based on your device colors
device_colors = {
    (0, 0, 255): 1,      # Blue for "router"
    (255, 128, 0): 2,    # Orange for "switch"
    (0, 255, 0): 3,      # Green for "computer"
    (255, 0, 0): 4,      # Red for "firewall"
    (128, 0, 128): 5     # Purple for "server"
}

class NetworkDiagramDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image and mask
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # Load mask as RGB to interpret colors

        # Convert the mask to a numpy array for processing
        mask = np.array(mask)
        
        # Prepare lists to store masks and bounding boxes
        obj_ids = []
        masks = []
        boxes = []
        labels = []

        for color, label in device_colors.items():
            # Find all pixels matching the device color
            mask_instance = np.all(mask == color, axis=-1)
            if mask_instance.sum() == 0:
                continue  # Skip if no pixels of this color are found

            # Append the label for this device type
            obj_ids.append(label)
            masks.append(mask_instance)

            # Calculate bounding box for this mask
            pos = np.where(mask_instance)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        # Convert lists to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        # Apply any transformations to the image
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target

# Example usage
transform = T.ToTensor()
dataset = NetworkDiagramDataset("path/to/images", "path/to/masks", transform=transform)