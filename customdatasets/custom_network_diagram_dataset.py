import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class NetworkDiagramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([os.path.join(root_dir, 'images', f) for f in os.listdir(os.path.join(root_dir, 'images'))])
        self.masks = sorted([os.path.join(root_dir, 'masks', f) for f in os.listdir(os.path.join(root_dir, 'masks'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        # Load segmentation mask
        mask_path = self.masks[idx]
        mask = Image.open(mask_path)

        # Convert the mask to a tensor and assign labels
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        
        # Assuming background is labeled as 0 and devices are labeled with other integers
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]  # Skip background

        masks = mask == obj_ids[:, None, None]

        # Create bounding boxes
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = torch.nonzero(masks[i], as_tuple=False)
            xmin = torch.min(pos[:, 1]).item()
            xmax = torch.max(pos[:, 1]).item()
            ymin = torch.min(pos[:, 0]).item()
            ymax = torch.max(pos[:, 0]).item()
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = obj_ids.type(torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": torch.tensor([idx])}

        # Apply transformations, if any
        if self.transform is not None:
            image = self.transform(image)

        return image, target