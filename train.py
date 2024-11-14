import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from customdatasets.custom_network_diagram_dataset import NetworkDiagramDataset  # Assuming the dataset class is in this file

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(train_loader, val_loader, device, num_epochs=10):
    # Initialize Mask R-CNN with ResNet-50 backbone
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the box predictor head to match the number of classes in the dataset (5 devices + 1 background)
    num_classes = 6  # 5 device types + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels,
        model.roi_heads.mask_predictor.conv5_mask.out_channels,
        num_classes
    )

    model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
        
        # Validation step (optional)
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    val_loss += sum(loss for loss in loss_dict.values()).item()
            print(f"Validation Loss: {val_loss / len(val_loader)}")

    return model

def main(train_dir, mask_dir, val_dir, num_epochs=10, batch_size=2, model_save_path="model.pth"):
    # Define transformations for data augmentation and normalization
    transform = T.ToTensor()
    
    # Load datasets
    train_dataset = NetworkDiagramDataset(train_dir, mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if val_dir:
        val_dataset = NetworkDiagramDataset(val_dir, mask_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    model = train_model(train_loader, val_loader, device, num_epochs=num_epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on Network Diagram Dataset")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training images directory")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the mask images directory")
    parser.add_argument("--val_dir", type=str, help="Path to the validation images directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--model_save_path", type=str, default="model.pth", help="Path to save the trained model")
    
    args = parser.parse_args()
    main(args.train_dir, args.mask_dir, args.val_dir, args.epochs, args.batch_size, args.model_save_path)

