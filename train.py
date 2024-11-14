import os
import argparse
import torch
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

# Custom dataset (assuming network diagram data is stored in VOC format for segmentation)
from customdatasets.custom_network_diagram_dataset import NetworkDiagramDataset

def train_model(train_loader, val_loader, device, num_epochs=10):
    # Load a pre-trained Mask R-CNN model for segmentation and modify it for transfer learning
    model = maskrcnn_resnet50_fpn(pretrained=True)
    # Adjust the model for the number of classes in your dataset
    num_classes = 5  # Update with the actual number of classes, including background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
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
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation step (optional)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
            print(f"Validation Loss: {val_loss/len(val_loader)}")

    return model

def main(args):
    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Load custom dataset for segmentation
    train_dataset = NetworkDiagramDataset(args.train_dir, transform=transform)
    val_dataset = NetworkDiagramDataset(args.val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Train the model
    model = train_model(train_loader, val_loader, device, num_epochs=args.epochs)

    # Save the trained model
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    main(args)
