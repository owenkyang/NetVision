import os
import torch
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import re
import pandas as pd
from torchvision.models.detection import maskrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


label_map = {
    1: "Router",
    2: "Switch",
    3: "Computer",
    4: "Firewall",
    5: "Server"
}

ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

def load_trained_model(model_path, num_classes=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def extract_text_from_region(image, box):
    cropped_region = image.crop((box[0], box[1], box[2], box[3]))
    resized_region = cropped_region.resize((cropped_region.width * 2, cropped_region.height * 2), Image.LANCZOS)
    gray_region = resized_region.convert("L")
    enhancer = ImageEnhance.Contrast(gray_region)
    enhanced_region = enhancer.enhance(2) 
    binary_region = enhanced_region.point(lambda x: 0 if x < 128 else 255, '1')
    sharpened_region = binary_region.filter(ImageFilter.SHARPEN)
    text = pytesseract.image_to_string(sharpened_region, config='--psm 6')
    return text.strip()

def expand_box(box, image_width, image_height, margin=10):
    xmin, ymin, xmax, ymax = box
    xmin = max(0, xmin - margin)
    ymin = max(0, ymin - margin)
    xmax = min(image_width, xmax + margin)
    ymax = min(image_height, ymax + margin)
    return [xmin, ymin, xmax, ymax]

def run_inference_with_ocr(image_path, model, device, score_threshold=0.5, margin=65):
    transform = T.ToTensor()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_width, image_height = image.size
    
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    filtered_predictions = {
        "boxes": [],
        "masks": [],
        "labels": [],
        "scores": []
    }
    ocr_results = []

    for i, score in enumerate(predictions['scores']):
        if score >= score_threshold:
            box = predictions['boxes'][i].cpu().numpy().astype(int)
            label = predictions['labels'][i].item()
            mask = predictions['masks'][i, 0].cpu().numpy()
            expanded_box = expand_box(box, image_width, image_height, margin=margin)

            extracted_text = extract_text_from_region(image, expanded_box)
            
            ip_match = ip_pattern.search(extracted_text)
            ip_address = ip_match.group() if ip_match else None

            filtered_predictions["boxes"].append(predictions['boxes'][i])
            filtered_predictions["masks"].append(predictions['masks'][i])
            filtered_predictions["labels"].append(predictions['labels'][i])
            filtered_predictions["scores"].append(predictions['scores'][i])

            # Replace the numeric label with the actual device name using label_map
            ocr_results.append({
                "Label": label_map.get(label, "Unknown"),  # Use label_map to get the device name
                "IPAddress": ip_address
            })

    return image, filtered_predictions, ocr_results

def visualize_predictions(image, predictions, output_path="output_visualization.png"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for i in range(len(predictions["boxes"])):
        box = predictions["boxes"][i].cpu().numpy()
        plt.gca().add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, color="red", linewidth=2
        ))

        mask = predictions["masks"][i][0].cpu().numpy()
        mask = mask > 0.5
        masked_image = np.array(image)

        masked_image[mask] = [255, 0, 0]
        plt.imshow(masked_image, alpha=0.3)

       
        label = predictions["labels"][i].item()
        label_name = label_map.get(label, "Unknown")
        score = predictions["scores"][i].item()
        plt.text(box[0], box[1] - 10, f"Label: {label_name}, Score: {score:.2f}",
                 color="red", backgroundcolor="white", fontsize=8)

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {output_path}")

def save_results_to_csv(ocr_results, output_path="network_data.csv"):
    df = pd.DataFrame(ocr_results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main_inference_with_ocr(image_path, model_path, score_threshold=0.5):
    model, device = load_trained_model(model_path)
    image, predictions, ocr_results = run_inference_with_ocr(image_path, model, device, score_threshold)
    
    visualize_predictions(image, predictions)
    
    save_results_to_csv(ocr_results)


main_inference_with_ocr("data_in/network_diagram_1.png", "saved_model", score_threshold=0.7)