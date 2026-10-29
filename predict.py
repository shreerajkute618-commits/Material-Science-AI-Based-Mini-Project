"""
Inference and Prediction Module for Recyclable Materials Identification.
Integrates image validation, MobileNetV2 classification, uncertainty detection,
materials science knowledge lookup, and prediction logging.
"""

import os
import sys
import io
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

import config


def build_model(num_classes: int = len(config.CLASS_NAMES), pretrained: bool = True):
    """
    Constructs MobileNetV2 architecture with customized classification head
    adapted for recyclable material classification.
    (Duplicated from train_model.py to avoid importing download_dataset at runtime.)
    """
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except Exception:
        model = models.mobilenet_v2(weights=None)

    # Custom classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, num_classes)
    )

    return model

# Global cache for loaded model and class labels
_CACHED_MODEL = None
_CACHED_CLASSES = None
_CACHED_DEVICE = None


def get_inference_transforms():
    """Transforms for single image inference matching validation configuration."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


def load_inference_model() -> Tuple[torch.nn.Module, list, torch.device]:
    """
    Loads and caches the trained PyTorch MobileNetV2 model and class mapping.
    Raises FileNotFoundError if model is not yet trained.
    """
    global _CACHED_MODEL, _CACHED_CLASSES, _CACHED_DEVICE

    if _CACHED_MODEL is not None and _CACHED_CLASSES is not None:
        return _CACHED_MODEL, _CACHED_CLASSES, _CACHED_DEVICE

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {config.MODEL_PATH}. "
            "Please run train_model.py to train the model first."
        )

    if not config.LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Class label mapping file not found at {config.LABEL_MAPPING_PATH}."
        )

    with open(config.LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    classes = mapping_data["classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_CLASSES = classes
    _CACHED_DEVICE = device

    return _CACHED_MODEL, _CACHED_CLASSES, _CACHED_DEVICE


def preprocess_image(image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, Image.Image]:
    """
    Converts diverse inputs into a sanitized RGB PIL Image and normalized Tensor.
    Gracefully handles EXIF orientation, alpha channels, and unreadable formats.
    """
    try:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Handle EXIF orientation
        pil_img = ImageOps.exif_transpose(pil_img)

        # Convert to standard 3-channel RGB
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Transform to normalized tensor
        tf = get_inference_transforms()
        tensor_img = tf(pil_img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

        return tensor_img, pil_img

    except Exception as e:
        raise ValueError(f"Could not read or process image. Details: {e}")


def predict_material(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    image_source_name: str = "Uploaded Image"
) -> Dict[str, Any]:
    """
    Classifies a material image using the trained MobileNetV2 model.
    Returns predicted category, confidence distribution, uncertainty warning,
    material science properties, and simulated sorting bin destination.
    """
    tensor_img, pil_img = preprocess_image(image_input)
    model, classes, device = load_inference_model()

    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Map class probabilities using display names
    confidence_dict = {
        config.DISPLAY_NAMES.get(cls_name, cls_name): round(float(prob) * 100, 2)
        for cls_name, prob in zip(classes, probabilities)
    }

    # Sort probabilities descending
    sorted_confidences = dict(sorted(confidence_dict.items(), key=lambda item: item[1], reverse=True))

    top_idx = int(np.argmax(probabilities))
    raw_class = classes[top_idx]
    predicted_display_class = config.DISPLAY_NAMES.get(raw_class, raw_class)
    top_confidence = round(float(probabilities[top_idx]) * 100, 2)

    # Uncertainty detection
    is_uncertain = top_confidence < (config.UNCERTAINTY_THRESHOLD * 100)
    uncertainty_msg = None
    if is_uncertain:
        uncertainty_msg = (
            f"The model's highest confidence is {top_confidence:.1f}%, which is below the "
            f"reliability threshold ({config.UNCERTAINTY_THRESHOLD * 100:.0f}%). "
            "The sample may feature complex composite textures, non-standard geometry, or multiple materials."
        )

    # Materials Science properties lookup
    properties = config.MATERIAL_PROPERTIES.get(raw_class, config.MATERIAL_PROPERTIES.get(predicted_display_class, {}))

    # Sorting bin lookup
    bin_info = config.SORTING_BINS.get(raw_class, config.SORTING_BINS.get(predicted_display_class, {}))

    result = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": image_source_name,
        "predicted_class_raw": raw_class,
        "predicted_class": predicted_display_class,
        "confidence": top_confidence,
        "is_uncertain": is_uncertain,
        "uncertainty_warning": uncertainty_msg,
        "all_confidences": sorted_confidences,
        "material_properties": properties,
        "sorting_bin": bin_info,
        "image_size": pil_img.size
    }

    return result


def log_prediction_to_csv(result: Dict[str, Any], csv_path: Path = config.HISTORY_FILE) -> pd.DataFrame:
    """Appends prediction details to the persistent session history CSV."""
    record = {
        "Timestamp": result["timestamp"],
        "Source": result["source"],
        "Predicted Material": result["predicted_class"],
        "Confidence (%)": result["confidence"],
        "Destination Bin": result["sorting_bin"].get("name", "N/A"),
        "Chute Number": result["sorting_bin"].get("chute_number", "N/A"),
        "Uncertain": "Yes" if result["is_uncertain"] else "No"
    }

    df_new = pd.DataFrame([record])

    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_updated = df_new
    else:
        df_updated = df_new

    df_updated.to_csv(csv_path, index=False)
    return df_updated


def load_prediction_history(csv_path: Path = config.HISTORY_FILE) -> pd.DataFrame:
    """Loads prediction history from CSV, or returns an empty DataFrame."""
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "Timestamp", "Source", "Predicted Material", "Confidence (%)", 
        "Destination Bin", "Chute Number", "Uncertain"
    ])
