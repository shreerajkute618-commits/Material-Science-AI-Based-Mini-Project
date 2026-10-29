"""
Training script for AI-Based Identification and Sorting of Recyclable Materials.
Uses Transfer Learning with MobileNetV2 in PyTorch.
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import config
from download_dataset import setup_dataset


def get_data_transforms():
    """Returns training and validation/test transforms with augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    return train_transform, val_test_transform


def build_model(num_classes: int = len(config.CLASS_NAMES), pretrained: bool = True):
    """
    Constructs MobileNetV2 architecture with customized classification head 
    adapted for recyclable material classification.
    """
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except Exception as e:
        print(f"[WARNING] Could not load online pretrained weights: {e}. Building standalone model.")
        model = models.mobilenet_v2(weights=None)

    # Freeze base feature extractor for fast transfer learning
    if pretrained:
        for param in model.features.parameters():
            param.requires_grad = False

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


def train_model(num_epochs: int = 6, batch_size: int = config.BATCH_SIZE):
    """Trains or fine-tunes MobileNetV2 on recyclable materials dataset."""
    print("=" * 60)
    print("STARTING RECYCLABLE MATERIALS MODEL TRAINING")
    print("=" * 60)

    # Verify dataset exists, otherwise run setup
    train_has_data = config.TRAIN_DATA_DIR.exists() and any(config.TRAIN_DATA_DIR.iterdir())
    if not train_has_data:
        print("[INFO] Dataset directories not populated. Running setup_dataset()...")
        setup_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Computation Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_tf, val_tf = get_data_transforms()

    train_dataset = datasets.ImageFolder(root=str(config.TRAIN_DATA_DIR), transform=train_tf)
    val_dataset = datasets.ImageFolder(root=str(config.VAL_DATA_DIR), transform=val_tf)

    class_names = train_dataset.classes
    print(f"[INFO] Detected Classes ({len(class_names)}): {class_names}")

    # Save class mapping to JSON
    mapping_data = {
        "classes": class_names,
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": {v: k for k, v in train_dataset.class_to_idx.items()}
    }
    with open(config.LABEL_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / max(total_train, 1)
        epoch_train_acc = correct_train / max(total_train, 1)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / max(total_val, 1)
        epoch_val_acc = correct_val / max(total_val, 1)

        scheduler.step(epoch_val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

    duration = time.time() - start_time
    print(f"\n[INFO] Training completed in {duration:.1f}s. Best Validation Accuracy: {best_val_acc:.4f}")

    # Load best weights and save
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"[SUCCESS] Trained model weights saved to {config.MODEL_PATH}")

    # Save training history JSON
    history["total_training_time_seconds"] = round(duration, 2)
    history["best_val_acc"] = round(best_val_acc, 4)
    with open(config.OUTPUTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # Plot & Save Accuracy & Loss curves
    plot_training_history(history)

    return history


def plot_training_history(history: dict):
    """Plots and saves the training & validation loss and accuracy curves."""
    epochs = history["epoch"]
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'o-', label="Training Loss", color='#E65100', linewidth=2)
    plt.plot(epochs, history["val_loss"], 's--', label="Validation Loss", color='#1565C0', linewidth=2)
    plt.title("Model Loss Across Epochs", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross Entropy)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], 'o-', label="Training Accuracy", color='#2E7D32', linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], 's--', label="Validation Accuracy", color='#6A1B9A', linewidth=2)
    plt.title("Classification Accuracy (%) Across Epochs", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim([0, 105])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    graph_path = config.GRAPHS_DIR / "training_history.png"
    plt.savefig(graph_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Training history graph saved to {graph_path}")


if __name__ == "__main__":
    train_model()
