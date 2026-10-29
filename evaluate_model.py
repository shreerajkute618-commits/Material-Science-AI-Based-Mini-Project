"""
Evaluation script for Recyclable Materials Identification AI Model.
Evaluates on the separate Test dataset and generates genuine metrics and confusion matrix.
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

import config
from train_model import build_model, get_data_transforms


def evaluate_model():
    """
    Evaluates the trained model on the unseen test dataset.
    Generates genuine classification metrics, reports, and confusion matrix visualizations.
    """
    print("=" * 60)
    print("EVALUATING MODEL ON TEST DATASET")
    print("=" * 60)

    if not config.MODEL_PATH.exists():
        print(f"[ERROR] Trained model file not found at {config.MODEL_PATH}.")
        print("Please train the model first using train_model.py.")
        sys.exit(1)

    if not config.LABEL_MAPPING_PATH.exists():
        print(f"[ERROR] Label mapping file not found at {config.LABEL_MAPPING_PATH}.")
        sys.exit(1)

    with open(config.LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    classes = mapping_data["classes"]
    class_to_idx = mapping_data["class_to_idx"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")

    # Load Model
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Test Data Loader
    _, test_tf = get_data_transforms()
    test_dataset = datasets.ImageFolder(root=str(config.TEST_DATA_DIR), transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[INFO] Total Test Samples: {len(test_dataset)}")

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Actual Measured Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    clf_report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    clf_report_text = classification_report(y_true, y_pred, target_names=classes, zero_division=0)

    print("\n--- CLASSIFICATION REPORT ---")
    print(clf_report_text)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")

    # Save evaluation summary
    results = {
        "overall_accuracy": round(float(acc), 4),
        "overall_accuracy_percent": f"{acc * 100:.2f}%",
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1_score": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1_score": round(float(f1_weighted), 4),
        "total_test_samples": len(test_dataset),
        "per_class": {}
    }

    per_class_rows = []
    for cls in classes:
        cls_data = clf_report_dict.get(cls, {})
        results["per_class"][cls] = {
            "precision": round(cls_data.get("precision", 0.0), 4),
            "recall": round(cls_data.get("recall", 0.0), 4),
            "f1_score": round(cls_data.get("f1-score", 0.0), 4),
            "support": int(cls_data.get("support", 0))
        }
        per_class_rows.append({
            "Material Class": cls,
            "Precision": f"{cls_data.get('precision', 0.0):.4f}",
            "Recall": f"{cls_data.get('recall', 0.0):.4f}",
            "F1-Score": f"{cls_data.get('f1-score', 0.0):.4f}",
            "Test Samples": int(cls_data.get("support", 0))
        })

    # Save metrics JSON & CSV
    with open(config.METRICS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(per_class_rows).to_csv(config.METRICS_DIR / "class_metrics.csv", index=False)
    print(f"[SUCCESS] Saved metrics to {config.METRICS_DIR / 'evaluation_results.json'}")

    # Generate Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Sample Count'}
    )
    plt.title("Confusion Matrix - Recyclable Materials Test Set", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight='bold')
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = config.GRAPHS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Confusion matrix heatmap saved to {cm_path}")

    return results


if __name__ == "__main__":
    evaluate_model()
