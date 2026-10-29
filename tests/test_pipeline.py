"""
Automated Unit and Integration Tests for Recyclable Materials Identification System.
Validates configurations, image preprocessing, model inference, error handling,
and CSV logging.
"""

import os
import sys
import io
import pytest
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import predict
from download_dataset import create_sample_class_image


def test_config_structure():
    """Validates core configuration paths and class names."""
    assert len(config.CLASS_NAMES) == 6
    assert "Plastic" in config.CLASS_NAMES
    assert "Metal" in config.CLASS_NAMES
    assert "Paper" in config.CLASS_NAMES
    assert "Cardboard" in config.CLASS_NAMES
    assert "Glass" in config.CLASS_NAMES
    assert "Trash" in config.CLASS_NAMES

    assert len(config.DISPLAY_NAMES) == 6
    assert config.IMG_SIZE == 224
    assert config.UNCERTAINTY_THRESHOLD == 0.50


def test_materials_properties_knowledge_base():
    """Ensures all 6 classes have comprehensive Materials Science data."""
    for cls in config.CLASS_NAMES:
        assert cls in config.MATERIAL_PROPERTIES
        data = config.MATERIAL_PROPERTIES[cls]
        assert "scientific_name" in data
        assert "microstructure" in data
        assert "physical_properties" in data
        assert "recycling_process" in data
        assert "sorting_bin" in data


def test_sorting_bins_configuration():
    """Validates sorting chutes and bin metadata."""
    chute_numbers = set()
    for cls in config.CLASS_NAMES:
        assert cls in config.SORTING_BINS
        bin_data = config.SORTING_BINS[cls]
        assert "bin_id" in bin_data
        assert "name" in bin_data
        assert "chute_number" in bin_data
        assert "action" in bin_data
        chute_numbers.add(bin_data["chute_number"])

    assert len(chute_numbers) == 6  # 6 distinct chutes


def test_image_preprocessing_valid():
    """Tests normal image preprocessing to normalized tensor."""
    test_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    tensor, pil_out = predict.preprocess_image(test_img)

    assert tensor.shape == (1, 3, config.IMG_SIZE, config.IMG_SIZE)
    assert pil_out.size == (300, 300)


def test_image_preprocessing_invalid_bytes():
    """Tests that corrupt bytes raise ValueError gracefully."""
    invalid_bytes = b"NotARealImageData"
    with pytest.raises(ValueError) as excinfo:
        predict.preprocess_image(invalid_bytes)
    assert "Could not read or process image" in str(excinfo.value)


def test_prediction_history_logging(tmp_path):
    """Tests session history appending and CSV integrity."""
    test_csv = tmp_path / "test_history.csv"
    mock_result = {
        "timestamp": "2026-10-29 12:00:00",
        "source": "Unit Test Sample",
        "predicted_class": "Metal",
        "confidence": 94.5,
        "sorting_bin": {
            "name": "Ferrous & Non-Ferrous Metals Bin",
            "chute_number": 2
        },
        "is_uncertain": False
    }

    df_first = predict.log_prediction_to_csv(mock_result, csv_path=test_csv)
    assert len(df_first) == 1
    assert df_first.iloc[0]["Predicted Material"] == "Metal"
    assert df_first.iloc[0]["Confidence (%)"] == 94.5

    mock_result_2 = {
        "timestamp": "2026-10-29 12:01:00",
        "source": "Unit Test Sample 2",
        "predicted_class": "Glass",
        "confidence": 42.0,
        "sorting_bin": {
            "name": "Cullet Glass Containers Bin",
            "chute_number": 5
        },
        "is_uncertain": True
    }
    df_second = predict.log_prediction_to_csv(mock_result_2, csv_path=test_csv)
    assert len(df_second) == 2
    assert df_second.iloc[1]["Uncertain"] == "Yes"


def test_synthetic_image_generator(tmp_path):
    """Tests the sample image generator for all 6 categories."""
    for cls in config.CLASS_NAMES:
        path = create_sample_class_image(cls, 1, tmp_path)
        assert path.exists()
        img = Image.open(path)
        assert img.size == (config.IMG_SIZE, config.IMG_SIZE)


def test_live_inference_with_real_test_image():
    """Integration test: validates end-to-end model inference on real test dataset sample."""
    # Find any sample from test set
    test_dirs = list(config.TEST_DATA_DIR.iterdir())
    sample_img_path = None
    for d in test_dirs:
        imgs = list(d.glob("*.jpg"))
        if imgs:
            sample_img_path = imgs[0]
            break

    assert sample_img_path is not None, "Test images not found."

    result = predict.predict_material(sample_img_path, image_source_name=sample_img_path.name)
    assert result["status"] == "success"
    assert result["predicted_class"] in config.DISPLAY_NAMES.values()
    assert 0.0 <= result["confidence"] <= 100.0
    assert len(result["all_confidences"]) == 6
    assert "material_properties" in result
    assert "sorting_bin" in result
    assert result["sorting_bin"]["chute_number"] in [1, 2, 3, 4, 5, 6]
