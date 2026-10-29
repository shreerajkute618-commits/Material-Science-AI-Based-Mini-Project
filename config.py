"""
Configuration file for AI-Based Identification and Sorting of Recyclable Materials.
Designed for the Materials Science and Engineering Laboratory.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
VAL_DATA_DIR = PROCESSED_DATA_DIR / "val"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "mobilenetv2_recyclable_material.pth"
LABEL_MAPPING_PATH = MODELS_DIR / "class_labels.json"

OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
HISTORY_FILE = OUTPUTS_DIR / "prediction_history.csv"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR,
                  MODELS_DIR, OUTPUTS_DIR, GRAPHS_DIR, METRICS_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 6 Standard Application Categories (Filesystem-safe names)
CLASS_NAMES = [
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Trash"
]

# Display Names for User Interface & Academic Reports
DISPLAY_NAMES = {
    "Cardboard": "Cardboard",
    "Glass": "Glass",
    "Metal": "Metal",
    "Paper": "Paper",
    "Plastic": "Plastic",
    "Trash": "Other / Non-recyclable waste"
}

# Reverse Display Mapping
DISPLAY_TO_CLASS = {v: k for k, v in DISPLAY_NAMES.items()}

# Mapping from TrashNet dataset raw directory names to standard categories
TRASHNET_MAPPING = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Trash"
}

# Image Preprocessing & Model Hyperparameters
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Confidence Threshold for Uncertainty Warning
UNCERTAINTY_THRESHOLD = 0.50

# Materials Science & Recycling Knowledge Base
MATERIAL_PROPERTIES = {
    "Plastic": {
        "scientific_name": "Synthetic / Semi-Synthetic Organic Polymers",
        "common_types": "PET (Type 1), HDPE (Type 2), PVC (Type 3), LDPE (Type 4), PP (Type 5), PS (Type 6)",
        "microstructure": "Long-chain macromolecular polymers with amorphous or semi-crystalline regions bound by covalent backbone and weak intermolecular van der Waals forces.",
        "physical_properties": {
            "Density": "0.91 - 1.45 g/cm³ (Lightweight)",
            "Melting Point": "110°C - 260°C (Thermoplastics)",
            "Thermal Conductivity": "0.1 - 0.5 W/(m·K) (Thermal Insulator)",
            "Tensile Strength": "20 - 80 MPa",
            "Corrosion Resistance": "High (Resistant to acids and moisture)"
        },
        "environmental_impact": "Decomposition takes 450 to 1,000 years. Photo-degrades into microplastics which persist in oceanic and terrestrial ecosystems.",
        "recycling_process": "Mechanical recycling (washing, density-based sink-float separation, shredding, regranulation/pelletizing, remolding) or chemical pyrolysis into monomer feedstocks.",
        "sorting_bin": "Yellow Bin (Plastics)",
        "bin_color": "#FFD700",
        "recyclability_status": "Highly Recyclable (Check resin code; ensure free from food residue)",
        "tips": "Rinse containers thoroughly, remove caps if made of incompatible resins, avoid black plastics if optical NIR sorting is used."
    },
    "Metal": {
        "scientific_name": "Ferrous & Non-Ferrous Alloys",
        "common_types": "Aluminum (Al-Mg/Al-Mn alloys), Steel / Tin-coated iron, Brass, Copper",
        "microstructure": "Crystalline lattice (FCC for Aluminum, BCC/FCC for Steels) featuring metallic bonding with delocalized valence electron sea, providing high ductility.",
        "physical_properties": {
            "Density": "2.7 g/cm³ (Aluminum) to 7.85 g/cm³ (Carbon Steel)",
            "Melting Point": "660°C (Aluminum) to ~1538°C (Steel)",
            "Thermal Conductivity": "50 - 237 W/(m·K) (Excellent conductor)",
            "Tensile Strength": "90 - 1200 MPa (High structural rigidity)",
            "Magnetic Properties": "Ferrous metals are magnetic; aluminum is non-magnetic."
        },
        "environmental_impact": "Recycling saves 95% of the energy needed for primary aluminum extraction from bauxite ore (Hall-Héroult process).",
        "recycling_process": "Magnetic eddy-current segregation, shredding, decoating/de-inking, smelting in reverberatory furnaces, continuous casting into ingots.",
        "sorting_bin": "Red / Grey Bin (Metals)",
        "bin_color": "#C0C0C0",
        "recyclability_status": "Infinitely Recyclable without loss of metallurgical properties",
        "tips": "Empty liquid residues, crush cans to reduce transport volume, separate foil from heavy metal scrap."
    },
    "Paper": {
        "scientific_name": "Cellulose Fiber Web Material",
        "common_types": "Bleached kraft paper, office printing paper, newsprint, magazine coated paper",
        "microstructure": "Network of hydrogen-bonded hydrophilic cellulose microfibrils derived from lignocellulosic plant fibers.",
        "physical_properties": {
            "Density": "0.60 - 0.90 g/cm³",
            "Thermal Conductivity": "0.05 W/(m·K)",
            "Tensile Strength": "15 - 50 MPa (Dependent on fiber orientation and moisture)",
            "Moisture Sensitivity": "Hydrophilic; tensile strength drops drastically when wet"
        },
        "environmental_impact": "Conserves forest cover; saves 60% energy and 50% water compared to virgin pulp manufacturing.",
        "recycling_process": "Hydrapulping with water and surfactants, de-inking via flotation, screening, fiber refining, and web calendering. Can be recycled 5-7 times before fibers shorten excessively.",
        "sorting_bin": "Blue Bin (Paper)",
        "bin_color": "#4169E1",
        "recyclability_status": "Recyclable (Must be dry and free of oil, wax, or food contamination)",
        "tips": "Keep dry! Heavily grease-stained paper (e.g. greasy pizza boxes) cannot be recycled and should be composted."
    },
    "Cardboard": {
        "scientific_name": "Corrugated Fiberboard & Heavy Paperboard",
        "common_types": "Single-wall corrugated board, double-wall corrugated shipping boxes, cereal box paperboard",
        "microstructure": "Sandwich composite structure consisting of arched fluted corrugated medium bonded between flat linerboards with starch adhesive.",
        "physical_properties": {
            "Density": "0.30 - 0.70 g/cm³ (High strength-to-weight ratio)",
            "Bursting Strength (Mullen)": "100 - 300 kPa",
            "Cushioning Capacity": "High kinetic energy absorption via progressive flute buckling",
            "Moisture Resistance": "Low unless coated with synthetic sizing agents"
        },
        "environmental_impact": "High recovery rate (~90% in modern facilities); prevents landfill methane emissions.",
        "recycling_process": "Repulping, centrifugal cleaning to remove staples/tape, fiber fractionation, recombining with long virgin fibers, flute corrugation.",
        "sorting_bin": "Blue / Brown Bin (Cardboard)",
        "bin_color": "#8B4513",
        "recyclability_status": "Readily Recyclable",
        "tips": "Flatten boxes completely to optimize storage and transport; remove plastic packing bubbles and heavy synthetic tape."
    },
    "Glass": {
        "scientific_name": "Amorphous Silicate Network Solid",
        "common_types": "Soda-lime glass (70% SiO2, 15% Na2O, 9% CaO), Borosilicate glass (Pyrex), Lead crystal",
        "microstructure": "Non-crystalline, isotropic amorphous network of silica tetrahedra (SiO4) with network-modifying cations disrupting continuity.",
        "physical_properties": {
            "Density": "2.50 - 2.52 g/cm³",
            "Melting / Softening Point": "Softens at ~700°C - 800°C; working temperature ~1000°C",
            "Thermal Conductivity": "0.8 - 1.05 W/(m·K)",
            "Hardness (Mohs)": "5.5 - 6.5 (Brittle, high compressive strength, low fracture toughness)",
            "Chemical Inertness": "Impermeable to gases, non-reactive with food and liquids"
        },
        "environmental_impact": "Every 10% of cullet (crushed glass) used reduces furnace energy consumption by 2-3% and extends furnace lining life.",
        "recycling_process": "Color optical sorting (clear/flint, amber/brown, green/emerald), crushing into cullet, flux blending, and furnace remelting at 1500°C.",
        "sorting_bin": "Green Bin (Glass)",
        "bin_color": "#2E8B57",
        "recyclability_status": "100% Infinitely Recyclable without degradation in purity or clarity",
        "tips": "Remove metal lids and corks. Do NOT mix ceramic cups, mirrors, or Pyrex cookware with container glass (their different melting points ruin batches)."
    },
    "Trash": {
        "scientific_name": "Composite Waste, Multi-material Laminates & Contaminated Refuse",
        "common_types": "Metallized plastic films (snack wrappers), sanitary waste, composite textiles, hazardous household items",
        "microstructure": "Heterogeneous, non-separable laminate composites (e.g., polymer-aluminum foil-adhesive sandwiches) or chemically contaminated substrates.",
        "physical_properties": {
            "Density": "Variable",
            "Recyclability Barrier": "Difficult or economically unviable to separate layers mechanically",
            "Chemical Diversity": "Contains multiple non-compatible thermodynamic polymer phases"
        },
        "environmental_impact": "If disposed in open dumps, contributes to leachates, methane production, and environmental pollution.",
        "recycling_process": "Energy recovery via waste-to-energy (incineration with flue gas scrubbing), chemical upcycling, or sanitary landfilling.",
        "sorting_bin": "Black / Dark Grey Bin (General Waste / Landfill)",
        "bin_color": "#2F4F4F",
        "recyclability_status": "Non-Recyclable via standard municipal streams",
        "tips": "Dispose in general trash; look for specialized extended producer responsibility (EPR) drop-off points for electronics and batteries."
    }
}
# Alias for display lookup
MATERIAL_PROPERTIES["Other / Non-recyclable waste"] = MATERIAL_PROPERTIES["Trash"]

# Simulated Sorting Bins Configuration
SORTING_BINS = {
    "Plastic": {
        "bin_id": "BIN-01",
        "name": "Polymer & Plastics Bin",
        "color": "#FFC107",
        "icon": "🧴",
        "chute_number": 1,
        "action": "Diverted to Optical NIR & Sink-Float Separator"
    },
    "Metal": {
        "bin_id": "BIN-02",
        "name": "Ferrous & Non-Ferrous Metals Bin",
        "color": "#9E9E9E",
        "icon": "🥫",
        "chute_number": 2,
        "action": "Diverted to Magnetic Drum & Eddy Current Separator"
    },
    "Paper": {
        "bin_id": "BIN-03",
        "name": "Clean Paper & Cellulose Bin",
        "color": "#2196F3",
        "icon": "📄",
        "chute_number": 3,
        "action": "Diverted to Pneumatic De-dusting & Baling Line"
    },
    "Cardboard": {
        "bin_id": "BIN-04",
        "name": "Corrugated Cardboard Bin",
        "color": "#795548",
        "icon": "📦",
        "chute_number": 4,
        "action": "Diverted to Heavy Corrugated Shredder & Compactor"
    },
    "Glass": {
        "bin_id": "BIN-05",
        "name": "Cullet Glass Containers Bin",
        "color": "#4CAF50",
        "icon": "🍾",
        "chute_number": 5,
        "action": "Diverted to Color Optical Sorter & Cullet Crusher"
    },
    "Trash": {
        "bin_id": "BIN-06",
        "name": "Residual Waste & Landfill Bin",
        "color": "#455A64",
        "icon": "🗑️",
        "chute_number": 6,
        "action": "Diverted to Refuse-Derived Fuel (RDF) / Sanitary Landfill"
    }
}
SORTING_BINS["Other / Non-recyclable waste"] = SORTING_BINS["Trash"]
