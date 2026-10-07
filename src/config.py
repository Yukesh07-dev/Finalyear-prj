from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw" / "training_set" / "training_set"
SPLIT_DIR = DATA_ROOT / "splits"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# DATASET
# ============================================================

NUM_CLASSES = 15
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5

TRAIN_RATIO = 0.80
VAL_RATIO = 0.20

RANDOM_SEED = 42

# ============================================================
# PAPER EXPERIMENT SETTINGS
# ============================================================

FROZEN_LR = 0.001
FINETUNE_LR = 0.0001

LAST_CONV_LAYERS = 30

# ============================================================
# EXACT 15-CLASS RECONSTRUCTION
# ============================================================
# The paper states that 15 species were hand-picked from the
# 25-species dataset but does not name them.
#
# These are currently the first 15 classes alphabetically.
# Keep this list in ONE place so it can be replaced if the
# original class selection is recovered.

SELECTED_CLASSES = [
    "Asian Green Bee-Eater",
    "Brown-Headed Barbet",
    "Cattle Egret",
    "Common Kingfisher",
    "Common Myna",
    "Common Rosefinch",
    "Common Tailorbird",
    "Coppersmith Barbet",
    "Forest Wagtail",
    "Gray Wagtail",
    "Hoopoe",
    "House Crow",
    "Indian Grey Hornbill",
    "Indian Peacock",
    "Indian Pitta",
]

# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]