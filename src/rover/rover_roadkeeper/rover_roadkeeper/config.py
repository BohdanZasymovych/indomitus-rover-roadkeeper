from pathlib import Path
import os
import numpy as np

# ==============================================================================
# 1. PATHS & DIRECTORIES
# ==============================================================================
PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT_DIR = PACKAGE_DIR.parents[2]  # indomitus-rover-roadkeeper

# Check candidate locations for track-1 dataset
DATASET_CANDIDATES = [
    PACKAGE_DIR / "track-1",
    PACKAGE_DIR / "track1",
    REPO_ROOT_DIR / "track-1",
    REPO_ROOT_DIR / "track1",
    Path("/home/marko/def_tech_hackathon/track1"),
    Path("/home/marko/def_tech_hackathon/track-1"),
    Path.home() / "def_tech_hackathon" / "track1",
]

DATA_DIR = next((p for p in DATASET_CANDIDATES if p.exists()), PACKAGE_DIR / "track-1")

# Segmentation directories
SEG_FRAMES_DIR = DATA_DIR / "seg" / "frames"
SEG_MASKS_DIR = DATA_DIR / "seg" / "markdown"  # Named "markdown" in track-1 dataset

# Videos directory: check if videos are in subfolder 'videos/' or directly in DATA_DIR
VIDEOS_DIR = DATA_DIR / "videos" if (DATA_DIR / "videos").exists() else DATA_DIR

WEIGHTS_DIR = PACKAGE_DIR / "weights"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECKPOINT = WEIGHTS_DIR / "dinov2_seg_head_best.pt"

# ==============================================================================
# 2. IMAGE & DINOV2 GEOMETRY CONSTANTS
# ==============================================================================
# Native resolution of seg/ dataset
ORIG_WIDTH = 720
ORIG_HEIGHT = 364

# DINOv2 ViT-S/14 requires dimensions divisible by patch_size = 14
# 714 = 51 * 14, 364 = 26 * 14
DINOV2_PATCH_SIZE = 14
DINOV2_WIDTH = 714
DINOV2_HEIGHT = 364
DINOV2_EMBED_DIM = 384
DINOV2_MODEL_NAME = "dinov2_vits14"

# Burned-in black HUD rectangle in bottom-left corner (95x35 px in 720x364 coordinates)
HUD_MASK_BOX = {
    "x_min": 0,
    "y_min": ORIG_HEIGHT - 35,  # 329
    "x_max": 95,
    "y_max": ORIG_HEIGHT,       # 364
}

# ==============================================================================
# 3. SEMANTIC CLASSES & COLOR MAPPINGS
# ==============================================================================
NUM_CLASSES = 5
IGNORE_INDEX = 0

CLASS_NAMES = {
    0: "ignore",      # Black: unlabeled background, bottom-left HUD box
    1: "road",        # Magenta: drivable surface / dirt track / ruts
    2: "sky",         # Blue: sky / dark background above horizon
    3: "vegetation",  # Cyan: non-drivable bushes / tall grass / edges
    4: "obstacle",    # Yellow: poles / vertical obstacles / hazards
}

# Exact RGB tuples in seg/markdown/*.png
RGB_TO_CLASS = {
    (0, 0, 0): 0,
    (255, 0, 255): 1,
    (0, 0, 255): 2,
    (0, 255, 255): 3,
    (255, 255, 0): 4,
}

# BGR colors for OpenCV tactical HUD visualization
CLASS_COLORS_BGR = np.array([
    [0, 0, 0],        # 0: ignore (transparent)
    [0, 210, 0],      # 1: road (bright tactical green)
    [180, 60, 0],     # 2: sky (muted blue)
    [0, 140, 255],    # 3: vegetation (orange warning boundary)
    [0, 0, 255],      # 4: obstacle (bright red hazard)
], dtype=np.uint8)

# Class weights for CrossEntropyLoss (upweighting thin yellow obstacles & vegetation)
CLASS_WEIGHTS = [0.0, 1.0, 0.8, 1.5, 3.5]

# ==============================================================================
# 4. DATASET SPLIT & TRAINING HYPERPARAMETERS
# ==============================================================================
# Contiguous block split (301 frames total: 001000..001300)
# First 240 frames -> Train, Last 61 frames -> Validation
TRAIN_SPLIT_COUNT = 240
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 25
SEED = 42

# ==============================================================================
# 5. PATH PLANNER, HEALTH MONITOR & CRUISE CONTROL PARAMETERS
# ==============================================================================
# Drivable ROI vertical range (fraction of image height)
ROI_Y_MIN_RATIO = 0.50   # Below horizon
ROI_Y_MAX_RATIO = 0.90   # Above UGV front hood / bottom HUD

# Lookahead horizontal bands inside ROI (from near to far)
NUM_LOOKAHEAD_BANDS = 4
BAND_WEIGHTS = [0.40, 0.30, 0.20, 0.10]  # Nearer bands have higher steering priority

# Obstacle repulsion strength
OBSTACLE_REPULSION_GAIN = 1.5
MIN_ROAD_PIXELS_PER_BAND = 150

# PD Steering Controller & Temporal Smoothing
KP_STEER = 0.70          # Proportional gain
KD_STEER = 0.22          # Derivative gain
EMA_ALPHA = 0.35         # Exponential moving average factor for target centroid smoothing
LANE_DEPARTURE_TH = 0.28 # Normalized lateral offset threshold to trigger LDW warning

# Camera & Link Health Thresholds
LINK_LOSS_MEAN_GRAY_TH = 20.0     # Mean pixel brightness (0-255) below which frame is black
LINK_LOSS_MIN_DURATION_SEC = 1.0  # Minimum seconds of black screen to declare LINK_LOST
GLARE_PIXEL_INTENSITY_TH = 245    # Pixel intensity for IR glare detection
GLARE_AREA_RATIO_TH = 0.12        # Fraction of central ROI saturated to trigger GLARE_WARNING
BUSH_ENTRAPMENT_VEG_RATIO = 0.75  # Vegetation ratio in ROI to trigger OFF_ROAD warning
LOW_ROAD_CONFIDENCE_TH = 0.25     # Below this confidence, UGV reduces speed or halts
