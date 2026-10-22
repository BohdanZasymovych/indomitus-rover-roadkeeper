"""
Global configuration for the Roadkeeper binary road perception pipeline.

Pure Python: NO ROS imports here. Every tunable threshold of the pipeline lives
in this file so that train.py, test_video.py and the ROS 2 node all agree.
"""
from pathlib import Path

# ==============================================================================
# 1. PATHS
# ==============================================================================
PACKAGE_DIR = Path(__file__).resolve().parent.parent          # src/rover/rover_roadkeeper
REPO_ROOT_DIR = PACKAGE_DIR.parents[2]                         # indomitus-rover-roadkeeper

_DATASET_CANDIDATES = [
    PACKAGE_DIR / "track-1",
    PACKAGE_DIR / "track1",
    REPO_ROOT_DIR / "track-1",
    REPO_ROOT_DIR / "track1",
    Path.home() / "def_tech_hackathon" / "track1",
    Path.home() / "def_tech_hackathon" / "track-1",
]
DATA_DIR = next((p for p in _DATASET_CANDIDATES if p.exists()), PACKAGE_DIR / "track-1")

SEG_FRAMES_DIR = DATA_DIR / "seg" / "frames"
SEG_MASKS_DIR = DATA_DIR / "seg" / "markdown"   # folder is literally named "markdown"
VIDEOS_DIR = DATA_DIR / "videos" if (DATA_DIR / "videos").exists() else DATA_DIR

WEIGHTS_DIR = PACKAGE_DIR / "weights"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "road_segmenter_best.pt"

# ==============================================================================
# 2. IMAGE GEOMETRY
# ==============================================================================
# Every frame (any source resolution) is resized to this before perception.
FRAME_WIDTH = 720
FRAME_HEIGHT = 364

# DINOv2 ViT-S/14 needs multiples of 14: 714 = 51*14, 364 = 26*14
DINOV2_MODEL_NAME = "dinov2_vits14"
DINOV2_INPUT_WIDTH = 714
DINOV2_INPUT_HEIGHT = 364
DINOV2_PATCH_SIZE = 14
DINOV2_EMBED_DIM = 384
DINOV2_NUM_LAYERS = 4          # concat patch tokens of the last N transformer blocks

# Burned-in black HUD rectangle (bottom-left, 95x35 px in 720x364 coordinates)
HUD_BOX = (0, FRAME_HEIGHT - 35, 95, FRAME_HEIGHT)   # (x_min, y_min, x_max, y_max)

# ==============================================================================
# 3. BINARY LABELS
# ==============================================================================
ROAD_RGB = (255, 0, 255)       # magenta in seg/markdown/*.png -> road (1)
LABEL_BACKGROUND = 0
LABEL_ROAD = 1
LABEL_IGNORE = 255             # HUD box (and optionally unlabeled black pixels)

# Unlabeled black (0,0,0) pixels (~13%): far darkness. Supervising them as
# background makes the model more conservative about calling things "road".
UNLABELED_AS_BACKGROUND = True

# ==============================================================================
# 4. PREPROCESSING (shared by training and inference!)
# ==============================================================================
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID = (8, 8)

# ==============================================================================
# 5. TRAINING
# ==============================================================================
TRAIN_SPLIT_COUNT = 240        # contiguous split: 001000..001239 train, 001240..001300 val
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
SEED = 42
# pos_weight < 1.0 down-weights the positive (road) term of BCE, i.e. relatively
# penalises false-positive road predictions -> fights "whole field is road".
BCE_POS_WEIGHT = 0.7
DICE_WEIGHT = 0.5

# ==============================================================================
# 6. MASK POST-PROCESSING
# ==============================================================================
ROAD_PROB_THRESHOLD = 0.65     # pixel is road only if p(road) > this (no argmax!)
MORPH_KERNEL_SIZE = 7          # elliptical kernel for open (denoise) + close (fill ruts)
LOWER_HALF_START_RATIO = 0.5   # largest component is chosen by its area in the lower half
MIN_COMPONENT_AREA_PX = 1500   # smaller "largest" components are rejected as noise

# ==============================================================================
# 7. SINGLE-ROI OFFSET + TEMPORAL SMOOTHING
# ==============================================================================
ROI_Y_MIN_RATIO = 0.55         # below horizon
ROI_Y_MAX_RATIO = 0.88         # above hood / HUD box
ROI_BOTTOM_WEIGHT = 2.0        # row weight ramps 1.0 (top of ROI) -> this (bottom)

MIN_ROAD_AREA_RATIO = 0.04     # road must fill >= 4% of the ROI to be a valid measurement
FULL_CONFIDENCE_AREA_RATIO = 0.25  # area ratio at which the area term of confidence saturates

EMA_ALPHA = 0.2                # strong smoothing (0.15..0.25)
MAX_OFFSET_DELTA = 0.05        # rate limiter: max |offset change| per frame
INVALID_DECAY = 0.95           # on invalid frames, held offset decays toward 0 by this factor

# ==============================================================================
# 8. ROS 2 DEFAULTS (used only by ros2_node.py; overridable via ROS parameters)
# ==============================================================================
ROS_NODE_NAME = "road_offset_perception_node"
ROS_IMAGE_TOPIC = "/rover/front_camera/image_raw"
ROS_OFFSET_TOPIC = "/rover/road_offset"
ROS_DEBUG_IMAGE_TOPIC = "/rover/road_perception/debug_image"
