"""Constants shared by training, the video tester and the ROS 2 node.

Pure Python, no ROS and no torch: every number the pipeline depends on lives
here so that train_dinov2.py, test_segmentation_video.py and
segmentation_node.py cannot disagree.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent.parent          # src/rover/rover_road_segmentation
REPO_ROOT_DIR = PACKAGE_DIR.parents[2] if len(PACKAGE_DIR.parents) > 2 else PACKAGE_DIR

WEIGHTS_FILENAME = 'road_segmenter_dinov2.pt'
DEFAULT_WEIGHTS = PACKAGE_DIR / 'weights' / WEIGHTS_FILENAME
OUTPUTS_DIR = PACKAGE_DIR / 'outputs'

#: Where the hackathon dataset (dev_XXXX.mp4 + seg/) may live. ROAD_SEG_DATA_DIR wins.
_DATA_CANDIDATES = [
    PACKAGE_DIR / 'track-1', PACKAGE_DIR / 'track1',
    REPO_ROOT_DIR / 'track-1', REPO_ROOT_DIR / 'track1',
    Path.home() / 'def_tech_hackathon' / 'track1',
    Path.home() / 'def_tech_hackathon' / 'track-1',
]


def find_data_dir():
    """The track-1 dataset root, or None if it is nowhere to be found."""
    env = os.environ.get('ROAD_SEG_DATA_DIR')
    if env:
        return Path(env).expanduser()
    return next((p for p in _DATA_CANDIDATES if p.exists()), None)


# ---------------------------------------------------------------------------
# DINOv2 geometry
# ---------------------------------------------------------------------------
DINOV2_MODEL_NAME = 'dinov2_vits14'
DINOV2_PATCH_SIZE = 14
DINOV2_EMBED_DIM = 384
DINOV2_NUM_LAYERS = 4          # patch tokens of the last N blocks are concatenated
HEAD_HIDDEN_DIM = 128

# Default inference resolution, both multiples of patch size (14):
# 448x224 provides a 2.6x token reduction (512 vs 1326 patches) and a 6.7x attention speedup
# over 714x364 while preserving 98.1% validation road IoU.
INFER_WIDTH = 32 * DINOV2_PATCH_SIZE     # 448
INFER_HEIGHT = 16 * DINOV2_PATCH_SIZE    # 224

# PyTorch runtime acceleration flags
TORCH_CUDNN_BENCHMARK = True
TORCH_MATMUL_PRECISION = 'high'

# ---------------------------------------------------------------------------
# Labels (track-1/seg/markdown/*.png)
# ---------------------------------------------------------------------------
ROAD_RGB = (255, 0, 255)       # magenta = road (1), everything else = 0
LABEL_BACKGROUND = 0
LABEL_ROAD = 1
LABEL_IGNORE = 255

#: Geometry of the labelled frames; the HUD box is defined in these pixels.
REF_WIDTH, REF_HEIGHT = 720, 364
#: Burned-in HUD box, bottom-left: x 0..95, y H-35..H (in REF_* pixels).
HUD_WIDTH_PX, HUD_HEIGHT_PX = 95, 35

# ---------------------------------------------------------------------------
# Preprocessing (identical in training and inference)
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID = (8, 8)
#: Mean grey level (0..255) below which a frame is treated as "no video"
#: (link-loss screen, lens cap) and gets an empty mask without inference.
MIN_FRAME_BRIGHTNESS = 8.0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_SPLIT_COUNT = 240        # contiguous: 001000..001239 train, 001240..001300 val
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
SEED = 42
#: < 1 down-weights the road term of the BCE: false-positive road costs more.
BCE_POS_WEIGHT = 0.7
DICE_WEIGHT = 0.5
#: Share of training samples replaced by a synthetic "no road" frame
#: (black link-loss screen / glare blob) labelled all background.
SYNTHETIC_NEGATIVE_PROB = 0.05

# ---------------------------------------------------------------------------
# Mask post-processing (scaled to whatever the output resolution is)
# ---------------------------------------------------------------------------
ROAD_PROB_THRESHOLD = 0.65     # road only if p(road) > this
MORPH_KERNEL_REF = 7           # [px at REF_HEIGHT] open (denoise) + close (fill ruts)
LOWER_HALF_START = 0.5         # the kept component is the largest one below this row fraction
MIN_COMPONENT_FRACTION = 0.01  # of the lower-half area; a smaller "largest" blob is noise

# ---------------------------------------------------------------------------
# Layer A: Mask Regularization, Rut Bridging & Temporal Smoothing
# ---------------------------------------------------------------------------
TEMPORAL_ALPHA = 0.60          # weight of current frame in EMA: p_hat = alpha * p + (1 - alpha) * p_prev
BRIDGE_KERNEL_WIDTH = 35       # wide horizontal closing kernel [px at INFER_WIDTH] to bridge ruts
BRIDGE_KERNEL_HEIGHT = 9       # horizontal closing kernel height
VERTICAL_KERNEL_HEIGHT = 25    # vertical closing kernel height [px at INFER_HEIGHT] for travel continuity
VERTICAL_KERNEL_WIDTH = 9      # vertical closing kernel width
ENABLE_CONVEX_SPAN_FILL = True # fill horizontal gap between ruts on lower scanlines
MAX_SPAN_GAP_FRACTION = 0.30   # max horizontal gap fraction across ruts to bridge (e.g. 30% of width)
SPAN_FILL_START_Y = 0.45       # row fraction above which span fill is applied (lower half of frame)
