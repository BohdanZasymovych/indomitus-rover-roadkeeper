# `rover_road_segmentation`

Lightweight, few-shot **DINOv2 (`dinov2_vits14`) Binary Road Segmentation** package for the Indomitus Rover.

It provides a clean, modular perception pipeline that:
1. Subscribes to the rover's camera feed (`sensor_msgs/msg/Image`).
2. Performs binary road segmentation (`0` = non-road, `1` = road) robustly across different lighting and terrain conditions.
3. Dynamically handles **any input resolution** (e.g., `1280x720`, `720x364`, `700x364`, `2688x1520`).
4. Publishes a binary mask (`sensor_msgs/msg/Image`, `encoding="mono8"`, values `0` and `1`) directly consumed by [`rover_road_assist`](../rover_road_assist).
5. Provides a ROS-independent CLI tool (`test_segmentation_video.py`) to run inference on dataset videos with interactive visualization.

---

## Architecture & Integration with `rover_road_assist`

```
                                      Indomitus Rover Perception & Assist Pipeline
                                      ────────────────────────────────────────────

  [ Camera Node ]                   [ road_segmentation_node ]               [ road_deviation_node ]              [ road_assist_node ]
  (rover_cameras)                  (rover_road_segmentation)                  (rover_road_assist)                 (rover_road_assist)
         │                                      │                                       │                                   │
         │  /rover/cameras/.../image_raw        │                                       │                                   │
         │  (rgb8/bgr8/mono8, any res)          │                                       │                                   │
         └─────────────────────────────────────►│                                       │                                   │
                                                │  /rover/perception/road_mask          │                                   │
                                                │  (mono8, 0/1, exact input res)        │                                   │
                                                └──────────────────────────────────────►│                                   │
                                                                                        │  /rover/road/state                │
                                                                                        │  (indomitus_interfaces/           │
                                                                                        │   RoadDeviation)                  │
                                                                                        └──────────────────────────────────►│
                                                                                                                            │  cmd_vel_remote
                                                                                                                            ▼
```

### Interface & Data Contract
- **Input Topic:** `/rover/cameras/arducam_front_bottom/image_raw` (or `/rover/cameras/arducam_mast/image_raw`, customizable via parameter).
- **Output Topic:** `/rover/perception/road_mask`
  - Message: `sensor_msgs/msg/Image`
  - Encoding: `mono8` (`step = width`, `dtype = uint8`)
  - Pixel values: Strictly `0` (non-road / background) and `1` (road)
  - Time synchronization: The incoming camera `header` (`stamp` and `frame_id`) is preserved verbatim so downstream nodes can track frame age (`max_road_age: 0.3s`).
- **Debug Topic:** `~/debug_overlay` (published when `publish_debug_vis: true` and active subscribers exist).

---

## Quickstart: Standalone Video Demo

## End-to-End Pipeline Demo (Video -> DINOv2 -> rover_road_assist Offset -> Dashboard HUD)

To validate the entire chain (reading video, DINOv2 road segmentation, `rover_road_assist` offset calculation, and HUD telemetry visualization):

```bash
# 1. Activate the environment
source src/rover/rover_roadkeeper/.venv/bin/activate

# 2. Run the end-to-end pipeline demo with live HUD visualization
python src/rover/rover_road_segmentation/test_end_to_end_demo.py \
    --input dev_0001.mp4 \
    --show

# 3. Save demonstration video to outputs/
python src/rover/rover_road_segmentation/test_end_to_end_demo.py \
    --input dev_0008.mp4 \
    --output src/rover/rover_road_segmentation/outputs/demo_pipeline_dev_0008.mp4
```

---

## Standalone Road Segmentation Tester

To test raw segmentation only on videos (`dev_0001.mp4` through `dev_0012.mp4`):

```bash
# Interactive GUI demo
python src/rover/rover_road_segmentation/test_segmentation_video.py \
    --input /home/marko/def_tech_hackathon/track1/videos/dev_0001.mp4 \
    --show
```

---

## Multi-Node ROS 2 Pipeline & Launch

### 1. Launch Complete End-to-End ROS 2 Demo Pipeline
Plays a video file into camera topic, segments road mask, calculates offset via `rover_road_assist`, and displays live HUD:
```bash
ros2 launch rover_road_segmentation demo_road_pipeline.launch.py video:=track-1/videos/dev_0001.mp4
```

### 2. Launch Segmentation + Downstream Road Deviation Node (Real Hardware Camera)
```bash
ros2 launch rover_road_segmentation road_segmentation.launch.py
```

### 2. Launch with Debug Overlay Enabled
```bash
ros2 launch rover_road_segmentation road_segmentation.launch.py debug_vis:=true
```
In `rqt_image_view` or RViz, view:
- `/rover/road_segmentation_node/debug_overlay` (colorized overlay)
- `/rover/road/state` (deviation telemetry from `rover_road_assist`)

### 3. Launch with Custom Camera Topic
```bash
ros2 launch rover_road_segmentation road_segmentation.launch.py \
    image_topic:=/rover/cameras/arducam_mast/image_raw
```

---

## Training & Evaluation

The package includes pre-trained weights in `weights/road_segmenter_dinov2.pt` (Val IoU: 98.6%, ~45 FPS on 720p).

To re-train or evaluate:
```bash
# Evaluate existing weights
python src/rover/rover_road_segmentation/train_dinov2.py --eval_only

# Train new head on track-1/seg (30 epochs, contiguous split)
python src/rover/rover_road_segmentation/train_dinov2.py --epochs 30
```

---

## Running Unit Tests
```bash
python -m pytest src/rover/rover_road_segmentation/test -v
```
