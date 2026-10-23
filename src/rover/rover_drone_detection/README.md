# rover_drone_detection

The node detects **waiter FPV drones**: ambush drones that sit on or beside the road and wait for a vehicle. It works
on a rover camera stream and:

- **warns** the operator with a confirmed level: CLEAR, WARN, or DANGER;
- **draws** the boxes and a banner on an annotated image;
- **logs** every sighting with the rover's pose and the drone's estimated position.

It never commands the drive.

```
cameras/<camera>/image_raw ─┐
cameras/<camera>/camera_info ┼─> drone_detection_node ─> drones/detections       vision_msgs/Detection2DArray
TF  odom <- base_footprint   │                          ─> drones/threat           indomitus_interfaces/DroneThreat
    odom <- <camera frame>  ─┘                          ─> drones/image_annotated  sensor_msgs/Image
                                                        ─> ~/drone_sightings/<date>/sightings.jsonl + snapshots/
```

## Install and run
```bash
git lfs pull                                                    # the models are in LFS
pip3 install -r src/rover/rover_drone_detection/requirements.txt   # ultralytics; on Jetson install JetPack torch first
colcon build --packages-up-to rover_drone_detection
ros2 launch rover_drone_detection drone_detection.launch.py                  # arducam_mast, yolo11s
ros2 launch rover_drone_detection drone_detection.launch.py camera:=arducam_front_bottom model:=drone_yolo26n_640.pt
ros2 topic echo /rover/drones/threat
```
The Docker image already has torch 2.10 (CUDA) and ultralytics, on both the Jetson and the dev image (see
`docker/Dockerfile`); the `pip3 install` line is only for running outside Docker.

`rover_cameras` publishes ROS images at `ros_fps` (5 by default in `cameras.yaml`). Raise that for the camera you
watch if you want the full detection `rate` (10 Hz).

## How a detection becomes a warning
The rules live in `threat.py` and are set in `config/drone_detection.yaml`.

- A drone has to appear in `confirm_hits` (3) frames within `confirm_window` (1 s) to raise the level.
  One stray frame never does.
- **DANGER** applies when the box is tall (at least `danger_size` of the image height), or when its bottom is in the
  path zone. The path zone is the lower-centre of the frame, `path_half_width` either side of the centre and below
  `horizon`; the annotated image outlines it.
- Any other confirmed drone is **WARN**.
- The level holds through misses and drops to CLEAR after `clear_after` (2 s) without a detection.

## Sighting log
One JSON line per record, written on every level change and every `log_interval` (1 s) while the level is WARN or
DANGER:

```json
{"stamp": 1792702960.46, "wall_time": 1792702960.46, "level": "DANGER", "model": "drone_yolo11s_640.pt",
 "frame_id": "arducam_mast_optical_frame", "image_size": [960, 600], "count": 1, "confidence": 0.50,
 "bbox": [512, 380, 540, 401], "bearing": -0.03, "size": 0.035, "range_est": 9.8,
 "detections": [{"bbox": [512, 380, 540, 401], "score": 0.50}],
 "rover": {"frame": "odom", "x": 12.1, "y": -3.4, "yaw": 0.12},
 "drone": {"frame": "odom", "x": 21.8, "y": -3.9},
 "gps": null, "snapshot": "snapshots/000240_463_DANGER.jpg"}
```

Field notes:
- `bearing` is in rad, positive to the left.
- `range_est` comes from the apparent size: `fx * drone_size_m / box_width`. It is rough. With no camera calibration,
  `fx` comes from `hfov_deg`.
- `drone` is that point on the ground, put into `world_frame` through TF.
- A field is `null` when what it needs is missing: no TF, no range estimate, or no `gps_topic`.

## Layout
| file | what |
|---|---|
| `threat.py` | detections → confirmed level (pure, tested) |
| `geometry.py` | intrinsics, bearing, range, camera → world (pure, tested) |
| `sighting_log.py` | JSONL + snapshots (pure, tested) |
| `image_msg.py` | Image <-> BGR numpy without cv_bridge, which breaks under NumPy 2 (pure, tested) |
| `draw.py` | the annotated image |
| `detector.py` | Ultralytics wrapper; `.pt`, `.onnx` or TensorRT `.engine` |
| `drone_detection_node.py` | the ROS side only |
| `models/` | the two shipped checkpoints and how they were chosen (see `models/README.md`) |

Run the tests without ROS (they need numpy, opencv and pytest):

    cd src/rover/rover_drone_detection && python3 -m pytest test
