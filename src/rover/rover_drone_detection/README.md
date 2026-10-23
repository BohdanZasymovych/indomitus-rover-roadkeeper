# rover_drone_detection

The node detects **waiter FPV drones**: ambush drones that sit on or beside the road and wait for a vehicle. It runs
YOLO on a rover camera stream, publishes the boxes, and draws them on an annotated image for the operator.

It never commands the drive.

```
cameras/<camera>/image_raw ─> drone_detection_node ─> drones/detections       vision_msgs/Detection2DArray
                                                   ─> drones/image_annotated  sensor_msgs/Image
```

## Install and run
```bash
git lfs pull                                                    # the models are in LFS
pip3 install -r src/rover/rover_drone_detection/requirements.txt   # ultralytics; on Jetson install JetPack torch first
colcon build --packages-up-to rover_drone_detection
ros2 launch rover_drone_detection drone_detection.launch.py                  # arducam_mast, yolo11s
ros2 launch rover_drone_detection drone_detection.launch.py camera:=arducam_front_bottom model:=drone_yolo26n_640.pt
ros2 run rqt_image_view rqt_image_view /rover/drones/image_annotated         # watch the boxes
```
The Docker image already has torch 2.10 (CUDA) and ultralytics, on both the Jetson and the dev image (see
`docker/Dockerfile`); the `pip3 install` line is only for running outside Docker.

`rover_cameras` publishes ROS images at `ros_fps` (5 by default in `cameras.yaml`). Raise that for the camera you
watch if you want the full detection `rate` (10 Hz). The annotated image is only drawn while something subscribes
to it.

## Layout
| file | what |
|---|---|
| `image_msg.py` | Image <-> BGR numpy without cv_bridge, which breaks under NumPy 2 (pure, tested) |
| `detector.py` | Ultralytics wrapper; `.pt`, `.onnx` or TensorRT `.engine` |
| `draw.py` | the annotated image (tested) |
| `drone_detection_node.py` | the ROS side only |
| `models/` | the two shipped checkpoints and how they were chosen (see `models/README.md`) |

Run the tests without ROS (they need numpy, opencv and pytest):

    cd src/rover/rover_drone_detection && python3 -m pytest test
