# Drone detection models

Single class `0: waiter_fpv_drone`: an FPV drone sitting on or beside the road, waiting for a vehicle.
Trained with Ultralytics 8.4.183 at imgsz 640 (Colab, `storozha/runs`, data commit `20e0cf6`). Stored with Git LFS,
so run `git lfs pull` after cloning.

| file | source run | params | best val mAP50 / recall | CPU 640x360 (M-series) |
|---|---|---|---|---|
| `drone_yolo11s_640.pt` (default) | `cv-yolo11s-640/fold1/weights/best.pt`, epoch 5 | 9.4 M | 0.733 / 0.642 (fold 1); 4-fold mean mAP50 0.52 | ~55 ms |
| `drone_yolo26n_640.pt` | `yolo26n-640-4/weights/best.pt`, epoch 7, NMS-free | 2.4 M | 0.557 / 0.538 | ~30 ms |

## Why yolo11s is the default
On the 48 frames of `yolo26n-640-4`'s own `val_batch*` mosaics (tiles, about 0.75x scale), counted as frames with
any detection:

| conf | yolo11s | yolo26n |
|---|---|---|
| 0.10 | 43/48 | 2/48 |
| 0.25 | 39/48 | 0/48 |
| 0.35 | 23/48 | 0/48 |

These drones are about 15-25 px, and the nano model barely fires on them. Treat the yolo11s numbers as optimistic:
fold 1 may have trained on some of these frames. It also produces false positives on ruts and puddles, and that is
why the node needs `confirm_hits` frames before it warns.

## Not shipped
- `rtdetr-l-640*`: 66 MB, mAP50 0.50.
- `yolo26n-1280*`: best at epoch 1 and unstable.

All the runs peak in the first 5-17 epochs, and their last-10-epoch mAP50 drops to 0.1-0.4. That points to overfitting
on a small dataset. More data, especially frames from our own cameras, would help more than tuning.

## On the Jetson
Export to TensorRT on the target itself, then point `model:` at the result:

    yolo export model=drone_yolo11s_640.pt format=engine quantize=16 imgsz=640
