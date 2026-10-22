#!/usr/bin/env python3
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure package root is in python path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rover_roadkeeper.config import (
    DEFAULT_CHECKPOINT,
    OUTPUTS_DIR,
    VIDEOS_DIR,
    ORIG_WIDTH,
    ORIG_HEIGHT,
)
from rover_roadkeeper.models.dinov2_segmenter import DINOv2Segmenter
from rover_roadkeeper.control.health_monitor import HealthMonitor
from rover_roadkeeper.control.path_planner import RowWiseCentroidPathPlanner
from rover_roadkeeper.control.pid_controller import PDSteeringController
from rover_roadkeeper.utils.hud_visualizer import HUDVisualizer


def resolve_video_path(video_arg: str) -> Path:
    """Finds video path from filename, relative path, or search in VIDEOS_DIR."""
    path = Path(video_arg)
    if path.exists():
        return path

    candidate = VIDEOS_DIR / video_arg
    if candidate.exists():
        return candidate

    if not video_arg.endswith(".mp4"):
        candidate_mp4 = VIDEOS_DIR / f"{video_arg}.mp4"
        if candidate_mp4.exists():
            return candidate_mp4

    raise FileNotFoundError(f"Video file not found: {video_arg} (searched in {VIDEOS_DIR})")


def run_demo(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Roadkeeper Cruise Demo on device: {device}")

    # 1. Resolve video input
    video_path = resolve_video_path(args.video)
    print(f"[*] Input Video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Video Specs: {width_in}x{height_in} @ {fps_in:.1f} FPS, {total_frames} total frames")

    # 2. Output Video Writer
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{video_path.stem}_cruise_demo.mp4"
    out_path = out_dir / out_filename

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (ORIG_WIDTH, ORIG_HEIGHT))

    # 3. Load Model
    print(f"[*] Loading perception model...")
    model = DINOv2Segmenter(pretrained_backbone=True).to(device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"[*] Loading trained weights from {ckpt_path}...")
        state = torch.load(str(ckpt_path), map_location=device)
        model.head.load_state_dict(state.get("head_state_dict", state))
    else:
        print(f"[!] Checkpoint {ckpt_path} not found. Running with initialized weights.")
    model.eval()

    # 4. Initialize Pipeline Modules
    health_monitor = HealthMonitor()
    path_planner = RowWiseCentroidPathPlanner(image_width=ORIG_WIDTH, image_height=ORIG_HEIGHT)
    controller = PDSteeringController()
    visualizer = HUDVisualizer(width=ORIG_WIDTH, height=ORIG_HEIGHT)

    # 5. Process Frames
    max_frames = min(args.max_frames, total_frames) if args.max_frames > 0 else total_frames
    pbar = tqdm(total=max_frames, desc=f"Rendering {video_path.stem}")

    frame_idx = 0
    t_start = time.time()
    rolling_fps = fps_in

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        f_start = time.time()
        sim_time_sec = frame_idx / fps_in

        # A. Perception
        class_mask, road_conf_map, mean_conf = model.predict(frame, device=device)

        # B. Health & Link Monitor
        health_state, diagnostics = health_monitor.update(frame, class_mask, current_time_sec=sim_time_sec)

        # C. Path Planning
        plan = path_planner.plan(class_mask, road_conf_map)

        # D. PD Cruise Controller
        cmd = controller.compute(
            raw_error=plan.normalized_error,
            road_confidence=plan.road_confidence,
            is_road_visible=plan.is_road_visible,
            health_state=health_state,
            current_time_sec=sim_time_sec,
            dt_fallback=1.0 / fps_in,
        )

        # Compute FPS
        f_time = max(time.time() - f_start, 1e-4)
        current_fps = 1.0 / f_time
        rolling_fps = 0.9 * rolling_fps + 0.1 * current_fps

        # E. Render Tactical HUD
        hud_frame = visualizer.render(
            frame_bgr=frame,
            class_mask=class_mask,
            plan=plan,
            command=cmd,
            health_state=health_state,
            fps=rolling_fps,
        )

        writer.write(hud_frame)

        if args.show:
            cv2.imshow("Roadkeeper Autonomous Cruise HUD", hud_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        pbar.update(1)

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    print(f"\n[DONE] Finished processing {frame_idx} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)")
    print(f"[+] Output Demo Video saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Roadkeeper Cruise Control Demo on UGV video")
    parser.add_argument("--video", type=str, default="dev_0001.mp4", help="Video name or path (e.g. dev_0001.mp4)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUTS_DIR), help="Output directory")
    parser.add_argument("--max_frames", type=int, default=0, help="Maximum frames to process (0 = all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--show", action="store_true", help="Display window while processing")
    args = parser.parse_args()

    run_demo(args)
