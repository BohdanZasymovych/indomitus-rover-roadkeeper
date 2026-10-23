#!/usr/bin/env python3
"""End-to-End Demo: Video -> Road Segmentation -> Road Deviation (rover_road_assist) -> Dashboard HUD.

Demonstrates the exact multi-module architecture:
  1. Input Video (dev_0001.mp4 ... dev_0012.mp4)
  2. DINOv2 Binary Road Segmentation (rover_road_segmentation) -> outputs 0/1 bitmask
  3. Road Offset & Heading Estimation (rover_road_assist.road_deviation) -> computes lateral offset & heading error
  4. Real-time Telemetry Dashboard (scanlines, vehicle centerline, offset gauge, heading arrow)

Usage:
  python test_end_to_end_demo.py --input dev_0001.mp4 --show
  python test_end_to_end_demo.py --input dev_0008.mp4 --output outputs/demo_dev_0008.mp4
  python test_end_to_end_demo.py --input dev_0003.mp4 --max_frames 200

Keys during --show:
  [Space] : Pause / Resume
  [q/Esc] : Quit
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure rover_road_segmentation and rover_road_assist are importable
PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parents[2]
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'rover' / 'rover_road_assist'))

from rover_road_assist.road_deviation import (  # noqa: E402
    DeviationFilter, estimate_deviation)
from rover_road_segmentation import config as C  # noqa: E402
from rover_road_segmentation.dashboard_visualizer import render_dashboard  # noqa: E402
from rover_road_segmentation.segmenter import PostProcessParams, RoadSegmenter  # noqa: E402


def resolve_video(input_path):
    p = Path(input_path).expanduser()
    if p.exists():
        return p
    data_dir = C.find_data_dir()
    if data_dir:
        candidates = [
            data_dir / p.name,
            data_dir / 'videos' / p.name,
        ]
        if len(p.parts) > 1:
            candidates.append(data_dir / Path(*p.parts[1:]))
        for cand in candidates:
            if cand.exists():
                return cand
    raise FileNotFoundError(f"Could not find video {input_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='dev_0001.mp4', help='Video filename or path')
    ap.add_argument('--output', help='Output MP4 path; default outputs/demo_pipeline_<name>.mp4')
    ap.add_argument('--weights', default=str(C.DEFAULT_WEIGHTS), help='DINOv2 head checkpoint')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--threshold', type=float, default=C.ROAD_PROB_THRESHOLD)
    ap.add_argument('--temporal_alpha', type=float, default=C.TEMPORAL_ALPHA,
                    help=f'EMA weight for current frame probability (default: {C.TEMPORAL_ALPHA})')
    ap.add_argument('--bridge_width', type=int, default=C.BRIDGE_KERNEL_WIDTH,
                    help=f'Horizontal rut bridging kernel width (default: {C.BRIDGE_KERNEL_WIDTH})')
    ap.add_argument('--no_span_fill', action='store_true',
                    help='disable scanline row span stabilization')
    ap.add_argument('--smoothing_alpha', type=float, default=0.5, help='DeviationFilter alpha (matches rover_road_assist)')
    ap.add_argument('--stride', type=int, default=1, help='Process every N-th frame')
    ap.add_argument('--start_sec', type=float, default=0.0)
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--show', action='store_true', help='Display live dashboard with cv2.imshow')
    ap.add_argument('--no_save', action='store_true', help='Skip saving output video')
    args = ap.parse_args()

    video_path = resolve_video(args.input)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        ap.error(f"Cannot open video {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    # 1. Initialize our DINOv2 segmenter
    post_params = PostProcessParams(
        threshold=args.threshold,
        temporal_alpha=args.temporal_alpha,
        bridge_kernel_width=args.bridge_width,
        enable_convex_span_fill=not args.no_span_fill,
    )
    seg = RoadSegmenter(
        weights=args.weights,
        device=args.device,
        post=post_params,
    )

    # 2. Initialize rover_road_assist's deviation filter
    dev_filter = DeviationFilter(alpha=args.smoothing_alpha)

    print("=" * 80)
    print(f"[*] Starting End-to-End Pipeline Demo")
    print(f"[*] Input video:     {video_path.name} ({w_in}x{h_in} @ {fps_in:.1f} FPS, {total_frames} frames)")
    print(f"[*] Segmentation:    {C.DINOV2_MODEL_NAME} on {seg.device} (threshold={args.threshold})")
    print(f"[*] Downstream Node: rover_road_assist.road_deviation (alpha={args.smoothing_alpha})")
    print("=" * 80)

    # Output writer
    out_path = Path(args.output) if args.output else (C.OUTPUTS_DIR / f"demo_pipeline_{video_path.stem}.mp4")
    writer = None
    if not args.no_save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps_in / max(1, args.stride),
            (w_in, h_in),
        )

    frame_idx = 0
    processed_count = 0
    t_start = time.perf_counter()
    paused = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if (frame_idx - 1) % args.stride != 0:
                continue

            t0 = time.perf_counter()

            # --- STEP 1: DINOv2 Road Segmentation ---
            # Output is strictly (H, W) uint8 with values 0 and 1
            mask = seg.segment(frame, 'bgr')

            # --- STEP 2: rover_road_assist Offset Estimation ---
            # Downstream node expects 0 = no road, > 0 = road
            raw_deviation = estimate_deviation(
                mask,
                row_fractions=(0.6, 0.7, 0.8, 0.9),
                horizon_fraction=0.4,
                target_col_fraction=0.5,
                min_run_fraction=0.05,
                min_rows=2,
                min_road_fraction=0.02,
                full_confidence_road_fraction=0.2,
            )
            filtered_deviation = dev_filter.update(raw_deviation)

            dt = time.perf_counter() - t0
            fps_live = 1.0 / max(dt, 1e-6)

            # --- STEP 3: Dashboard HUD Rendering ---
            dashboard = render_dashboard(
                frame_bgr=frame,
                mask=mask,
                deviation=filtered_deviation,
                fps=fps_live,
                latency_ms=dt * 1000.0,
            )

            if writer is not None:
                writer.write(dashboard)

            processed_count += 1
            if processed_count % 30 == 0:
                deg = np.rad2deg(filtered_deviation.heading_error)
                print(f" Frame {frame_idx:4d}/{total_frames} | {fps_live:4.1f} FPS ({dt*1000:3.0f} ms) | "
                      f"Offset: {filtered_deviation.lateral_offset:+.2f} hw | Heading: {deg:+.1f}° | "
                      f"Conf: {filtered_deviation.confidence*100:3.0f}% | "
                      f"{'VALID' if filtered_deviation.valid else 'NO ROAD'}")

            if args.show:
                disp = dashboard
                if w_in > 1280:
                    disp = cv2.resize(dashboard, (1280, int(1280 * h_in / w_in)))
                cv2.imshow("Road Assist Pipeline Demo", disp)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord(' '):
                    paused = not paused

            if args.max_frames and processed_count >= args.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    print("=" * 80)
    print(f"[DONE] Processed {processed_count} frames in {total_time:.1f}s ({processed_count/max(total_time,1e-6):.1f} FPS)")
    if writer is not None:
        print(f"Saved demo video to: {out_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
