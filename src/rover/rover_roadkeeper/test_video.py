#!/usr/bin/env python3
"""
Standalone tester: runs the exact RoadPerceptionCore used by the ROS 2 node on a video
(or a folder of frames) and shows / saves the debug overlay. No ROS needed.

    python test_video.py --video track-1/videos/dev_0008.mp4 --output outputs/test_0008.mp4 --display
    python test_video.py --video dev_0008 --start_sec 60 --max_frames 600
    python test_video.py --video track-1/seg/frames

Keys in --display mode: q = quit, space = pause.
Prints a stability summary at the end (valid ratio, frame-to-frame jitter raw vs smoothed).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rover_roadkeeper import config as C  # noqa: E402
from rover_roadkeeper.perception.pipeline import RoadPerceptionCore  # noqa: E402


def resolve_source(arg: str) -> Path:
    for cand in (Path(arg), C.VIDEOS_DIR / arg, C.VIDEOS_DIR / f"{arg}.mp4", C.DATA_DIR / arg):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"'{arg}' not found (also searched {C.VIDEOS_DIR})")


def frame_source(src: Path, start_sec: float):
    """Yields (frame_bgr) and returns fps via first value."""
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".png"))
        yield 10.0
        for f in files:
            yield cv2.imread(str(f))
        return
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    yield fps
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video path / dev_XXXX name / folder of frames")
    ap.add_argument("--output", default=None, help="output mp4 (default: outputs/test_<name>.mp4 unless --display only)")
    ap.add_argument("--display", action="store_true", help="show live window")
    ap.add_argument("--weights", default=str(C.DEFAULT_CHECKPOINT))
    ap.add_argument("--threshold", type=float, default=C.ROAD_PROB_THRESHOLD)
    ap.add_argument("--start_sec", type=float, default=0.0)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    src = resolve_source(args.video)
    out_path = Path(args.output) if args.output else (None if args.display else C.OUTPUTS_DIR / f"test_{src.stem}.mp4")

    core = RoadPerceptionCore(args.weights, device=args.device, prob_threshold=args.threshold)
    print(f"[*] Source: {src} | device: {core.device} | threshold: {args.threshold}")

    frames = frame_source(src, args.start_sec)
    fps_in = next(frames)
    writer = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (C.FRAME_WIDTH, C.FRAME_HEIGHT))

    offsets, raws, valids, latencies = [], [], [], []
    paused, t0 = False, time.time()
    for i, frame in enumerate(frames):
        if frame is None:
            continue
        if args.max_frames and i >= args.max_frames:
            break

        r = core.process_frame(frame)
        offsets.append(r.offset); raws.append(r.raw_offset); valids.append(r.is_valid)
        latencies.append(r.latency_ms)

        if writer:
            writer.write(r.debug_frame)
        if args.display:
            cv2.imshow("Road offset perception", r.debug_frame)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
        if i % 200 == 0:
            print(f"  frame {i:6d} | offset {r.offset:+.3f} | valid {r.is_valid} | conf {r.road_confidence:.2f} "
                  f"| {r.latency_ms:.1f} ms")

    if writer:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    # ---- stability summary ----
    n = len(offsets)
    if n == 0:
        print("No frames processed."); return
    off = np.array(offsets)
    raw_valid = np.array([x for x in raws if x is not None])
    print("\n" + "=" * 70)
    print(f" Frames: {n} in {time.time() - t0:.1f}s | mean latency {np.mean(latencies):.1f} ms "
          f"({1000.0 / np.mean(latencies):.1f} FPS perception)")
    print(f" Valid road measurement: {np.mean(valids) * 100:.1f}% of frames")
    if len(raw_valid) > 1:
        print(f" Jitter |d offset|/frame : raw {np.mean(np.abs(np.diff(raw_valid))):.4f}  ->  "
              f"smoothed {np.mean(np.abs(np.diff(off))):.4f}")
    print(f" Smoothed offset: mean {off.mean():+.3f}, std {off.std():.3f}, min {off.min():+.3f}, max {off.max():+.3f}")
    if out_path:
        print(f" Saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
