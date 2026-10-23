#!/usr/bin/env python3
"""Run the DINOv2 road segmenter on videos or image folders, without ROS.

Uses the exact RoadSegmenter the ROS 2 node uses, at whatever resolution and
FPS the input has, and writes an overlay video (green = road) with inference
FPS, frame size and road share burned in.

    python test_segmentation_video.py --input track-1/dev_0008.mp4 --show
    python test_segmentation_video.py --input dev_0003.mp4 --output outputs/seg_dev_0003.mp4
    python test_segmentation_video.py --input ~/def_tech_hackathon/track1/videos      # all dev_*.mp4
    python test_segmentation_video.py --input ~/def_tech_hackathon/track1/seg/frames  # + IoU vs labels

Bare names (dev_0008.mp4, track-1/dev_0008.mp4) are also looked up in the
dataset found by config.find_data_dir() (env ROAD_SEG_DATA_DIR).
Keys with --show: q / Esc = quit, space = pause, n = next input.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rover_road_segmentation import config as C  # noqa: E402
from rover_road_segmentation.visualization import info_panel, overlay, side_by_side  # noqa: E402

VIDEO_EXT = {'.mp4', '.avi', '.mkv', '.mov', '.m4v', '.webm'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def resolve_input(path):
    """The path itself, or the same name inside the dataset (e.g. track-1/dev_0008.mp4)."""
    p = Path(path).expanduser()
    if p.exists():
        return p
    data = C.find_data_dir()
    if data is not None:
        candidates = [data / p.name, data / 'videos' / p.name]
        if len(p.parts) > 1:
            candidates.append(data / Path(*p.parts[1:]))     # drop a leading "track-1/"
        for cand in candidates:
            if cand.exists():
                return cand
    raise FileNotFoundError(f'{path} not found (dataset dir: {data})')


def list_jobs(src):
    """[(name, kind, path)] with kind 'video' or 'images'."""
    if src.is_file():
        return [(src.stem, 'video', src)]
    videos = sorted(f for f in src.iterdir() if f.suffix.lower() in VIDEO_EXT)
    if videos:
        return [(v.stem, 'video', v) for v in videos]
    if any(f.suffix.lower() in IMAGE_EXT for f in src.iterdir()):
        return [(src.parent.name + '_' + src.name if src.name == 'frames' else src.name, 'images', src)]
    raise FileNotFoundError(f'no videos or images in {src}')


def iter_video(path, start_sec, stride):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield frame, None, f'{start_sec + i / fps:7.2f}s'
            i += 1
    finally:
        cap.release()


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    meta = (cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return meta


def iter_images(folder, stride, gt_dir):
    from rover_road_segmentation.dataset import rgb_mask_to_labels
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXT)[::stride]
    for f in files:
        frame = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        labels = None
        if gt_dir is not None and (gt_dir / f'{f.stem}.png').exists():
            m = cv2.imread(str(gt_dir / f'{f.stem}.png'), cv2.IMREAD_COLOR)
            if m.shape[:2] != frame.shape[:2]:
                m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            labels = rgb_mask_to_labels(cv2.cvtColor(m, cv2.COLOR_BGR2RGB))
        yield frame, labels, f.stem


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_job(seg, name, kind, path, args):
    if kind == 'video':
        fps_in, total, w, h = video_meta(path)
        frames = iter_video(path, args.start_sec, args.stride)
        out_fps = fps_in / args.stride
        print(f'[*] {name}: {w}x{h} @ {fps_in:.1f} fps, {total} frames')
    else:
        gt_dir = args.gt_dir
        if gt_dir is None and path.name == 'frames' and (path.parent / 'markdown').is_dir():
            gt_dir = path.parent / 'markdown'
        frames = iter_images(path, args.stride, gt_dir)
        out_fps, total = args.fps, None
        print(f'[*] {name}: image folder{" (IoU vs " + str(gt_dir) + ")" if gt_dir else ""}')

    out_path = Path(args.output) if args.output and len(args.jobs) == 1 else \
        Path(args.output_dir) / f'seg_{name}.mp4'
    writer, out_size = None, None
    inter = union = 0
    stats = {'n': 0, 'empty': 0, 'road': [], 'ms': []}
    paused = False

    for frame, labels, tag in frames:
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        mask = seg.segment(frame, 'bgr')
        dt = time.perf_counter() - t0

        assert mask.shape == (h, w) and mask.dtype == np.uint8 and mask.max(initial=0) <= 1
        road = float(mask.mean())
        stats['n'] += 1
        stats['empty'] += int(road == 0.0)
        stats['road'].append(road)
        stats['ms'].append(dt * 1000.0)

        lines = [f'{name}  {tag}', f'{w}x{h}  infer {1.0 / max(dt, 1e-6):5.1f} FPS ({dt * 1000:4.0f} ms)',
                 f'road {road * 100:5.1f}%' + ('   NO ROAD' if road == 0 else '')]
        if labels is not None:
            valid = labels != C.LABEL_IGNORE
            gt = labels == C.LABEL_ROAD
            i_, u_ = int(np.sum(mask.astype(bool) & gt & valid)), int(np.sum((mask.astype(bool) | gt) & valid))
            inter, union = inter + i_, union + u_
            lines.append(f'IoU {100.0 * i_ / max(u_, 1):5.1f}%')

        if args.layout == 'side':
            vis = side_by_side(frame, mask, lines, args.alpha)
        else:
            vis = info_panel(overlay(frame, mask, args.alpha), lines)
        if out_size is None:
            out_size = (vis.shape[1], vis.shape[0])
        if (vis.shape[1], vis.shape[0]) != out_size:
            vis = cv2.resize(vis, out_size)

        if not args.no_save:
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), out_fps, out_size)
            writer.write(vis)

        if args.show:
            disp = vis
            if disp.shape[1] > args.max_display_width:
                s = args.max_display_width / disp.shape[1]
                disp = cv2.resize(disp, (int(disp.shape[1] * s), int(disp.shape[0] * s)))
            cv2.imshow('road segmentation', disp)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key in (ord('q'), 27):
                args.quit = True
                break
            if key == ord('n'):
                break
            if key == ord(' '):
                paused = not paused

        if args.max_frames and stats['n'] >= args.max_frames:
            break
        if stats['n'] % 100 == 0:
            done = f'/{total // args.stride}' if total else ''
            print(f'    {stats["n"]}{done} frames, {np.mean(stats["ms"][-100:]):.0f} ms/frame')

    if writer is not None:
        writer.release()
    if stats['n']:
        ms = np.array(stats['ms'][5:] or stats['ms'])
        msg = (f'[=] {name}: {stats["n"]} frames | {1000.0 / ms.mean():.1f} FPS ({ms.mean():.0f} ms) | '
               f'mean road {100 * np.mean(stats["road"]):.1f}% | empty masks {100.0 * stats["empty"] / stats["n"]:.1f}%')
        if union:
            msg += f' | IoU {100.0 * inter / union:.1f}%'
        if writer is not None:
            msg += f' -> {out_path}'
        print(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='video file, folder of videos, or folder of images')
    ap.add_argument('--output', help='output .mp4 (single input only); default outputs/seg_<name>.mp4')
    ap.add_argument('--output_dir', default=str(C.OUTPUTS_DIR))
    ap.add_argument('--weights', default=str(C.DEFAULT_WEIGHTS))
    ap.add_argument('--device', default='auto')
    ap.add_argument('--no_fp16', action='store_true')
    ap.add_argument('--threshold', type=float, default=C.ROAD_PROB_THRESHOLD)
    ap.add_argument('--no_hud_mask', action='store_true', help='do not blank the bottom-left HUD box')
    ap.add_argument('--layout', choices=['overlay', 'side'], default='side',
                    help='side = [overlay | raw mask*255], overlay = overlay only')
    ap.add_argument('--alpha', type=float, default=0.45)
    ap.add_argument('--stride', type=int, default=1, help='process every N-th frame')
    ap.add_argument('--start_sec', type=float, default=0.0)
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--fps', type=float, default=10.0, help='output FPS for image folders')
    ap.add_argument('--gt_dir', type=Path, help='label masks for an image folder (auto for seg/frames)')
    ap.add_argument('--show', action='store_true', help='display with cv2.imshow')
    ap.add_argument('--max_display_width', type=int, default=1600)
    ap.add_argument('--no_save', action='store_true')
    args = ap.parse_args()
    args.stride = max(1, args.stride)
    args.quit = False

    from rover_road_segmentation.segmenter import PostProcessParams, RoadSegmenter
    src = resolve_input(args.input)
    args.jobs = list_jobs(src)
    seg = RoadSegmenter(args.weights, device=args.device, use_fp16=not args.no_fp16,
                        post=PostProcessParams(threshold=args.threshold, mask_hud=not args.no_hud_mask))
    print(f'[*] {C.DINOV2_MODEL_NAME} on {seg.device} (fp16={seg.use_fp16}), threshold {args.threshold}, '
          f'{len(args.jobs)} input(s)')
    try:
        for name, kind, path in args.jobs:
            run_job(seg, name, kind, path, args)
            if args.quit:
                break
    finally:
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
