"""
accuracy_comparison.py — Detection accuracy comparison: Framework A vs Framework B

Since we have no ground truth labels, this script measures:

1. Total detections per video per framework
2. Detection rate (frames with faces / total frames)
3. Confidence score distribution (mean, min, max, std)
4. Face count per frame distribution
5. Frame-level agreement between A and B
   (same frame → do both detect roughly same number of faces?)

Outputs:
    results/accuracy_comparison_<timestamp>.txt
    results/accuracy_comparison_<timestamp>.csv

Usage:
    python3 accuracy_comparison.py
    python3 accuracy_comparison.py --videos ../../videos --frames 200
"""

import av
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import cv2
import numpy as np
import csv
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict

from config import Config
from utils.trt_inference import TRTFaceDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("accuracy_comparison")

Gst.init(None)

Path("../../results").mkdir(parents=True, exist_ok=True)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrameDetection:
    frame_idx:   int
    framework:   str
    video_file:  str
    num_faces:   int
    confidences: List[float] = field(default_factory=list)


@dataclass
class VideoAccuracy:
    video_file:         str
    framework:          str
    total_frames:       int   = 0
    frames_with_faces:  int   = 0
    total_detections:   int   = 0
    avg_conf:           float = 0.0
    min_conf:           float = 0.0
    max_conf:           float = 0.0
    std_conf:           float = 0.0
    avg_faces_per_frame:float = 0.0
    detection_rate:     float = 0.0   # frames_with_faces / total_frames
    frame_data: List[FrameDetection] = field(default_factory=list)

    def compute(self):
        if not self.frame_data:
            return
        self.total_frames      = len(self.frame_data)
        self.frames_with_faces = sum(1 for f in self.frame_data if f.num_faces > 0)
        self.total_detections  = sum(f.num_faces for f in self.frame_data)
        self.detection_rate    = self.frames_with_faces / self.total_frames if self.total_frames > 0 else 0
        self.avg_faces_per_frame = self.total_detections / self.total_frames if self.total_frames > 0 else 0

        all_confs = []
        for f in self.frame_data:
            all_confs.extend(f.confidences)

        if all_confs:
            self.avg_conf = float(np.mean(all_confs))
            self.min_conf = float(np.min(all_confs))
            self.max_conf = float(np.max(all_confs))
            self.std_conf = float(np.std(all_confs))


# ── Framework A — FFmpeg + PyAV ───────────────────────────────────────────────

def run_framework_a(video_path: Path, detector: TRTFaceDetector,
                    num_frames: int) -> VideoAccuracy:
    acc = VideoAccuracy(video_file=video_path.name, framework="Framework_A_FFmpeg")

    try:
        container    = av.open(str(video_path))
        stream       = container.streams.video[0]
        stream.thread_type = "AUTO"
    except Exception as e:
        log.error(f"[A] Cannot open {video_path.name}: {e}")
        return acc

    frame_count = 0
    for packet in container.demux(stream):
        if packet.size == 0:
            continue
        try:
            frames = list(packet.decode())
        except Exception:
            continue
        if not frames:
            continue

        img   = frames[0].to_ndarray(format="bgr24")
        dets  = detector.detect(img)
        frame_count += 1

        acc.frame_data.append(FrameDetection(
            frame_idx   = frame_count,
            framework   = "Framework_A_FFmpeg",
            video_file  = video_path.name,
            num_faces   = len(dets),
            confidences = [d[4] for d in dets],
        ))

        if frame_count >= num_frames:
            break

    container.close()
    acc.compute()
    return acc


# ── Framework B — GStreamer + TRT ─────────────────────────────────────────────

def run_framework_b(video_path: Path, detector: TRTFaceDetector,
                    cfg: Config, num_frames: int) -> VideoAccuracy:
    acc         = VideoAccuracy(video_file=video_path.name, framework="Framework_B_GStreamer")
    frame_count = 0
    done_event  = threading.Event()
    loop        = GLib.MainLoop()

    pipeline_str = (
        f"filesrc location={video_path} "
        f"! decodebin "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx,"
        f"width={cfg.decode_width},height={cfg.decode_height} "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink name=acc_sink max-buffers=1 drop=false "
        f"emit-signals=true sync=false"
    )

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        log.error(f"[B] Pipeline failed: {e}")
        return acc

    def on_sample(appsink):
        nonlocal frame_count
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf  = sample.get_buffer()
        caps = sample.get_caps()
        s    = caps.get_structure(0)
        w    = s.get_value("width")
        h    = s.get_value("height")

        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame = np.frombuffer(mi.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mi)

        dets = detector.detect(frame)
        frame_count += 1

        acc.frame_data.append(FrameDetection(
            frame_idx   = frame_count,
            framework   = "Framework_B_GStreamer",
            video_file  = video_path.name,
            num_faces   = len(dets),
            confidences = [d[4] for d in dets],
        ))

        if frame_count >= num_frames:
            done_event.set()
            GLib.idle_add(loop.quit)

        return Gst.FlowReturn.OK

    def on_bus(bus, message):
        if message.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            done_event.set()
            GLib.idle_add(loop.quit)

    sink = pipeline.get_by_name("acc_sink")
    sink.connect("new-sample", on_sample)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus)

    pipeline.set_state(Gst.State.PLAYING)
    t = threading.Thread(target=loop.run, daemon=True)
    t.start()
    done_event.wait(timeout=300)
    pipeline.set_state(Gst.State.NULL)

    acc.compute()
    return acc


# ── Frame-level agreement ─────────────────────────────────────────────────────

def compute_agreement(a_acc: VideoAccuracy, b_acc: VideoAccuracy) -> dict:
    """
    Compares frame-by-frame detection counts between A and B.
    Agreement = both detect same number of faces on same frame.
    """
    n = min(len(a_acc.frame_data), len(b_acc.frame_data))
    if n == 0:
        return {}

    exact_match = 0
    both_zero   = 0
    only_a      = 0
    only_b      = 0
    both_detect = 0
    diffs       = []

    for i in range(n):
        a_faces = a_acc.frame_data[i].num_faces
        b_faces = b_acc.frame_data[i].num_faces
        diff    = abs(a_faces - b_faces)
        diffs.append(diff)

        if a_faces == b_faces:
            exact_match += 1
            if a_faces == 0:
                both_zero += 1
        if a_faces > 0 and b_faces == 0:
            only_a += 1
        if b_faces > 0 and a_faces == 0:
            only_b += 1
        if a_faces > 0 and b_faces > 0:
            both_detect += 1

    return {
        "frames_compared":  n,
        "exact_match":      exact_match,
        "exact_match_pct":  round(exact_match / n * 100, 1),
        "both_zero":        both_zero,
        "only_a_detects":   only_a,
        "only_b_detects":   only_b,
        "both_detect":      both_detect,
        "avg_count_diff":   round(float(np.mean(diffs)), 2),
    }


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results: List[dict], prefix: str):
    lines = [
        "=" * 70,
        "  ACCURACY COMPARISON — Framework A vs Framework B",
        "  Same videos, same model, same confidence threshold",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
    ]

    for r in results:
        a = r["a"]
        b = r["b"]
        ag = r["agreement"]

        lines += [
            f"\nVideo: {a.video_file}",
            f"  {'Metric':<35} {'Framework A':>14} {'Framework B':>14}",
            f"  {'-'*65}",
            f"  {'Total frames':<35} {a.total_frames:>14} {b.total_frames:>14}",
            f"  {'Frames with faces':<35} {a.frames_with_faces:>14} {b.frames_with_faces:>14}",
            f"  {'Detection rate':<35} {a.detection_rate:>13.1%} {b.detection_rate:>13.1%}",
            f"  {'Total detections':<35} {a.total_detections:>14} {b.total_detections:>14}",
            f"  {'Avg faces per frame':<35} {a.avg_faces_per_frame:>14.2f} {b.avg_faces_per_frame:>14.2f}",
            f"  {'Avg confidence':<35} {a.avg_conf:>14.3f} {b.avg_conf:>14.3f}",
            f"  {'Min confidence':<35} {a.min_conf:>14.3f} {b.min_conf:>14.3f}",
            f"  {'Max confidence':<35} {a.max_conf:>14.3f} {b.max_conf:>14.3f}",
            f"  {'Conf std deviation':<35} {a.std_conf:>14.3f} {b.std_conf:>14.3f}",
        ]

        if ag:
            lines += [
                f"\n  Frame-level agreement:",
                f"  {'Exact face count match':<35} {ag['exact_match']:>6} frames ({ag['exact_match_pct']}%)",
                f"  {'Only A detects faces':<35} {ag['only_a_detects']:>6} frames",
                f"  {'Only B detects faces':<35} {ag['only_b_detects']:>6} frames",
                f"  {'Both detect faces':<35} {ag['both_detect']:>6} frames",
                f"  {'Avg face count difference':<35} {ag['avg_count_diff']:>6}",
            ]

    # Overall summary
    all_a = [r["a"] for r in results if r["a"].total_frames > 0]
    all_b = [r["b"] for r in results if r["b"].total_frames > 0]

    if all_a and all_b:
        lines += [
            "\n" + "=" * 70,
            "  OVERALL SUMMARY",
            "=" * 70,
            f"  {'Metric':<35} {'Framework A':>14} {'Framework B':>14}",
            f"  {'-'*65}",
            f"  {'Avg detection rate':<35} "
            f"{np.mean([a.detection_rate for a in all_a]):>13.1%} "
            f"{np.mean([b.detection_rate for b in all_b]):>13.1%}",
            f"  {'Total detections':<35} "
            f"{sum(a.total_detections for a in all_a):>14} "
            f"{sum(b.total_detections for b in all_b):>14}",
            f"  {'Avg confidence':<35} "
            f"{np.mean([a.avg_conf for a in all_a]):>14.3f} "
            f"{np.mean([b.avg_conf for b in all_b]):>14.3f}",
            "=" * 70,
        ]

    text = "\n".join(lines)
    print("\n" + text)
    with open(f"{prefix}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    log.info(f"Report: {prefix}.txt")

    # CSV
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "framework", "total_frames", "frames_with_faces",
            "detection_rate", "total_detections", "avg_faces_per_frame",
            "avg_conf", "min_conf", "max_conf", "std_conf"
        ])
        for r in results:
            for acc in [r["a"], r["b"]]:
                writer.writerow([
                    acc.video_file, acc.framework, acc.total_frames,
                    acc.frames_with_faces, round(acc.detection_rate, 4),
                    acc.total_detections, round(acc.avg_faces_per_frame, 3),
                    round(acc.avg_conf, 4), round(acc.min_conf, 4),
                    round(acc.max_conf, 4), round(acc.std_conf, 4),
                ])
    log.info(f"CSV: {prefix}.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="../../videos")
    parser.add_argument("--frames", type=int, default=200,
                        help="Frames per video per framework (default: 200)")
    args = parser.parse_args()

    cfg      = Config()
    detector = TRTFaceDetector(
        engine_path=cfg.engine_path,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        input_size=(cfg.input_w, cfg.input_h),
    )
    log.info("TRT engine loaded.")

    video_files = sorted(Path(args.videos).glob("*.dav"))
    if not video_files:
        log.error(f"No .dav files in {args.videos}")
        exit(1)

    log.info(f"Found {len(video_files)} videos — running A vs B accuracy comparison")

    all_results = []

    for video_path in video_files:
        log.info(f"\n{'='*50}")
        log.info(f"Video: {video_path.name}")

        log.info(f"  Running Framework A (FFmpeg)...")
        a_acc = run_framework_a(video_path, detector, args.frames)
        log.info(
            f"  [A] {a_acc.frames_with_faces}/{a_acc.total_frames} frames "
            f"with faces | {a_acc.total_detections} total | "
            f"avg conf: {a_acc.avg_conf:.3f}"
        )

        log.info(f"  Running Framework B (GStreamer)...")
        b_acc = run_framework_b(video_path, detector, cfg, args.frames)
        log.info(
            f"  [B] {b_acc.frames_with_faces}/{b_acc.total_frames} frames "
            f"with faces | {b_acc.total_detections} total | "
            f"avg conf: {b_acc.avg_conf:.3f}"
        )

        agreement = compute_agreement(a_acc, b_acc)
        all_results.append({"a": a_acc, "b": b_acc, "agreement": agreement})

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"../../results/accuracy_comparison_{ts}"
    save_results(all_results, prefix)
    log.info("Done.")
