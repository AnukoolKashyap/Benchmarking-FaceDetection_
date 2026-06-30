"""
benchmark_b_offline.py — Framework B offline benchmark
GStreamer + Python TensorRT on Jetson using .dav video files

Pipeline:
    filesrc → decodebin → nvvidconv → videoconvert → appsink → TensorRT

Same .dav files as Windows benchmark_offline.py for fair comparison.

Outputs:
    ../../results/benchmark_b_offline_<timestamp>.json
    ../../results/benchmark_b_offline_<timestamp>.csv
    ../../results/benchmark_b_offline_<timestamp>_summary.txt

Usage:
    python3 benchmark_b_offline.py
    python3 benchmark_b_offline.py --videos ../../videos --frames 100
"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import sys
import numpy as np
import time
import json
import csv
import logging
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List

from config import Config
from utils.trt_inference import TRTFaceDetector

# ── Logging ───────────────────────────────────────────────────────────────────
Path("../../results").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("benchmark_b_offline")

Gst.init(None)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    video_file:     str
    frame_idx:      int
    framework:      str
    decode_ms:      float
    preprocess_ms:  float
    inference_ms:   float
    postprocess_ms: float
    total_ms:       float
    fps:            float
    faces_detected: int


@dataclass
class VideoStats:
    video_file:         str
    framework:          str
    total_frames:       int   = 0
    avg_fps:            float = 0.0
    min_fps:            float = 0.0
    max_fps:            float = 0.0
    avg_decode_ms:      float = 0.0
    avg_preprocess_ms:  float = 0.0
    avg_inference_ms:   float = 0.0
    avg_postprocess_ms: float = 0.0
    avg_total_ms:       float = 0.0
    total_faces:        int   = 0
    results: List[FrameResult] = field(default_factory=list)

    def compute(self):
        if not self.results:
            return
        self.total_frames        = len(self.results)
        self.avg_fps             = float(np.mean([r.fps            for r in self.results]))
        self.min_fps             = float(np.min( [r.fps            for r in self.results]))
        self.max_fps             = float(np.max( [r.fps            for r in self.results]))
        self.avg_decode_ms       = float(np.mean([r.decode_ms      for r in self.results]))
        self.avg_preprocess_ms   = float(np.mean([r.preprocess_ms  for r in self.results]))
        self.avg_inference_ms    = float(np.mean([r.inference_ms   for r in self.results]))
        self.avg_postprocess_ms  = float(np.mean([r.postprocess_ms for r in self.results]))
        self.avg_total_ms        = float(np.mean([r.total_ms       for r in self.results]))
        self.total_faces         = sum(r.faces_detected for r in self.results)


# ── Per-video benchmarker ─────────────────────────────────────────────────────

class VideoProcessor:
    """
    Processes one video file through GStreamer + TRT pipeline.
    Uses threading.Event to signal completion from GStreamer callback.
    """

    def __init__(self, detector: TRTFaceDetector, frames_per_video: int, warmup: int):
        self.detector         = detector
        self.frames_per_video = frames_per_video
        self.warmup           = warmup

    def process(self, video_path: Path, cfg: Config) -> VideoStats:
        stats         = VideoStats(video_file=video_path.name, framework="GStreamer+TRT")
        frame_count   = 0
        collected     = 0
        done_event    = threading.Event()
        loop          = GLib.MainLoop()

        # Pipeline string — filesrc instead of rtspsrc
        # decodebin auto-detects codec (works for both H264 and H265 .dav files)
        pipeline_str = (
            f"filesrc location={video_path} "
            f"! decodebin "
            f"! nvvidconv "
            f"! video/x-raw,format=BGRx,"
            f"width={cfg.decode_width},height={cfg.decode_height} "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink name=bench_sink max-buffers=1 drop=false "
            f"emit-signals=true sync=false"
        )

        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            log.error(f"Pipeline build failed: {e}")
            return stats

        def on_new_sample(appsink):
            nonlocal frame_count, collected

            if done_event.is_set():
                return Gst.FlowReturn.EOS

            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR

            buf  = sample.get_buffer()
            caps = sample.get_caps()
            s    = caps.get_structure(0)
            w    = s.get_value("width")
            h    = s.get_value("height")

            # Decode timing (includes appsink GPU→CPU transfer)
            t_dec = time.perf_counter()
            ok, map_info = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.ERROR
            try:
                frame = np.frombuffer(
                    map_info.data, dtype=np.uint8
                ).reshape((h, w, 3)).copy()
            finally:
                buf.unmap(map_info)
            decode_ms = (time.perf_counter() - t_dec) * 1000

            frame_count += 1

            # Skip warmup
            if frame_count <= self.warmup:
                return Gst.FlowReturn.OK

            # Preprocess
            t_pre  = time.perf_counter()
            tensor, scale, pad = self.detector._preprocess(frame)
            pre_ms = (time.perf_counter() - t_pre) * 1000

            # Inference
            t_inf  = time.perf_counter()
            raw    = self.detector._infer(tensor)
            inf_ms = (time.perf_counter() - t_inf) * 1000

            # Postprocess
            t_post  = time.perf_counter()
            dets    = self.detector._postprocess(
                raw, frame.shape[:2], scale, pad
            )
            post_ms = (time.perf_counter() - t_post) * 1000

            total_ms = decode_ms + pre_ms + inf_ms + post_ms
            fps      = 1000.0 / total_ms if total_ms > 0 else 0.0

            stats.results.append(FrameResult(
                video_file     = video_path.name,
                frame_idx      = collected + 1,
                framework      = "GStreamer+TRT",
                decode_ms      = round(decode_ms,  3),
                preprocess_ms  = round(pre_ms,     3),
                inference_ms   = round(inf_ms,     3),
                postprocess_ms = round(post_ms,    3),
                total_ms       = round(total_ms,   3),
                fps            = round(fps,        2),
                faces_detected = len(dets),
            ))
            collected += 1

            if collected >= self.frames_per_video:
                done_event.set()
                GLib.idle_add(loop.quit)

            return Gst.FlowReturn.OK

        def on_bus_message(bus, message):
            if message.type == Gst.MessageType.EOS:
                log.info(f"  EOS — {collected} frames collected")
                done_event.set()
                GLib.idle_add(loop.quit)
            elif message.type == Gst.MessageType.ERROR:
                err, dbg = message.parse_error()
                log.error(f"GStreamer error: {err.message} — {dbg}")
                done_event.set()
                GLib.idle_add(loop.quit)

        sink = pipeline.get_by_name("bench_sink")
        sink.connect("new-sample", on_new_sample)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_bus_message)

        pipeline.set_state(Gst.State.PLAYING)

        # Run loop in thread
        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        done_event.wait(timeout=300)

        pipeline.set_state(Gst.State.NULL)
        stats.compute()

        log.info(
            f"  [B] {video_path.name} — "
            f"{collected} frames | "
            f"avg fps: {stats.avg_fps:.1f} | "
            f"avg inference: {stats.avg_inference_ms:.1f}ms | "
            f"faces: {stats.total_faces}"
        )
        return stats


# ── Main benchmark ────────────────────────────────────────────────────────────

class BenchmarkBOffline:

    def __init__(
        self,
        cfg: Config,
        videos_dir: str = "../../videos",
        frames_per_video: int = 100,
        warmup_frames: int = 10,
    ):
        self.cfg              = cfg
        self.videos_dir       = Path(videos_dir)
        self.frames_per_video = frames_per_video
        self.warmup_frames    = warmup_frames

        log.info(f"Loading TRT engine: {cfg.engine_path}")
        self.detector = TRTFaceDetector(
            engine_path=cfg.engine_path,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            input_size=(cfg.input_w, cfg.input_h),
        )
        log.info("TRT engine ready.")

        self.video_files = sorted(self.videos_dir.glob("*.dav"))
        log.info(f"Found {len(self.video_files)} .dav files in {videos_dir}/")

    def run(self) -> List[VideoStats]:
        if not self.video_files:
            log.error(f"No .dav files found in {self.videos_dir}/")
            return []

        log.info("=" * 60)
        log.info("  Framework B — GStreamer + Python TRT (offline)")
        log.info(f"  Videos       : {len(self.video_files)}")
        log.info(f"  Frames/video : {self.frames_per_video}")
        log.info(f"  Warmup       : {self.warmup_frames}")
        log.info("=" * 60)

        processor = VideoProcessor(
            self.detector,
            self.frames_per_video,
            self.warmup_frames,
        )
        all_stats = []

        for i, video_path in enumerate(self.video_files, 1):
            log.info(f"\nVideo {i}/{len(self.video_files)}: {video_path.name}")
            stats = processor.process(video_path, self.cfg)
            all_stats.append(stats)
            time.sleep(1)  # small gap between videos

        return all_stats


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(all_stats: List[VideoStats], prefix: str):
    # JSON
    data = {
        "benchmark_date": datetime.now().isoformat(),
        "framework":      "GStreamer+TRT (offline .dav)",
        "videos": [
            {
                "summary": {k: v for k, v in asdict(s).items() if k != "results"},
                "frames":  [asdict(r) for r in s.results],
            }
            for s in all_stats
        ],
    }
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info(f"JSON: {prefix}.json")

    # CSV
    fields = [
        "video_file", "frame_idx", "framework",
        "decode_ms", "preprocess_ms", "inference_ms",
        "postprocess_ms", "total_ms", "fps", "faces_detected"
    ]
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in all_stats:
            for r in s.results:
                writer.writerow(asdict(r))
    log.info(f"CSV: {prefix}.csv")

    # Summary
    lines = [
        "=" * 65,
        "  Framework B — GStreamer + Python TRT (offline benchmark)",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        f"{'Video':<45} {'FPS':>6} {'Decode':>8} {'Infer':>7} {'Faces':>6}",
        "-" * 65,
    ]
    for s in all_stats:
        if s.total_frames > 0:
            lines.append(
                f"{s.video_file[:44]:<45} {s.avg_fps:>6.1f} "
                f"{s.avg_decode_ms:>7.1f}ms {s.avg_inference_ms:>6.1f}ms "
                f"{s.total_faces:>6}"
            )

    valid = [s for s in all_stats if s.total_frames > 0]
    if valid:
        lines += [
            "-" * 65,
            f"{'OVERALL AVERAGE':<45} "
            f"{np.mean([s.avg_fps for s in valid]):>6.1f} "
            f"{np.mean([s.avg_decode_ms for s in valid]):>7.1f}ms "
            f"{np.mean([s.avg_inference_ms for s in valid]):>6.1f}ms",
            "=" * 65,
        ]

    text = "\n".join(lines)
    print("\n" + text)
    with open(f"{prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    log.info(f"Summary: {prefix}_summary.txt")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="../../videos")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    cfg   = Config()
    bench = BenchmarkBOffline(
        cfg,
        videos_dir=args.videos,
        frames_per_video=args.frames,
        warmup_frames=args.warmup,
    )
    all_stats = bench.run()

    if all_stats:
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"../../results/benchmark_b_offline_{ts}"
        save_results(all_stats, prefix)
        log.info("Done.")
