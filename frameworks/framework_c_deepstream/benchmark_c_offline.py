"""
benchmark_c_offline.py — Framework C: DeepStream nvinfer (offline .dav files)
Jetson Orin — DeepStream 7.1 — CUDA 12.6

Pipeline (frame NEVER leaves GPU):
    filesrc
        → decodebin          : auto-detect codec, Jetson VPU decode
        → nvvidconv          : GPU colour convert
        → nvstreammux        : creates NvDsBatchMeta (required by nvinfer)
        → nvinfer            : TensorRT inference INSIDE GStreamer
        → nvvidconv          : convert back for appsink
        → appsink            : annotated frame comes to Python

Key difference from Framework B:
    Framework B: appsink gets raw frame → Python runs TRT → results
    Framework C: nvinfer runs TRT inside GStreamer → Python only times throughput

Outputs:
    ../../results/benchmark_c_offline_<timestamp>.json
    ../../results/benchmark_c_offline_<timestamp>.csv
    ../../results/benchmark_c_offline_<timestamp>_summary.txt
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

# ── Logging ───────────────────────────────────────────────────────────────────
Path("../../results").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("benchmark_c_offline")

Gst.init(None)

# nvinfer config file — must be in same folder as this script
NVINFER_CONFIG = "config_infer_yolov8face.txt"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    video_file:   str
    frame_idx:    int
    framework:    str
    pipeline_ms:  float
    fps:          float
    frame_w:      int
    frame_h:      int


@dataclass
class VideoStats:
    video_file:      str
    framework:       str
    total_frames:    int   = 0
    avg_fps:         float = 0.0
    min_fps:         float = 0.0
    max_fps:         float = 0.0
    avg_pipeline_ms: float = 0.0
    results: List[FrameResult] = field(default_factory=list)

    def compute(self):
        if not self.results:
            return
        self.total_frames    = len(self.results)
        self.avg_fps         = float(np.mean([r.fps         for r in self.results]))
        self.min_fps         = float(np.min( [r.fps         for r in self.results]))
        self.max_fps         = float(np.max( [r.fps         for r in self.results]))
        self.avg_pipeline_ms = float(np.mean([r.pipeline_ms for r in self.results]))


# ── Per-video processor ───────────────────────────────────────────────────────

class VideoProcessorC:
    """
    Processes one video file through DeepStream nvinfer pipeline.

    DeepStream 7.x requires nvstreammux before nvinfer.
    nvstreammux creates NvDsBatchMeta which nvinfer uses to attach detections.

    Pipeline:
        filesrc → decodebin → nvvidconv → nvstreammux → nvinfer
        → nvvidconv → videoconvert → appsink
    """

    def __init__(self, frames_per_video: int, warmup: int, decode_w: int, decode_h: int):
        self.frames_per_video = frames_per_video
        self.warmup           = warmup
        self.decode_w         = decode_w
        self.decode_h         = decode_h

    def process(self, video_path: Path) -> VideoStats:
        stats       = VideoStats(video_file=video_path.name, framework="DeepStream+nvinfer")
        frame_count = 0
        collected   = 0
        last_time   = [None]
        done_event  = threading.Event()
        loop        = GLib.MainLoop()

        # ── Pipeline string ───────────────────────────────────────────────────
        # nvstreammux is mandatory before nvinfer in DeepStream 7.x
        # it creates the NvDsBatchMeta structure nvinfer needs
        pipeline_str = (
            f"filesrc location={video_path} "
            f"! decodebin "
            f"! nvvidconv "
            f"! video/x-raw(memory:NVMM),format=RGBA,"
            f"width={self.decode_w},height={self.decode_h} "
            f"! m.sink_0 nvstreammux name=m "
            f"batch-size=1 "
            f"width={self.decode_w} "
            f"height={self.decode_h} "
            f"! nvinfer config-file-path={NVINFER_CONFIG} "
            f"! nvvidconv "
            f"! video/x-raw,format=BGRx "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink name=ds_sink max-buffers=1 drop=false "
            f"emit-signals=true sync=false"
        )

        log.info(f"  [C] Building pipeline for: {video_path.name}")

        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            log.error(f"Pipeline build failed: {e}")
            return stats

        def on_new_sample(appsink):
            nonlocal frame_count, collected

            if done_event.is_set():
                return Gst.FlowReturn.EOS

            now = time.perf_counter()
            if last_time[0] is not None:
                pipeline_ms = (now - last_time[0]) * 1000
            else:
                pipeline_ms = 0.0
            last_time[0] = now

            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR

            buf  = sample.get_buffer()
            caps = sample.get_caps()
            s    = caps.get_structure(0)
            w    = s.get_value("width")
            h    = s.get_value("height")

            ok, map_info = buf.map(Gst.MapFlags.READ)
            if ok:
                buf.unmap(map_info)

            frame_count += 1

            if frame_count <= self.warmup:
                return Gst.FlowReturn.OK

            if pipeline_ms > 0:
                fps = 1000.0 / pipeline_ms
                stats.results.append(FrameResult(
                    video_file  = video_path.name,
                    frame_idx   = collected + 1,
                    framework   = "DeepStream+nvinfer",
                    pipeline_ms = round(pipeline_ms, 3),
                    fps         = round(fps, 2),
                    frame_w     = w,
                    frame_h     = h,
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
                log.error(f"GStreamer error: {err.message}")
                log.error(f"Debug: {dbg}")
                done_event.set()
                GLib.idle_add(loop.quit)

        sink = pipeline.get_by_name("ds_sink")
        sink.connect("new-sample", on_new_sample)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_bus_message)

        pipeline.set_state(Gst.State.PLAYING)
        log.info(f"  [C] Pipeline PLAYING — warmup {self.warmup} frames...")

        t = threading.Thread(target=loop.run, daemon=True)
        t.start()
        done_event.wait(timeout=300)

        pipeline.set_state(Gst.State.NULL)
        stats.compute()

        log.info(
            f"  [C] {video_path.name} — "
            f"{collected} frames | "
            f"avg fps: {stats.avg_fps:.1f} | "
            f"avg pipeline: {stats.avg_pipeline_ms:.1f}ms"
        )
        return stats


# ── Main benchmark ────────────────────────────────────────────────────────────

class BenchmarkCOffline:

    def __init__(
        self,
        videos_dir:       str = "../../videos",
        frames_per_video: int = 100,
        warmup_frames:    int = 10,
        decode_w:         int = 1280,
        decode_h:         int = 720,
    ):
        self.videos_dir       = Path(videos_dir)
        self.frames_per_video = frames_per_video
        self.warmup_frames    = warmup_frames
        self.decode_w         = decode_w
        self.decode_h         = decode_h
        self.video_files      = sorted(self.videos_dir.glob("*.dav"))
        log.info(f"Found {len(self.video_files)} .dav files")

    def run(self) -> List[VideoStats]:
        if not self.video_files:
            log.error(f"No .dav files in {self.videos_dir}/")
            return []

        log.info("=" * 60)
        log.info("  Framework C — DeepStream nvinfer (offline)")
        log.info(f"  Videos       : {len(self.video_files)}")
        log.info(f"  Frames/video : {self.frames_per_video}")
        log.info(f"  Warmup       : {self.warmup_frames}")
        log.info(f"  nvinfer cfg  : {NVINFER_CONFIG}")
        log.info("=" * 60)

        processor = VideoProcessorC(
            self.frames_per_video,
            self.warmup_frames,
            self.decode_w,
            self.decode_h,
        )
        all_stats = []

        for i, video_path in enumerate(self.video_files, 1):
            log.info(f"\nVideo {i}/{len(self.video_files)}: {video_path.name}")
            stats = processor.process(video_path)
            all_stats.append(stats)
            time.sleep(1)

        return all_stats


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(all_stats: List[VideoStats], prefix: str):
    # JSON
    data = {
        "benchmark_date": datetime.now().isoformat(),
        "framework":      "DeepStream+nvinfer (offline .dav)",
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
        "pipeline_ms", "fps", "frame_w", "frame_h"
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
        "  Framework C — DeepStream nvinfer (offline benchmark)",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        f"{'Video':<45} {'FPS':>6} {'Pipeline ms':>12}",
        "-" * 65,
    ]
    for s in all_stats:
        if s.total_frames > 0:
            lines.append(
                f"{s.video_file[:44]:<45} {s.avg_fps:>6.1f} "
                f"{s.avg_pipeline_ms:>10.1f}ms"
            )

    valid = [s for s in all_stats if s.total_frames > 0]
    if valid:
        lines += [
            "-" * 65,
            f"{'OVERALL AVERAGE':<45} "
            f"{np.mean([s.avg_fps for s in valid]):>6.1f} "
            f"{np.mean([s.avg_pipeline_ms for s in valid]):>10.1f}ms",
            "=" * 65,
            "",
            "  NOTE: Framework C measures total pipeline throughput.",
            "  nvinfer runs TRT inside GStreamer — no Python inference.",
            "  Frame never crosses GPU to CPU during inference.",
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
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    bench = BenchmarkCOffline(
        videos_dir=args.videos,
        frames_per_video=args.frames,
        warmup_frames=args.warmup,
        decode_w=args.width,
        decode_h=args.height,
    )
    all_stats = bench.run()

    if all_stats:
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"../../results/benchmark_c_offline_{ts}"
        save_results(all_stats, prefix)
        log.info("Done.")