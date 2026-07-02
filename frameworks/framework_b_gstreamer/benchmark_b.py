"""
benchmark_b.py — Framework B benchmark: GStreamer + Python TensorRT (Jetson)

Measures per-frame timing for the GStreamer pipeline:
    decode_ms       : nvv4l2decoder + nvvidconv + appsink time
    preprocess_ms   : resize + normalize on CPU
    inference_ms    : TensorRT engine
    postprocess_ms  : NMS + box decode
    total_ms        : end to end
    fps             : 1000 / total_ms

Outputs:
    ../../results/benchmark_b_<timestamp>.json
    ../../results/benchmark_b_<timestamp>.csv
    ../../results/benchmark_b_<timestamp>_summary.txt
"""

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GLib

import sys
import cv2
import numpy as np
import time
import json
import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List

from config import Config
from utils.trt_inference import TRTFaceDetector

# ── Logging ──────────────────────────────────────────────────────────────────
Path("../../results").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("benchmark_b")

Gst.init(None)

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrameResult:
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
class BenchmarkStats:
    framework:         str
    total_frames:      int   = 0
    avg_decode_ms:     float = 0.0
    avg_preprocess_ms: float = 0.0
    avg_inference_ms:  float = 0.0
    avg_postprocess_ms:float = 0.0
    avg_total_ms:      float = 0.0
    avg_fps:           float = 0.0
    min_fps:           float = 0.0
    max_fps:           float = 0.0
    total_faces:       int   = 0
    results: List[FrameResult] = field(default_factory=list)

    def compute(self):
        if not self.results:
            return
        self.total_frames       = len(self.results)
        self.avg_decode_ms      = float(np.mean([r.decode_ms      for r in self.results]))
        self.avg_preprocess_ms  = float(np.mean([r.preprocess_ms  for r in self.results]))
        self.avg_inference_ms   = float(np.mean([r.inference_ms   for r in self.results]))
        self.avg_postprocess_ms = float(np.mean([r.postprocess_ms for r in self.results]))
        self.avg_total_ms       = float(np.mean([r.total_ms       for r in self.results]))
        self.avg_fps            = float(np.mean([r.fps            for r in self.results]))
        self.min_fps            = float(np.min( [r.fps            for r in self.results]))
        self.max_fps            = float(np.max( [r.fps            for r in self.results]))
        self.total_faces        = sum(r.faces_detected for r in self.results)


# ── Benchmarker ───────────────────────────────────────────────────────────────

class BenchmarkB:
    """
    Runs the GStreamer + TRT pipeline for N frames and records timing.
    Uses a threading.Event to coordinate between GStreamer callback and main thread.
    """

    def __init__(self, cfg: Config, num_frames: int = 200, warmup: int = 30):
        self.cfg        = cfg
        self.num_frames = num_frames
        self.warmup     = warmup
        self.stats      = BenchmarkStats(framework="GStreamer+TRT")
        self.pipeline   = None
        self.loop       = None
        self._frame_count = 0
        self._collected   = 0
        self._done_event  = threading.Event()

        log.info(f"Loading TRT engine: {cfg.engine_path}")
        self.detector = TRTFaceDetector(
            engine_path=cfg.engine_path,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            input_size=(cfg.input_w, cfg.input_h),
        )
        log.info("TRT engine ready.")

    def _build_pipeline(self) -> str:
        return (
            f"rtspsrc location={self.cfg.rtsp_url} latency=200 protocols=tcp "
            f"! rtph265depay ! h265parse "
            f"! nvv4l2decoder "
            f"! nvvidconv "
            f"! video/x-raw,format=BGRx,"
            f"width={self.cfg.decode_width},height={self.cfg.decode_height} "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink name=bench_sink max-buffers=1 drop=true "
            f"emit-signals=true sync=false"
        )

    def _on_new_sample(self, appsink):
        """GStreamer callback — called for every decoded frame."""
        if self._done_event.is_set():
            return Gst.FlowReturn.EOS

        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf  = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        w = structure.get_value("width")
        h = structure.get_value("height")

        # ── Decode timing (includes appsink transfer = GPU→CPU) ──────────────
        t_dec_start = time.perf_counter()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(map_info)
        t_dec_end = time.perf_counter()
        decode_ms = (t_dec_end - t_dec_start) * 1000

        self._frame_count += 1

        # Skip warmup frames
        if self._frame_count <= self.warmup:
            if self._frame_count == self.warmup:
                log.info(f"Warmup done. Collecting {self.num_frames} frames...")
            return Gst.FlowReturn.OK

        # ── Preprocess timing ─────────────────────────────────────────────────
        t_pre = time.perf_counter()
        tensor, scale, pad = self.detector._preprocess(frame)
        pre_ms = (time.perf_counter() - t_pre) * 1000

        # ── Inference timing ──────────────────────────────────────────────────
        t_inf = time.perf_counter()
        raw   = self.detector._infer(tensor)
        inf_ms = (time.perf_counter() - t_inf) * 1000

        # ── Postprocess timing ────────────────────────────────────────────────
        t_post = time.perf_counter()
        dets   = self.detector._postprocess(raw, frame.shape[:2], scale, pad)
        post_ms = (time.perf_counter() - t_post) * 1000

        total_ms = decode_ms + pre_ms + inf_ms + post_ms
        fps      = 1000.0 / total_ms if total_ms > 0 else 0.0

        self.stats.results.append(FrameResult(
            frame_idx      = self._collected + 1,
            framework      = "GStreamer+TRT",
            decode_ms      = round(decode_ms,  3),
            preprocess_ms  = round(pre_ms,     3),
            inference_ms   = round(inf_ms,     3),
            postprocess_ms = round(post_ms,    3),
            total_ms       = round(total_ms,   3),
            fps            = round(fps,        2),
            faces_detected = len(dets),
        ))
        self._collected += 1

        if self._collected % 50 == 0:
            recent = self.stats.results[-50:]
            avg_fps = np.mean([r.fps for r in recent])
            log.info(f"[B] {self._collected}/{self.num_frames} frames | avg fps: {avg_fps:.1f}")

        if self._collected >= self.num_frames:
            self._done_event.set()
            GLib.idle_add(self.loop.quit)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            log.error(f"GStreamer error: {err.message} — {dbg}")
            self._done_event.set()
            GLib.idle_add(self.loop.quit)

    def run(self) -> BenchmarkStats:
        log.info("=" * 55)
        log.info("  Framework B — GStreamer + Python TRT benchmark")
        log.info(f"  Frames : {self.num_frames}  Warmup: {self.warmup}")
        log.info("=" * 55)

        pipeline_str = self._build_pipeline()
        self.pipeline = Gst.parse_launch(pipeline_str)

        sink = self.pipeline.get_by_name("bench_sink")
        sink.connect("new-sample", self._on_new_sample)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)
        log.info(f"Pipeline PLAYING — warming up {self.warmup} frames...")

        self.loop = GLib.MainLoop()

        # Run GLib loop in a thread so we can monitor from main thread
        t = threading.Thread(target=self.loop.run, daemon=True)
        t.start()

        # Wait for benchmark to complete
        self._done_event.wait(timeout=300)   # 5 min max

        self.pipeline.set_state(Gst.State.NULL)
        self.stats.compute()

        log.info(
            f"[B] Done — {self._collected} frames | "
            f"avg fps: {self.stats.avg_fps:.1f} | "
            f"avg inference: {self.stats.avg_inference_ms:.1f}ms"
        )
        return self.stats


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(stats: BenchmarkStats, prefix: str):
    # JSON
    data = {
        "framework": stats.framework,
        "summary":   {k: v for k, v in asdict(stats).items() if k != "results"},
        "frames":    [asdict(r) for r in stats.results],
    }
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info(f"JSON: {prefix}.json")

    # CSV
    fields = ["frame_idx","framework","decode_ms","preprocess_ms",
              "inference_ms","postprocess_ms","total_ms","fps","faces_detected"]
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in stats.results:
            writer.writerow(asdict(r))
    log.info(f"CSV: {prefix}.csv")

    # Summary
    lines = [
        "=" * 55,
        "  Framework B — GStreamer + Python TRT",
        f"  Frames : {stats.total_frames}",
        "=" * 55,
        f"  Avg FPS            : {stats.avg_fps:.2f}",
        f"  Min FPS            : {stats.min_fps:.2f}",
        f"  Max FPS            : {stats.max_fps:.2f}",
        f"  Avg total ms       : {stats.avg_total_ms:.2f}",
        f"  Avg decode ms      : {stats.avg_decode_ms:.2f}",
        f"  Avg preprocess ms  : {stats.avg_preprocess_ms:.2f}",
        f"  Avg inference ms   : {stats.avg_inference_ms:.2f}",
        f"  Avg postprocess ms : {stats.avg_postprocess_ms:.2f}",
        f"  Total faces found  : {stats.total_faces}",
        "=" * 55,
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
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    cfg   = Config()
    bench = BenchmarkB(cfg, num_frames=args.frames, warmup=args.warmup)
    stats = bench.run()

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"../../results/benchmark_b_{ts}"
    save_results(stats, prefix)
