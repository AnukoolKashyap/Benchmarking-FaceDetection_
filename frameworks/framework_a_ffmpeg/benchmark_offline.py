"""
benchmark_offline.py — Framework A: FFmpeg + PyAV offline benchmark
Reads .dav video files using PyAV (FFmpeg) + TensorRT face detection

Same .dav files as Framework B and C for fair comparison on Jetson.

Per-frame metrics:
    decode_ms       : PyAV decode time
    preprocess_ms   : resize + normalize
    inference_ms    : TensorRT engine
    postprocess_ms  : NMS + box decode
    total_ms        : end to end
    fps             : 1000 / total_ms

Outputs:
    ../../results/benchmark_a_offline_<timestamp>.json
    ../../results/benchmark_a_offline_<timestamp>.csv
    ../../results/benchmark_a_offline_<timestamp>_summary.txt

Usage:
    python3 benchmark_offline.py
    python3 benchmark_offline.py --videos ../../videos --frames 100
"""

import av
import cv2
import time
import json
import csv
import logging
import argparse
import numpy as np
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
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("benchmark_a_offline")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    video_file:     str
    frame_idx:      int
    decode_mode:    str
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
    decode_mode:        str
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


# ── Benchmark runner ──────────────────────────────────────────────────────────

class OfflineBenchmark:
    """
    Runs face detection pipeline on offline .dav video files using PyAV.
    Tests GPU decode (hevc_cuvid) and CPU decode paths.
    """

    def __init__(
        self,
        cfg: Config,
        videos_dir:       str = "../../videos",
        frames_per_video: int = 100,
        warmup_frames:    int = 10,
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
        if not self.video_files:
            self.video_files = sorted(self.videos_dir.glob("*.mp4"))
        log.info(f"Found {len(self.video_files)} video files in {videos_dir}/")

    def _open_container(self, video_path: str, mode: str):
        """Opens video file with GPU or CPU decode."""
        base_options = {"fflags": "nobuffer"}

        if mode == "gpu":
            # Try hevc_cuvid for GPU decode
            base_options.update({
                "hwaccel":        "cuda",
                "hwaccel_device": "0",
                "vcodec":         "hevc_cuvid",
            })

        for attempt in range(1, 4):
            try:
                container = av.open(str(video_path), options=base_options)
                return container
            except Exception as e:
                log.warning(f"Open attempt {attempt}/3 failed ({mode}): {e}")
                if attempt == 1 and mode == "gpu":
                    # GPU open failed — try without vcodec hint
                    base_options.pop("vcodec", None)
                time.sleep(1)
        raise RuntimeError(f"Failed to open {video_path}")

    def _process_video(self, video_path: Path, mode: str) -> VideoStats:
        """Processes one video file and returns timing stats."""
        stats = VideoStats(video_file=video_path.name, decode_mode=mode)

        try:
            container    = self._open_container(str(video_path), mode)
            video_stream = container.streams.video[0]
            if mode == "cpu":
                video_stream.thread_type = "AUTO"
        except Exception as e:
            log.error(f"Could not open {video_path.name}: {e}")
            return stats

        frame_count = 0
        collected   = 0

        log.info(
            f"  [{mode.upper()}] {video_path.name} — "
            f"{video_stream.codec_context.width}x{video_stream.codec_context.height} "
            f"@ {float(video_stream.average_rate):.1f}fps"
        )

        try:
            for packet in container.demux(video_stream):
                if packet.size == 0:
                    continue

                # Decode timing
                t_dec = time.perf_counter()
                try:
                    frames = list(packet.decode())
                except Exception:
                    continue
                if not frames:
                    continue
                img = frames[0].to_ndarray(format="bgr24")
                decode_ms = (time.perf_counter() - t_dec) * 1000

                frame_count += 1

                # Skip warmup
                if frame_count <= self.warmup_frames:
                    continue

                # Preprocess timing
                t_pre  = time.perf_counter()
                tensor, scale, pad = self.detector._preprocess(img)
                pre_ms = (time.perf_counter() - t_pre) * 1000

                # Inference timing
                t_inf  = time.perf_counter()
                raw    = self.detector._infer(tensor)
                inf_ms = (time.perf_counter() - t_inf) * 1000

                # Postprocess timing
                t_post  = time.perf_counter()
                dets    = self.detector._postprocess(
                    raw, img.shape[:2], scale, pad
                )
                post_ms = (time.perf_counter() - t_post) * 1000

                total_ms = decode_ms + pre_ms + inf_ms + post_ms
                fps      = 1000.0 / total_ms if total_ms > 0 else 0.0

                stats.results.append(FrameResult(
                    video_file     = video_path.name,
                    frame_idx      = collected + 1,
                    decode_mode    = mode,
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
                    break

        except Exception as e:
            log.warning(f"Error during decode: {e}")
        finally:
            container.close()

        stats.compute()
        log.info(
            f"  [{mode.upper()}] Done — {collected} frames | "
            f"avg fps: {stats.avg_fps:.1f} | "
            f"avg inference: {stats.avg_inference_ms:.1f}ms | "
            f"faces: {stats.total_faces}"
        )
        return stats

    def run(self) -> List[VideoStats]:
        if not self.video_files:
            log.error(f"No .dav files found in {self.videos_dir}/")
            return []

        all_stats = []

        log.info("=" * 60)
        log.info("  Framework A — FFmpeg + PyAV offline benchmark")
        log.info(f"  Videos        : {len(self.video_files)}")
        log.info(f"  Frames/video  : {self.frames_per_video}")
        log.info(f"  Warmup frames : {self.warmup_frames}")
        log.info(f"  Engine        : {self.cfg.engine_path}")
        log.info("=" * 60)

        for i, video_path in enumerate(self.video_files, 1):
            log.info(f"\nVideo {i}/{len(self.video_files)}: {video_path.name}")

            # GPU path
            gpu_stats = self._process_video(video_path, "gpu")
            all_stats.append(gpu_stats)
            time.sleep(1)

            # CPU path
            cpu_stats = self._process_video(video_path, "cpu")
            all_stats.append(cpu_stats)
            time.sleep(1)

        return all_stats


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(all_stats: List[VideoStats], prefix: str):
    # JSON
    data = {
        "benchmark_date": datetime.now().isoformat(),
        "framework":      "FFmpeg+PyAV (offline .dav)",
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
        "video_file", "frame_idx", "decode_mode",
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
    gpu_stats = [s for s in all_stats if s.decode_mode == "gpu" and s.total_frames > 0]
    cpu_stats = [s for s in all_stats if s.decode_mode == "cpu" and s.total_frames > 0]

    lines = [
        "=" * 65,
        "  Framework A — FFmpeg + PyAV (offline benchmark)",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        f"{'Video':<40} {'Mode':<5} {'FPS':>6} {'Decode':>8} {'Infer':>7}",
        "-" * 65,
    ]

    videos = {}
    for s in all_stats:
        if s.video_file not in videos:
            videos[s.video_file] = {}
        videos[s.video_file][s.decode_mode] = s

    for vname, modes in videos.items():
        for mode, s in modes.items():
            if s.total_frames > 0:
                lines.append(
                    f"{vname[:39]:<40} {mode:<5} {s.avg_fps:>6.1f} "
                    f"{s.avg_decode_ms:>7.1f}ms {s.avg_inference_ms:>6.1f}ms"
                )

    if gpu_stats and cpu_stats:
        lines += [
            "-" * 65,
            f"{'GPU AVERAGE':<40} {'gpu':<5} "
            f"{np.mean([s.avg_fps for s in gpu_stats]):>6.1f} "
            f"{np.mean([s.avg_decode_ms for s in gpu_stats]):>7.1f}ms "
            f"{np.mean([s.avg_inference_ms for s in gpu_stats]):>6.1f}ms",
            f"{'CPU AVERAGE':<40} {'cpu':<5} "
            f"{np.mean([s.avg_fps for s in cpu_stats]):>6.1f} "
            f"{np.mean([s.avg_decode_ms for s in cpu_stats]):>7.1f}ms "
            f"{np.mean([s.avg_inference_ms for s in cpu_stats]):>6.1f}ms",
            "=" * 65,
        ]

    text = "\n".join(lines)
    print("\n" + text)
    with open(f"{prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    log.info(f"Summary: {prefix}_summary.txt")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Framework A offline benchmark — FFmpeg+PyAV"
    )
    parser.add_argument("--videos", default="../../videos")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    cfg   = Config()
    bench = OfflineBenchmark(
        cfg,
        videos_dir=args.videos,
        frames_per_video=args.frames,
        warmup_frames=args.warmup,
    )
    all_stats = bench.run()

    if all_stats:
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"../../results/benchmark_a_offline_{ts}"
        save_results(all_stats, prefix)
        log.info(f"All results saved to results/benchmark_a_offline_{ts}.*")
