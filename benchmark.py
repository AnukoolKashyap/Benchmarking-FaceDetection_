"""
benchmark.py — RTSP Face Detection Pipeline Latency Benchmark

Compares two architectures head-to-head:
  A) Bounded Queue   — single inference consumer, deque(maxlen=N) auto-evicts oldest on full
  B) Parallel Workers — N inference threads drain a shared Queue concurrently

Metrics collected per frame, per architecture:
  capture_ms   — time from camera frame produced → entered queue   (network + decode)
  wait_ms      — time frame spent sitting in queue before inference starts (queue pressure)
  inference_ms — time YOLO model took to run on the frame
  total_ms     — end-to-end: camera → inference complete
  queue_depth  — how deep the buffer was at moment of consumption
  drop_rate    — % of produced frames never evaluated

Output files (written to --output folder):
  benchmark_plots.png   — 6-panel comparison chart
  benchmark_report.txt  — mean / p50 / p95 / p99 summary table
  benchmark_results.json — raw arrays for external analysis

Usage:
    # Live RTSP camera
    python benchmark.py --rtsp-url "rtsp://user:pass@ip:554/stream1" --model yolov8n-face.pt

    # No camera — synthetic random frames (useful for testing the benchmark itself)
    python benchmark.py --simulate --model yolov8n-face.pt

    # Pull RTSP URL from stream.py
    python -c "from stream import RTSP_URL; import os; os.system(f'python benchmark.py --rtsp-url {RTSP_URL}')"
"""

import argparse
import collections
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — plots will be skipped. pip install matplotlib")

from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def pstats(arr: np.ndarray) -> dict:
    """Percentile statistics for a metric array."""
    if len(arr) == 0:
        return dict(count=0, mean=0, p50=0, p95=0, p99=0, min=0, max=0)
    return dict(
        count=int(len(arr)),
        mean=float(np.mean(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frame sources
# ─────────────────────────────────────────────────────────────────────────────

class RTSPGrabber:
    """
    Background thread that continuously reads an RTSP stream and exposes the
    most recent frame with a monotonic frame_id and high-res timestamp.
    """
    def __init__(self, rtsp_url: str, reconnect_delay: float = 3.0):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self._cap = None
        self._lock = threading.Lock()
        self._latest = None          # (frame_id, frame, t_grabbed)
        self._frame_id = 0
        self.total_produced = 0
        self.running = False
        self._thread = None

    def _connect(self) -> bool:
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self._cap.isOpened()

    def _run(self):
        while self.running:
            if self._cap is None or not self._cap.isOpened():
                print("[grabber] connecting to stream...")
                if not self._connect():
                    time.sleep(self.reconnect_delay)
                    continue
            ok, frame = self._cap.read()
            if not ok:
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue
            t = time.perf_counter()
            with self._lock:
                self._frame_id += 1
                self._latest = (self._frame_id, frame, t)
                self.total_produced += 1

    def get_latest(self):
        with self._lock:
            return self._latest

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()


class SyntheticGrabber:
    """
    Generates random 1280x720 BGR frames at a fixed rate — lets you run the
    benchmark without a live RTSP feed. Inference will find no faces but all
    latency timings are real.
    """
    def __init__(self, fps: float = 25.0):
        self._interval = 1.0 / fps
        self._lock = threading.Lock()
        self._latest = None
        self._frame_id = 0
        self.total_produced = 0
        self.running = False
        self._thread = None

    def _run(self):
        while self.running:
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            t = time.perf_counter()
            with self._lock:
                self._frame_id += 1
                self._latest = (self._frame_id, frame, t)
                self.total_produced += 1
            time.sleep(self._interval)

    def get_latest(self):
        with self._lock:
            return self._latest

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe metrics store
# ─────────────────────────────────────────────────────────────────────────────

class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.records = []

    def add(self, r: dict):
        with self._lock:
            self.records.append(r)

    def arrays(self) -> dict:
        rs = self.records
        if not rs:
            return {}
        keys = ["capture_ms", "wait_ms", "inference_ms", "total_ms", "queue_depth", "t"]
        return {k: np.array([r[k] for r in rs]) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Architecture A — Bounded Queue, single consumer
# ─────────────────────────────────────────────────────────────────────────────

def bench_bounded_queue(
    grabber, model, conf, imgsz,
    duration, fps_target, warmup, maxlen
):
    """
    A feeder thread drains the grabber at full camera rate into a bounded deque.
    A single consumer pops from it at fps_target pace and runs inference.
    When the deque is full, the oldest unprocessed frame is evicted silently.
    """
    metrics = MetricsStore()
    buf = collections.deque(maxlen=maxlen)
    buf_lock = threading.Lock()
    feeder_stop = threading.Event()
    last_fed_id = [None]
    frames_fed = [0]

    def feeder():
        """Runs as fast as the camera produces frames, pushing each into the deque."""
        while not feeder_stop.is_set():
            item = grabber.get_latest()
            if item is None:
                time.sleep(0.001)
                continue
            fid, frame, t_grab = item
            if fid == last_fed_id[0]:
                time.sleep(0.001)
                continue
            last_fed_id[0] = fid
            t_queued = time.perf_counter()
            with buf_lock:
                buf.append((fid, frame.copy(), t_grab, t_queued))
            frames_fed[0] += 1

    print(f"\n{'='*60}")
    print(f"  Architecture A — Bounded Queue (maxlen={maxlen}, single consumer)")
    print(f"{'='*60}")
    print(f"  Warming up {warmup}s ...")
    time.sleep(warmup)

    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    run_start = time.perf_counter()
    print(f"  Benchmarking for {duration}s at {fps_target} fps target ...")

    frame_interval = 1.0 / fps_target
    next_tick = run_start

    while time.perf_counter() - run_start < duration:
        now = time.perf_counter()
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
            continue
        next_tick = now + frame_interval

        with buf_lock:
            if not buf:
                continue
            fid, frame, t_grab, t_queued = buf.popleft()
            depth = len(buf)

        t_inf_start = time.perf_counter()
        model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        t_inf_end = time.perf_counter()

        metrics.add({
            "capture_ms":   (t_queued - t_grab) * 1000,
            "wait_ms":      (t_inf_start - t_queued) * 1000,
            "inference_ms": (t_inf_end - t_inf_start) * 1000,
            "total_ms":     (t_inf_end - t_grab) * 1000,
            "queue_depth":  depth,
            "t":            t_inf_end - run_start,
        })

    feeder_stop.set()
    feeder_thread.join(timeout=2)

    n_processed = len(metrics.records)
    n_produced  = frames_fed[0]
    n_dropped   = max(0, n_produced - n_processed)
    drop_pct    = (n_dropped / n_produced * 100) if n_produced else 0
    throughput  = n_processed / duration

    print(f"  Done — processed={n_processed}  produced={n_produced}  "
          f"dropped={n_dropped} ({drop_pct:.1f}%)  throughput={throughput:.2f} fps")

    return metrics, dict(
        processed=n_processed, produced=n_produced,
        dropped=n_dropped, drop_pct=drop_pct, throughput=throughput
    )


# ─────────────────────────────────────────────────────────────────────────────
# Architecture B — Parallel Workers
# ─────────────────────────────────────────────────────────────────────────────

def bench_parallel_workers(
    grabber, model, conf, imgsz,
    duration, fps_target, warmup, num_workers, queue_maxsize
):
    """
    Main loop feeds a shared Queue at camera rate.
    N independent worker threads each hold their own YOLO instance — deep-copied
    from the already-loaded model so no re-reading the weights file per thread.
    When the queue is full, put_nowait() drops the incoming frame rather than
    blocking the feeder.
    """
    metrics = MetricsStore()
    task_q = queue.Queue(maxsize=queue_maxsize)
    stop_evt = threading.Event()
    counter_lock = threading.Lock()
    processed = [0]
    run_start_ref = [None]

    print(f"\n{'='*60}")
    print(f"  Architecture B — Parallel Workers ({num_workers} workers, queue maxsize={queue_maxsize})")
    print(f"{'='*60}")

    # Load all worker models HERE in the main thread, before warmup or threads start.
    # torch.load + pickle can fail inside worker threads on Windows because Python's
    # module import system isn't fully thread-safe during unpickling of custom classes.
    # Loading in the main thread (same context where Arch A already succeeded) is safe.
    ckpt_path = str(model.ckpt_path)
    print(f"  Pre-loading {num_workers} worker models in main thread ...")
    worker_models = []
    for i in range(num_workers):
        worker_models.append(YOLO(ckpt_path))
        print(f"    [{i+1}/{num_workers}] ready")

    def worker(local_model):
        """Worker receives an already-loaded model object — no file I/O in the thread."""
        while not stop_evt.is_set():
            try:
                fid, frame, t_grab, t_queued, depth = task_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t_inf_start = time.perf_counter()
            local_model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
            t_inf_end = time.perf_counter()

            run_start = run_start_ref[0]
            if run_start is not None:
                elapsed = t_inf_end - run_start
                if elapsed >= 0:
                    metrics.add({
                        "capture_ms":   (t_queued - t_grab) * 1000,
                        "wait_ms":      (t_inf_start - t_queued) * 1000,
                        "inference_ms": (t_inf_end - t_inf_start) * 1000,
                        "total_ms":     (t_inf_end - t_grab) * 1000,
                        "queue_depth":  depth,
                        "t":            elapsed,
                    })
                    with counter_lock:
                        processed[0] += 1
            task_q.task_done()

    print(f"  Warming up {warmup}s ...")
    time.sleep(warmup)

    workers = [threading.Thread(target=worker, args=(worker_models[i],), daemon=True)
               for i in range(num_workers)]
    for w in workers:
        w.start()

    run_start_ref[0] = time.perf_counter()
    run_start = run_start_ref[0]
    print(f"  Benchmarking for {duration}s ...")

    last_fed_id = None
    frames_fed = 0
    frames_dropped_full = 0

    while time.perf_counter() - run_start < duration:
        item = grabber.get_latest()
        if item is None:
            time.sleep(0.001)
            continue
        fid, frame, t_grab = item
        if fid == last_fed_id:
            time.sleep(0.001)
            continue
        last_fed_id = fid

        t_queued = time.perf_counter()
        depth = task_q.qsize()
        try:
            task_q.put_nowait((fid, frame.copy(), t_grab, t_queued, depth))
            frames_fed += 1
        except queue.Full:
            frames_dropped_full += 1

    stop_evt.set()
    for w in workers:
        w.join(timeout=5)

    n_processed = processed[0]
    n_produced  = frames_fed
    n_dropped   = frames_dropped_full
    drop_pct    = (n_dropped / (n_produced + n_dropped) * 100) if (n_produced + n_dropped) else 0
    throughput  = n_processed / duration

    print(f"  Done — processed={n_processed}  produced={n_produced}  "
          f"dropped={n_dropped} ({drop_pct:.1f}%)  throughput={throughput:.2f} fps")

    return metrics, dict(
        processed=n_processed, produced=n_produced,
        dropped=n_dropped, drop_pct=drop_pct, throughput=throughput
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def write_report(path: Path, results_a: dict, results_b: dict, meta_a: dict, meta_b: dict, args):
    arr_a = results_a
    arr_b = results_b

    lines = []
    lines.append("=" * 70)
    lines.append("  RTSP FACE DETECTION PIPELINE — BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append(f"  Model          : {args.model}")
    lines.append(f"  Resolution     : imgsz={args.imgsz}")
    lines.append(f"  Confidence     : {args.conf}")
    lines.append(f"  Duration       : {args.duration}s per architecture")
    lines.append(f"  Warmup         : {args.warmup}s (discarded)")
    lines.append(f"  FPS target     : {args.fps}")
    lines.append(f"  Mode           : {'Synthetic (no camera)' if args.simulate else args.rtsp_url}")
    lines.append("")

    for label, arr, meta in [("A — Bounded Queue", arr_a, meta_a), ("B — Parallel Workers", arr_b, meta_b)]:
        lines.append(f"  ── Architecture {label} ──")
        lines.append(f"  Frames processed : {meta['processed']}")
        lines.append(f"  Frames produced  : {meta['produced']}")
        lines.append(f"  Frames dropped   : {meta['dropped']}  ({meta['drop_pct']:.1f}%)")
        lines.append(f"  Throughput       : {meta['throughput']:.2f} fps")
        lines.append("")
        lines.append(f"  {'Metric':<25} {'Mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'Min':>8} {'Max':>8}")
        lines.append(f"  {'-'*73}")

        metric_labels = [
            ("capture_ms",   "Capture latency (ms)"),
            ("wait_ms",      "Queue wait time (ms)"),
            ("inference_ms", "Inference latency (ms)"),
            ("total_ms",     "Total end-to-end (ms)"),
            ("queue_depth",  "Queue depth (frames)"),
        ]
        for key, label_m in metric_labels:
            if key not in arr:
                continue
            s = pstats(arr[key])
            lines.append(
                f"  {label_m:<25} {s['mean']:>8.1f} {s['p50']:>8.1f} "
                f"{s['p95']:>8.1f} {s['p99']:>8.1f} {s['min']:>8.1f} {s['max']:>8.1f}"
            )
        lines.append("")

    # Delta summary
    if arr_a and arr_b:
        lines.append("  ── Head-to-head delta (B minus A, negative = B is better) ──")
        for key, label_m in [
            ("total_ms",     "Total latency mean"),
            ("wait_ms",      "Queue wait mean"),
            ("inference_ms", "Inference mean"),
        ]:
            if key in arr_a and key in arr_b:
                delta = np.mean(arr_b[key]) - np.mean(arr_a[key])
                sign = "+" if delta >= 0 else ""
                lines.append(f"  {label_m:<30} {sign}{delta:.1f} ms")
        delta_drop = meta_b["drop_pct"] - meta_a["drop_pct"]
        sign = "+" if delta_drop >= 0 else ""
        lines.append(f"  {'Drop rate':<30} {sign}{delta_drop:.1f} pp")
        lines.append("")

    lines.append("=" * 70)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print("\n" + text)
    return text


def write_plots(path: Path, arr_a: dict, arr_b: dict, meta_a: dict, meta_b: dict):
    if not HAS_MPL:
        print("[plots] matplotlib not available — skipping.")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("RTSP Face Detection Pipeline — Benchmark Results", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    COLORS = {"A": "#3a86ff", "B": "#ff006e"}
    ALPHA = 0.85

    # ── Panel 1: End-to-end latency over time ──
    ax1 = fig.add_subplot(gs[0, 0])
    if "t" in arr_a and "total_ms" in arr_a:
        ax1.plot(arr_a["t"], arr_a["total_ms"], color=COLORS["A"], alpha=ALPHA,
                 linewidth=0.8, label="A — Bounded Queue")
    if "t" in arr_b and "total_ms" in arr_b:
        ax1.plot(arr_b["t"], arr_b["total_ms"], color=COLORS["B"], alpha=ALPHA,
                 linewidth=0.8, label="B — Parallel Workers")
    ax1.set_title("End-to-End Latency Over Time")
    ax1.set_xlabel("Elapsed time (s)")
    ax1.set_ylabel("Total latency (ms)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Queue depth over time ──
    ax2 = fig.add_subplot(gs[0, 1])
    if "t" in arr_a and "queue_depth" in arr_a:
        ax2.plot(arr_a["t"], arr_a["queue_depth"], color=COLORS["A"], alpha=ALPHA,
                 linewidth=0.8, label="A — Bounded Queue")
    if "t" in arr_b and "queue_depth" in arr_b:
        ax2.plot(arr_b["t"], arr_b["queue_depth"], color=COLORS["B"], alpha=ALPHA,
                 linewidth=0.8, label="B — Parallel Workers")
    ax2.set_title("Queue Depth Over Time")
    ax2.set_xlabel("Elapsed time (s)")
    ax2.set_ylabel("Frames in queue")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: CDF of total latency ──
    ax3 = fig.add_subplot(gs[0, 2])
    for label, arr, col in [("A — Bounded Queue", arr_a, COLORS["A"]),
                              ("B — Parallel Workers", arr_b, COLORS["B"])]:
        if "total_ms" in arr and len(arr["total_ms"]) > 0:
            sorted_lat = np.sort(arr["total_ms"])
            cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
            ax3.plot(sorted_lat, cdf, color=col, alpha=ALPHA, linewidth=1.5, label=label)
            ax3.axvline(np.percentile(sorted_lat, 95), color=col, linestyle="--",
                        linewidth=0.8, alpha=0.7, label=f"p95 ({label[0]})")
    ax3.set_title("CDF — Total Latency")
    ax3.set_xlabel("Latency (ms)")
    ax3.set_ylabel("Cumulative probability")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Latency component breakdown (grouped bars) ──
    ax4 = fig.add_subplot(gs[1, 0])
    metrics_order = ["capture_ms", "wait_ms", "inference_ms"]
    metric_names  = ["Capture", "Queue Wait", "Inference"]
    x = np.arange(len(metrics_order))
    width = 0.3
    for i, (label, arr, col) in enumerate([("A", arr_a, COLORS["A"]), ("B", arr_b, COLORS["B"])]):
        means = [float(np.mean(arr[k])) if k in arr and len(arr[k]) > 0 else 0
                 for k in metrics_order]
        p95s  = [float(np.percentile(arr[k], 95)) if k in arr and len(arr[k]) > 0 else 0
                 for k in metrics_order]
        bars = ax4.bar(x + (i - 0.5) * width, means, width,
                       label=f"{'Bounded Q' if i == 0 else 'Parallel W'} (mean)",
                       color=col, alpha=ALPHA)
        ax4.errorbar(x + (i - 0.5) * width, means,
                     yerr=[np.zeros(len(means)), np.array(p95s) - np.array(means)],
                     fmt="none", color="black", capsize=4, linewidth=1, label=f"p95 ({'A' if i==0 else 'B'})")
    ax4.set_title("Latency Breakdown per Stage\n(bars=mean, whisker=p95)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names)
    ax4.set_ylabel("ms")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, axis="y")

    # ── Panel 5: p50 / p95 / p99 comparison ──
    ax5 = fig.add_subplot(gs[1, 1])
    pcts = [50, 95, 99]
    for label, arr, col in [("A — Bounded Q", arr_a, COLORS["A"]),
                              ("B — Parallel W", arr_b, COLORS["B"])]:
        if "total_ms" in arr and len(arr["total_ms"]) > 0:
            vals = [float(np.percentile(arr["total_ms"], p)) for p in pcts]
            ax5.plot(pcts, vals, "o-", color=col, label=label, linewidth=2, markersize=7)
    ax5.set_title("Total Latency Percentiles")
    ax5.set_xlabel("Percentile")
    ax5.set_ylabel("Latency (ms)")
    ax5.set_xticks(pcts)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Drop rate + throughput comparison ──
    ax6 = fig.add_subplot(gs[1, 2])
    arch_labels = ["Bounded Queue", "Parallel Workers"]
    drop_rates  = [meta_a["drop_pct"], meta_b["drop_pct"]]
    throughputs = [meta_a["throughput"], meta_b["throughput"]]
    x6 = np.arange(len(arch_labels))
    ax6b = ax6.twinx()
    b1 = ax6.bar(x6 - 0.2, drop_rates, 0.35, color=[COLORS["A"], COLORS["B"]],
                 alpha=ALPHA, label="Drop rate (%)")
    b2 = ax6b.bar(x6 + 0.2, throughputs, 0.35, color=[COLORS["A"], COLORS["B"]],
                  alpha=0.45, label="Throughput (fps)", hatch="//")
    ax6.set_title("Drop Rate & Throughput")
    ax6.set_ylabel("Drop rate (%)", color="black")
    ax6b.set_ylabel("Throughput (fps)", color="gray")
    ax6.set_xticks(x6)
    ax6.set_xticklabels(arch_labels, fontsize=9)
    ax6.grid(True, alpha=0.3, axis="y")
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark bounded-queue vs parallel-worker RTSP inference pipeline"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--rtsp-url", help="RTSP stream URL")
    src.add_argument("--simulate", action="store_true",
                     help="Use synthetic random frames instead of a live camera")

    parser.add_argument("--model",      default="yolov8n-face.pt")
    parser.add_argument("--conf",       type=float, default=0.4)
    parser.add_argument("--imgsz",      type=int,   default=1280)
    parser.add_argument("--fps",        type=float, default=5.0,
                        help="Target FPS for Arch A consumer throttle")
    parser.add_argument("--duration",   type=int,   default=60,
                        help="Benchmark window per architecture (seconds)")
    parser.add_argument("--warmup",     type=int,   default=8,
                        help="Warmup seconds before each run (discarded)")
    parser.add_argument("--queue-maxlen",  type=int, default=10,
                        help="Arch A: bounded deque max length")
    parser.add_argument("--num-workers",   type=int, default=2,
                        help="Arch B: number of parallel inference workers")
    parser.add_argument("--queue-maxsize", type=int, default=20,
                        help="Arch B: shared queue max size before drop")
    parser.add_argument("--output",     default="benchmark_output",
                        help="Folder to write report, plots, and JSON")
    args = parser.parse_args()

    if not args.simulate and not args.rtsp_url:
        parser.error("Provide --rtsp-url or --simulate")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # ── Build frame source ──
    if args.simulate:
        print("[source] using synthetic grabber (25 fps random frames)")
        grabber = SyntheticGrabber(fps=25.0)
    else:
        print(f"[source] RTSP: {args.rtsp_url}")
        grabber = RTSPGrabber(args.rtsp_url)

    grabber.start()
    time.sleep(2)  # let the grabber settle before loading the model

    print(f"[model] loading {args.model} ...")
    model = YOLO(args.model)

    # ── Run Architecture A ──
    metrics_a, meta_a = bench_bounded_queue(
        grabber, model, args.conf, args.imgsz,
        args.duration, args.fps, args.warmup, args.queue_maxlen
    )

    # ── Run Architecture B ──
    metrics_b, meta_b = bench_parallel_workers(
        grabber, model, args.conf, args.imgsz,
        args.duration, args.fps, args.warmup, args.num_workers, args.queue_maxsize
    )

    grabber.stop()

    arr_a = metrics_a.arrays()
    arr_b = metrics_b.arrays()

    # ── Report ──
    write_report(out / "benchmark_report.txt", arr_a, arr_b, meta_a, meta_b, args)

    # ── Plots ──
    write_plots(out / "benchmark_plots.png", arr_a, arr_b, meta_a, meta_b)

    # ── Raw JSON ──
    raw = {
        "arch_a": {k: v.tolist() for k, v in arr_a.items()} | {"meta": meta_a},
        "arch_b": {k: v.tolist() for k, v in arr_b.items()} | {"meta": meta_b},
        "config": vars(args),
    }
    json_path = out / "benchmark_results.json"
    json_path.write_text(json.dumps(raw, indent=2))
    print(f"[json]  saved → {json_path}")
    print(f"\n[done]  all outputs in ./{args.output}/")


if __name__ == "__main__":
    main()
