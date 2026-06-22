# Benchmark — RTSP Face Detection Pipeline

Latency and throughput comparison between two architectural approaches for processing a live RTSP camera feed through a YOLO face detection model.

---

## What is being measured

End-to-end latency is broken into three isolated components so the actual bottleneck is visible rather than just the total number:

| Component | Definition |
|---|---|
| **Capture latency** | Frame produced by camera → entered the buffer. Network + RTSP decode time. Roughly constant across both architectures. |
| **Queue wait time** | Frame sitting in buffer → inference started. This is where the two architectures diverge most. |
| **Inference latency** | YOLO model running on the frame. Single `predict()` call duration. |
| **Total end-to-end** | Camera frame produced → inference complete. Sum of the three above. |

Drop rate measures how many produced frames were never evaluated — evicted from the buffer before inference got to them.

---

## Architectures compared

**Architecture A — Bounded Queue (single consumer)**

A background thread feeds camera frames into a `collections.deque(maxlen=10)`. A single consumer pops frames at a 5 fps target rate and runs inference. If inference falls behind, the deque fills and the oldest frame is silently evicted.

```
Camera (~25 fps) → deque(maxlen=10) → single worker → inference
```

**Architecture B — Parallel Workers**

A feeder runs at full camera rate, putting frames into a `queue.Queue(maxsize=20)`. Two independent worker threads each hold their own YOLO model instance and drain the queue as fast as inference allows. No artificial fps throttle on the workers.

```
Camera (~25 fps) → Queue(maxsize=20) → worker 1 → inference
                                     → worker 2 → inference
```

---

## How to run

**Against your live RTSP stream:**
```bash
python benchmark.py --rtsp-url "rtsp://user:pass@ip:554/stream1" --model yolov8n-face.pt
```

**Without a camera (synthetic random frames):**
```bash
python benchmark.py --simulate --model yolov8n-face.pt
```

**Full options:**
```bash
python benchmark.py \
  --rtsp-url "rtsp://..." \
  --model yolov8n-face.pt \
  --conf 0.4 \
  --imgsz 1280 \
  --fps 5.0 \
  --duration 60 \
  --warmup 8 \
  --queue-maxlen 10 \
  --num-workers 2 \
  --queue-maxsize 20 \
  --output benchmark_output
```

### Configuration reference

| Flag | Default | Effect |
|---|---|---|
| `--rtsp-url` | — | Live RTSP stream URL |
| `--simulate` | off | Use synthetic frames instead of camera |
| `--model` | `yolov8n-face.pt` | YOLO face weights |
| `--conf` | `0.4` | Detection confidence threshold |
| `--imgsz` | `1280` | Inference resolution (long side px) |
| `--fps` | `5.0` | Arch A consumer throttle rate |
| `--duration` | `60` | Measurement window per architecture (s) |
| `--warmup` | `8` | Warmup seconds discarded before measurement |
| `--queue-maxlen` | `10` | Arch A: bounded deque capacity |
| `--num-workers` | `2` | Arch B: number of parallel inference workers |
| `--queue-maxsize` | `20` | Arch B: shared queue capacity before drop |
| `--output` | `benchmark_output` | Output folder |

> **Why a warmup period?** PyTorch JIT-compiles parts of the model on the first few inference calls, making early frames artificially slow. Discarding the first 8 seconds ensures only steady-state performance is measured.

---

## Output files

```
benchmark_output/
├── benchmark_report.txt    # Terminal-formatted summary table (mean / p50 / p95 / p99)
├── benchmark_report.html   # Self-contained HTML report with interactive charts
├── benchmark_plots.png     # 6-panel matplotlib comparison chart
└── benchmark_results.json  # Raw per-frame latency arrays for further analysis
```

The `.json` file contains full arrays of every per-frame measurement, useful for custom analysis or re-plotting without re-running the benchmark:

```json
{
  "arch_a": {
    "capture_ms": [...],
    "wait_ms": [...],
    "inference_ms": [...],
    "total_ms": [...],
    "queue_depth": [...],
    "meta": { "processed": 275, "produced": 952, "dropped": 677, "throughput": 4.58 }
  },
  "arch_b": { ... },
  "config": { "model": "yolov8n-face.pt", "imgsz": 1280, ... }
}
```

---

## Results

**Test environment:**
- Camera: CP IP Cam · 2560×1440 · ~25 fps native
- Hardware: Windows 10, CPU inference (no GPU)
- Model: `yolov8n-face.pt` (WIDERFACE fine-tuned, nano variant)
- Date: 18 Jun 2026

### Headline numbers

| Metric | Arch A — Bounded Queue | Arch B — Parallel Workers | Delta |
|---|---|---|---|
| Frames processed | 275 | 831 | +556 |
| Frames produced | 952 | 831 | — |
| **Drop rate** | **71.1%** | **0.0%** | **−71.1 pp** |
| Throughput | 4.58 fps | 13.85 fps | +9.27 fps |
| Mean total latency | 754.7 ms | 99.0 ms | −655.6 ms |

### Detailed latency breakdown

**Architecture A — Bounded Queue**

| Metric | Mean | p50 | p95 | p99 | Min | Max |
|---|---|---|---|---|---|---|
| Capture latency (ms) | 6.7 | 5.3 | 14.8 | 17.3 | 0.8 | 24.6 |
| **Queue wait time (ms)** | **680.7** | **525.6** | **1595.9** | **1985.5** | 2.8 | 2058.2 |
| Inference latency (ms) | 67.2 | 62.4 | 78.1 | 113.5 | 30.2 | 1371.1 |
| **Total end-to-end (ms)** | **754.7** | **595.5** | **1660.7** | **2056.0** | 124.5 | 2238.0 |
| Queue depth (frames) | 8.0 | 9.0 | 9.0 | 9.0 | 0.0 | 9.0 |

**Architecture B — Parallel Workers (×2)**

| Metric | Mean | p50 | p95 | p99 | Min | Max |
|---|---|---|---|---|---|---|
| Capture latency (ms) | 11.1 | 6.9 | 15.6 | 16.8 | 0.7 | 2734.6 |
| **Queue wait time (ms)** | **42.6** | **3.1** | **244.0** | **352.9** | 2.0 | 488.6 |
| Inference latency (ms) | 45.3 | 44.1 | 69.0 | 81.1 | 26.8 | 363.0 |
| **Total end-to-end (ms)** | **99.0** | **59.0** | **306.0** | **409.0** | 30.5 | 3094.0 |
| Queue depth (frames) | 1.3 | 0.0 | 8.0 | 13.0 | 0.0 | 16.0 |

---

## Interpretation

**Queue wait time is the bottleneck, not inference.** In Architecture A, the 680ms mean queue wait accounts for 90% of total latency. The deque was nearly permanently at capacity (average depth 8 out of max 10), actively evicting frames the entire run. The inference model itself (67ms) was never the problem.

**Parallel workers eliminate the backlog.** Architecture B's mean queue wait drops to 42ms with a p50 of just 3ms — the queue was almost always empty (average depth 1.3), meaning workers were keeping up with frame arrivals in near real-time.

**Inference runs faster in B despite the same model.** Mean inference time is 45ms in B vs 67ms in A. When Architecture A's single consumer was fighting backlog and OS scheduling pressure, individual `predict()` calls ran slower. Workers in B had clean execution windows with no queue pressure.

**Architecture B's throughput exceeds the 5 fps target** because workers have no artificial rate limiter — they drain as fast as inference allows. For production use at a specific rate, add a throttle on the feeder side to cap output while keeping the zero-drop property.

**The p99 tail in B (409ms)** is likely OS thread scheduling jitter on Windows. A Linux deployment or process-level workers (using `multiprocessing` instead of `threading`) would reduce this tail significantly.

---

## Why p95/p99 matter more than the mean

A camera running at 25 fps produces one frame every 40ms. If your mean latency is 99ms but p99 is 409ms, then 1 in every 100 frames — roughly once every 4 seconds — arrives at the inference result more than 400ms late. For a live monitoring system, that tail is what determines real-world responsiveness, not the average.

A system can look healthy on mean latency while silently failing on tail latency. Always check p95 and p99 before calling a pipeline production-ready.
