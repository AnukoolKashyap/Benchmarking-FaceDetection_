# RTSP face detection pipeline — Transline Technologies

A Python pipeline that connects to a live RTSP camera feed, runs YOLO face detection on it, and saves every frame containing one or more faces into a local folder. Built and benchmarked during an internship at Transline Technologies, New Delhi (OKI Program, IIIT Hyderabad).

---

## Features

- Live RTSP stream ingestion via OpenCV + FFMPEG
- YOLO face detection (WIDERFACE fine-tuned weights) at a configurable FPS target
- Connection-loss resilience — auto-reconnects, logs every gap, writes a session summary on shutdown
- Latency benchmark comparing bounded queue vs parallel worker architectures
- Automated simulation test for connection resilience (no camera needed)

---

## Repository structure

```
RTSP-FaceDetection_Transline/
│
├── rtsp_face_capture.py        main capture pipeline
├── benchmark.py                latency benchmark (bounded queue vs parallel workers)
├── requirements.txt
├── .gitignore
├── README.md
│
├── tests/
│   └── test_resilience.py      connection-loss simulation test
│
├── docs/
│   ├── architecture.md         full pipeline architecture and design decisions
│   ├── frame_loss_prevention.md  frame loss problem and fixes
│   ├── BENCHMARK.md            benchmark methodology, results, and interpretation
│   └── CONNECTION_RESILIENCE.md  connection resilience design and test walkthrough
│
├── benchmark_output/
│   ├── benchmark_report.txt    mean / p50 / p95 / p99 summary table
│   ├── benchmark_report.html   self-contained HTML report with charts
│   └── benchmark_plots.png     6-panel matplotlib comparison chart
│
└── captured_faces/
    └── .gitkeep                folder tracked by git, contents gitignored
```

---

## How it works

The pipeline runs two concurrent pieces to avoid stale-frame lag:

**Frame grabber thread** — continuously reads frames off the RTSP stream and always keeps the newest one in a shared buffer, regardless of how fast inference is running.

**Throttled inference loop** — every 200ms (5 FPS target), grabs whatever the latest frame is, runs YOLO face detection on it, and saves it to disk if at least one face is found.

This split means the camera's native frame rate (typically 25–30 fps) is decoupled from the inference rate — you're always evaluating the most recent frame rather than working through a backlog.

If the camera drops mid-run, the grabber reconnects automatically. Every disconnect and reconnect is logged to `session_gaps.jsonl` as it happens, and a `session_summary.json` is written on clean shutdown.

See [`docs/architecture.md`](docs/architecture.md) for the full breakdown and [`docs/frame_loss_prevention.md`](docs/frame_loss_prevention.md) for the frame loss problem and fixes.

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/AnukoolKashyap/RTSP-FaceDetection_Transline.git
cd RTSP-FaceDetection_Transline
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download YOLO face weights**

Stock YOLO weights (trained on COCO) have no face class. Download weights fine-tuned on a face dataset and place the `.pt` file in the project root:

- [`akanametov/yolo-face`](https://github.com/akanametov/yolo-face) — yolov8n-face.pt through yolov12n-face.pt
- [`lindevs/yolov8-face`](https://github.com/lindevs/yolov8-face) — trained from scratch on WIDERFACE

**4. Set your RTSP credentials**

Create a `stream.py` in the project root (gitignored — never committed):

```python
RTSP_URL = "rtsp://username:password@192.168.x.x:554/stream1"
```

---

## Usage

**Basic run**

```bash
python rtsp_face_capture.py --rtsp-url "rtsp://user:pass@ip:554/stream1"
```

**With all options**

```bash
python rtsp_face_capture.py \
  --rtsp-url "rtsp://user:pass@ip:554/stream1" \
  --model yolov8n-face.pt \
  --output captured_faces \
  --fps 5 \
  --conf 0.4 \
  --imgsz 1280 \
  --save-crops
```

**Pull URL from stream.py**

```bash
python -c "from stream import RTSP_URL; import subprocess; subprocess.run(['python', 'rtsp_face_capture.py', '--rtsp-url', RTSP_URL])"
```

---

## Configuration

| Flag | Default | What it does |
|---|---|---|
| `--rtsp-url` | *(required)* | RTSP stream URL |
| `--model` | `yolov8n-face.pt` | Path to YOLO face weights |
| `--output` | `captured_faces` | Folder where frames are saved |
| `--fps` | `5.0` | Target inference rate |
| `--conf` | `0.4` | Minimum detection confidence |
| `--imgsz` | `1280` | Inference resolution (long side in px) |
| `--no-boxes` | off | Save raw frame without drawn bounding boxes |
| `--save-crops` | off | Also save each detected face as a separate crop |

> `--imgsz 1280` is set as the default because the camera produces 2560×1440 frames. The Ultralytics default of 640px shrinks frames 4× before inference — small or distant faces become invisible. At 1280 the shrink is 2× and recall on wide-angle shots improves significantly.

---

## Output

```
captured_faces/
├── 20260618_150004_765_faces1.jpg     timestamp + face count in filename
├── 20260618_150512_002_faces3.jpg
├── detections.jsonl                   one JSON line per saved frame
├── session_gaps.jsonl                 one line per connect / disconnect event
├── session_summary.json               written on clean Ctrl+C shutdown
└── crops/                             only present if --save-crops is used
```

Sample `detections.jsonl` entry:
```json
{"timestamp": "2026-06-18T15:00:04.765", "filename": "20260618_150004_765_faces1.jpg", "face_count": 1, "boxes": [{"xyxy": [860, 940, 905, 995], "confidence": 0.49}]}
```

Sample `session_summary.json`:
```json
{
  "session_start": "2026-06-23T09:00:01",
  "session_end":   "2026-06-23T11:05:22",
  "duration_s":    7521.0,
  "active_stream_s": 7511.2,
  "total_gap_s":   9.8,
  "gap_count":     1,
  "frames_saved":  4821,
  "faces_detected": 5103
}
```

---

## Running the benchmark

Compares bounded queue (single consumer) vs parallel workers (two consumers) across latency, throughput, and drop rate over a 60-second window.

```bash
python benchmark.py --rtsp-url "rtsp://user:pass@ip:554/stream1" --model yolov8n-face.pt
```

Outputs to `benchmark_output/`: a text report, a self-contained HTML report with charts, a matplotlib PNG, and a raw JSON for further analysis. See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for full methodology and results.

**Benchmark results (2560×1440 CP IP Cam, CPU inference):**

| Metric | Bounded queue | Parallel workers |
|---|---|---|
| Mean total latency | 754.7 ms | 99.0 ms |
| Drop rate | 71.1% | 0.0% |
| Throughput | 4.58 fps | 13.85 fps |
| Queue wait (mean) | 680.7 ms | 42.6 ms |

---

## Running the tests

```bash
python tests/test_resilience.py
```

Simulates a full connect → drop → reconnect cycle using a mock camera (no real hardware needed) and verifies that gap events are logged correctly and the session summary is accurate. Runs in ~8 seconds.

---

## Docs

| File | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Pipeline components, threading design, output schema |
| [`docs/frame_loss_prevention.md`](docs/frame_loss_prevention.md) | Why frames get dropped and four ways to fix it |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Benchmark methodology, results, p95/p99 interpretation |
| [`docs/CONNECTION_RESILIENCE.md`](docs/CONNECTION_RESILIENCE.md) | Gap logging, session summary, simulation test design |

---

## Known limitations

- **Small or distant faces** — wide-angle cameras covering a large area are worst-case. Raise `--imgsz` or consider tiled inference for crowded scenes.
- **Off-axis faces** — back-of-head and near-full profile views fall outside what a WIDERFACE-trained model recognises. This is a model limitation, not a tuning problem.
- **No deduplication** — a person standing still gets saved once per throttle tick. Filter by bounding box position via `detections.jsonl` if unique captures are needed.
- **Single stream per process** — run multiple instances with different `--output` folders for multi-camera setups.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLO model loading and inference |
| `opencv-python` | RTSP stream connection, frame I/O, image writing |
| `matplotlib` | Benchmark plots |

---

## .gitignore highlights

| Entry | Reason |
|---|---|
| `captured_faces/*` | Output data, not source code |
| `!captured_faces/.gitkeep` | Keeps the folder tracked so the output path always exists |
| `stream.py` | Contains RTSP credentials — never commit |
| `yolov8n-face.pt` | Large binary — download separately |
| `benchmark_output/*.json` | Large raw data arrays — report and plots are committed instead |