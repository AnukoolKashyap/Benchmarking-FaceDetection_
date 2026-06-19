# RTSP face detection capture pipeline — architecture overview

## Purpose

This system continuously ingests a live RTSP camera feed, runs face detection on it at a throttled rate, and persists every frame containing one or more detected faces — along with structured metadata — into an output folder. It is designed to run unattended for long periods against a single live stream.

## High-level architecture

The pipeline is split into two concurrent halves: a **producer** that does nothing but keep pulling frames off the network as fast as the camera delivers them, and a **consumer** that samples whatever the producer most recently captured on its own fixed schedule. This split exists because RTSP streams and inference run at fundamentally different rates, and coupling them naively causes a growing backlog of stale frames (see "Why two threads" below).

```mermaid
flowchart TD
    A[RTSP camera<br/>live video source] --> B[Frame grabber thread<br/>always keeps newest frame]
    B --> C[Shared frame buffer<br/>thread-safe, single slot]
    C --> D[5 FPS throttle<br/>samples buffer every ~200ms]
    D --> E[YOLO face model<br/>runs detection inference]
    E --> F{Face(s) detected?}
    F -->|yes| G[Save frame + JSONL log<br/>+ optional per-face crops]
    F -->|no| H[Discard frame]
    G --> D
    H --> D
```

## Components

### 1. RTSP source

The camera (or NVR) exposes a stream at a URL such as `rtsp://user:pass@ip:554/stream1`. The pipeline connects via `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`, which handles the RTSP handshake and H.264/H.265 decode transparently. No assumptions are made about the camera's native resolution or frame rate — both are read as-is.

### 2. Frame grabber thread (producer)

A background daemon thread that does one thing in a tight loop: read a frame, store it, repeat. It never blocks on anything downstream. If the stream drops (read failure, camera reboot, network blip) it releases the capture object and retries the connection on a fixed delay rather than crashing the process.

**Why a dedicated thread:** `cv2.VideoCapture` keeps an internal decode buffer. If frames are read slower than the camera produces them, that buffer fills with old frames, and every `.read()` call returns video that is progressively further behind real time. Running the grabber on its own thread that always overwrites a single shared slot means the consumer is guaranteed to see the *newest* available frame whenever it asks, not the oldest one waiting in a queue.

### 3. Shared frame buffer

A single in-memory frame slot guarded by a lock. The grabber thread writes to it on every successful read; the main loop reads (and copies) from it whenever its timer fires. There is intentionally no queue — only ever one frame, the latest one. Older frames are simply overwritten and lost, which is the correct behavior for a live-monitoring use case (you care about *now*, not a backlog of *then*).

### 4. Throttle / scheduler

A simple time-based gate in the main loop: it records when inference last ran and waits until `1 / target_fps` seconds have elapsed before running again. This is deliberately time-based rather than frame-counted, since RTSP delivery rate isn't perfectly steady — a counter-based "process every Nth frame" approach would drift if the camera's actual rate fluctuates.

### 5. YOLO face detection model

Stock YOLO weights (trained on COCO) have no "face" class — the closest is "person." This pipeline requires weights specifically fine-tuned on a face dataset (WIDERFACE is the standard), loaded the same way as any other Ultralytics model. Two relevant parameters at inference time:

- **`conf`** — minimum confidence to keep a detection. Lower catches more faces but risks false positives.
- **`imgsz`** — the resolution the frame is resized to before inference. This matters more than it sounds: a 2560×1440 source frame run at the Ultralytics default of `imgsz=640` is shrunk 4x before the model ever sees it, which can shrink a distant face down to single-digit pixels and make it undetectable. Raising `imgsz` (1280, 1536...) trades inference time for the ability to catch smaller or farther faces.

### 6. Decision and multi-face handling

YOLO returns a list of bounding boxes per frame — "one face" and "five faces" are simply `len(boxes) == 1` vs `5`, with no special-casing needed in the model call itself. The pipeline branches on whether that list is empty:

- **Empty** → frame is discarded, loop continues.
- **Non-empty** → the frame is saved once with every box drawn on it, a metadata log entry is written covering all boxes, and (optionally) each box is additionally cropped into its own file.

### 7. Output / persistence layer

```
captured_faces/
├── 20260618_150004_765_faces1.jpg     ← one saved frame, name encodes timestamp + face count
├── 20260618_150512_002_faces3.jpg
├── detections.jsonl                   ← one JSON line per saved frame
└── crops/                             ← only if --save-crops is set
    ├── 20260618_150004_765_face0.jpg
    └── 20260618_150512_002_face0.jpg, _face1.jpg, _face2.jpg
```

Each line in `detections.jsonl` looks like:

```json
{"timestamp": "2026-06-18T15:00:04.765", "filename": "20260618_150004_765_faces1.jpg", "face_count": 1, "boxes": [{"xyxy": [860, 940, 905, 995], "confidence": 0.49}]}
```

This log is what makes the image folder queryable later — counting detections over time, filtering by confidence, or re-cropping faces without re-running inference all become simple JSON parsing rather than image inspection.

## Configuration parameters

| Flag | Default | Effect |
|---|---|---|
| `--rtsp-url` | *(required)* | The camera stream to connect to |
| `--model` | `yolov8n-face.pt` | Face-detection weights file |
| `--output` | `captured_faces` | Destination folder |
| `--fps` | `5.0` | Target inference rate |
| `--conf` | `0.4` | Minimum detection confidence kept |
| `--imgsz` | `1280` | Inference resolution (long side, px) |
| `--no-boxes` | off | Save the raw frame without drawn boxes |
| `--save-crops` | off | Also save each detected face as its own file |

## Known limitations

- **Small/distant faces in wide shots.** Even with a raised `imgsz`, a face that occupies only a handful of pixels in the source frame may stay below detection threshold. Wide-angle or fisheye cameras covering a whole room are the worst case for this.
- **Off-axis and occluded faces.** Profile views, back-of-head shots, or faces obscured by masks/objects fall outside what a frontal-face-trained model recognizes — this is a model limitation, not something `conf` or `imgsz` can fix.
- **No deduplication.** A person standing still in frame gets saved repeatedly, once per throttle tick, since each tick is evaluated independently.
- **Single stream, single process.** The current design handles one RTSP source per running instance.

## Possible extensions

- **Tiled inference** — split a high-resolution wide shot into overlapping tiles, run detection on each at native resolution, then merge boxes back into original coordinates. Standard approach for crowded wide-angle scenes where downscaling the full frame destroys small faces.
- **Larger model variant** — swapping the nano weights for a small/medium variant improves recall at the cost of per-frame inference time, which still fits comfortably within a 5 fps budget on most hardware.
- **Deduplication** — compare a new detection's box position against the last saved one for the same general area and skip the save if it hasn't moved meaningfully.
- **Fisheye dewarping** — a pre-processing step that corrects lens distortion before detection, which should help recall on faces away from the image center.
- **Multi-camera support** — one grabber thread + one throttled consumer loop per RTSP source, all writing to separate output subfolders.

## Tech stack

Python, OpenCV (`cv2`) for stream I/O, Ultralytics YOLO for inference, the standard library's `threading` for the producer/consumer split, and JSON Lines for the detection log.
