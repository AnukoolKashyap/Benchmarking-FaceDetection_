# Face Detection — RTSP Live Camera Pipeline

A Python pipeline that connects to a live RTSP camera feed, detects human faces using a YOLO face model, and saves every frame containing one or more faces into a local folder. Runs at a throttled rate (default 5 FPS) and logs all detections with bounding box metadata.

Built during an internship project at Transline Technologies, New Delhi

---

## File structure

```
FACE DETECTION/
│
├── captured_faces/          # Saved frames land here (gitignored)
│   └── .gitkeep             # Keeps the folder tracked by git
│
├── rtsp_face_capture.py     # Main pipeline script
├── stream.py                # Your RTSP URL and credentials (gitignored)
├── yolov8n-face.pt          # YOLO face detection weights (gitignored)
│
├── requirements.txt
├── .gitignore
│
├── frame_loss_prevention.md          # Architecture doc — frame loss problem
└── rtsp_face_pipeline_architecture.md  # Architecture doc — pipeline overview
```

---

## How it works

The pipeline runs two concurrent pieces to avoid frame lag:

- **Frame grabber thread** — continuously reads frames off the RTSP stream and always keeps the newest one in a shared buffer, regardless of how fast inference is running.
- **Throttled inference loop** — every 200ms (5 FPS), grabs whatever the latest frame is, runs YOLO face detection on it, and saves it if at least one face is found.

This split ensures you're always evaluating the most recent frame rather than working through a stale backlog — the most common problem when reading RTSP streams naively in a single loop.

See [`rtsp_face_pipeline_architecture.md`](rtsp_face_pipeline_architecture.md) for the full architecture breakdown and [`frame_loss_prevention.md`](frame_loss_prevention.md) for a detailed discussion of the frame loss problem and the fixes.

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/hemish22/face-detection.git
cd face-detection
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download YOLO face weights**

Stock YOLO weights (trained on COCO) have no face class — you need weights fine-tuned on a face dataset. Download from either of these and place the `.pt` file in the project root:

- [`akanametov/yolo-face`](https://github.com/akanametov/yolo-face) — yolov8n-face.pt through yolov12n-face.pt
- [`lindevs/yolov8-face`](https://github.com/lindevs/yolov8-face) — trained from scratch on WIDERFACE

**4. Set your RTSP credentials**

Create a `stream.py` file in the project root (this file is gitignored and will never be committed):

```python
RTSP_URL = "rtsp://username:password@192.168.x.x:554/stream1"
```

---

## Usage

**Basic run**

```bash
python rtsp_face_capture.py --rtsp-url "rtsp://username:password@ip:554/stream1"
```

**With all options**

```bash
python rtsp_face_capture.py \
  --rtsp-url "rtsp://username:password@ip:554/stream1" \
  --model yolov8n-face.pt \
  --output captured_faces \
  --fps 5 \
  --conf 0.4 \
  --imgsz 1280 \
  --save-crops
```

**Pull RTSP URL from stream.py**

```bash
python -c "from stream import RTSP_URL; import os; os.system(f'python rtsp_face_capture.py --rtsp-url {RTSP_URL}')"
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
| `--save-crops` | off | Also save each detected face as its own cropped file |

> **Note on `--imgsz`:** The default YOLO inference resolution is 640px. For a 2560×1440 CCTV source this shrinks frames 4× before the model sees them, making small or distant faces effectively invisible. Setting `--imgsz 1280` halves the shrink factor and significantly improves recall on wide-angle shots.

---

## Output

```
captured_faces/
├── 20260618_150004_765_faces1.jpg     # Timestamp + face count in filename
├── 20260618_150512_002_faces3.jpg
├── detections.jsonl                   # One JSON line per saved frame
└── crops/                             # Only present if --save-crops is set
    ├── 20260618_150004_765_face0.jpg
    └── 20260618_150512_002_face0.jpg
```

Each line in `detections.jsonl`:

```json
{
  "timestamp": "2026-06-18T15:00:04.765",
  "filename": "20260618_150004_765_faces1.jpg",
  "face_count": 1,
  "boxes": [
    { "xyxy": [860, 940, 905, 995], "confidence": 0.49 }
  ]
}
```

---

## Known limitations

- **Small or distant faces** — wide-angle cameras covering a whole room are worst-case. Raise `--imgsz` or consider tiled inference for crowded scenes.
- **Off-axis faces** — back-of-head shots and near-full profiles fall outside what a WIDERFACE-trained model recognises. This is a model limitation, not a tuning problem.
- **No deduplication** — a person standing still gets saved once per throttle tick. Filter by position via the `detections.jsonl` log if you need unique captures.
- **Single stream per process** — run multiple instances pointing at different `--output` folders for multi-camera setups.

---

## Docs

- [`rtsp_face_pipeline_architecture.md`](rtsp_face_pipeline_architecture.md) — full pipeline architecture, component breakdown, design decisions
- [`frame_loss_prevention.md`](frame_loss_prevention.md) — the frame loss problem explained with diagrams, and all fixes compared

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLO model loading and inference |
| `opencv-python` | RTSP stream connection, frame I/O, image writing |

---

## .gitignore highlights

| Entry | Reason |
|---|---|
| `captured_faces/*` | Saved frames are output data, not source code |
| `!captured_faces/.gitkeep` | Keeps the folder itself tracked so the output path always exists |
| `stream.py` | Contains RTSP URL and credentials — never commit these |
| `yolov8n-face.pt` | Model weights are large binary files — download separately |
