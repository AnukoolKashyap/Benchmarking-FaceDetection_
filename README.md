# Real-Time Video Analytics Framework Benchmarking

Benchmarking and comparative analysis of three real-time video analytics pipelines — **FFmpeg+PyAV**, **GStreamer**, and **NVIDIA DeepStream** — for face detection using a custom-trained YOLOv8 model. Evaluates FPS, decode/inference latency, and detection accuracy across identical test videos on an NVIDIA Jetson Orin.

---

## Project Overview

The goal of this project is a fair, apples-to-apples benchmark of three different approaches to building a real-time video analytics pipeline:

- **Framework A** — FFmpeg (via PyAV) for decode + TensorRT Python API for inference
- **Framework B** — GStreamer for decode + TensorRT Python API for inference
- **Framework C** — NVIDIA DeepStream, with TensorRT inference running natively inside the GStreamer pipeline via `nvinfer`

All three frameworks run the **same YOLOv8-face model**, on the **same hardware**, against the **same set of test video recordings**, to isolate the effect of pipeline architecture on performance and accuracy.

---

## Hardware & Environment

| Component | Spec |
|---|---|
| Platform | NVIDIA Jetson Orin |
| OS | Ubuntu 22.04.5 LTS |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| DeepStream SDK | 7.1.0 |
| GStreamer | 1.20.3 |
| Python | 3.10.12 |
| Model | YOLOv8n-face (FP32, TensorRT engine ~13.5MB) |
| Test videos | 6 recorded HEVC clips, 2560×1440, from a fixed IP camera |

---

## Repository Structure

```
frameworks/
├── framework_a_ffmpeg/          FFmpeg (PyAV) + TensorRT Python API
│   ├── benchmark_offline.py     GPU vs CPU decode benchmark on .dav files
│   ├── save_face_crops_a.py     saves cropped face detections to disk
│   ├── config.py
│   └── utils/
│       ├── trt_inference.py     TensorRT 10/11 engine wrapper (Jetson CUDA-context safe)
│       └── frame_saver.py
│
├── framework_b_gstreamer/       GStreamer + TensorRT Python API
│   ├── benchmark_b_offline.py   GStreamer decode benchmark on .dav files
│   ├── accuracy_comparison.py   frame-level accuracy comparison vs Framework A
│   ├── save_face_crops_b.py     saves cropped face detections to disk
│   ├── gst_pipeline.py          live RTSP pipeline (for camera deployment)
│   ├── test_gst.py              quick pipeline connectivity test
│   ├── config.py
│   └── utils/
│       ├── trt_inference.py     same TensorRT wrapper, shared CUDA-context fix
│       └── frame_saver.py
│
├── framework_c_deepstream/      DeepStream + nvinfer (TensorRT inside GStreamer)
│   └── benchmark_c_offline.py   pipeline throughput benchmark (nvstreammux + nvinfer)
│
├── models/                      TensorRT engine files (Jetson-specific, not portable)
├── results/                     benchmark CSV / JSON / summary outputs
└── results2/                    cropped face detection images per framework
```

---

## How Each Pipeline Works

### Framework A — FFmpeg + PyAV + TensorRT

```
.dav file → PyAV (libavformat/libavcodec) → decoded frame in CPU RAM
   → OpenCV preprocess (resize, normalize) → TensorRT Python API (pycuda)
   → bounding boxes → save
```
Frame crosses GPU↔CPU twice: once when PyAV hands the decoded frame to Python, and once when the tensor is copied back to GPU for inference.

### Framework B — GStreamer + TensorRT

```
.dav file → GStreamer (nvv4l2decoder, nvvidconv) → appsink hands frame to Python
   → OpenCV preprocess → TensorRT Python API (pycuda)
   → bounding boxes → save
```
Same two GPU↔CPU crossings as Framework A, but decode happens on the Jetson's dedicated VPU chip via `nvv4l2decoder` instead of software/NVDEC decode.

### Framework C — DeepStream + nvinfer

```
.dav file → GStreamer (nvv4l2decoder, nvvidconv) → nvstreammux (batches frames)
   → nvinfer (TensorRT runs INSIDE GStreamer) → metadata only crosses to Python
```
The frame itself never leaves GPU memory — only the final detection metadata (a handful of numbers) crosses to the CPU. This eliminates the memory-crossing bottleneck present in Frameworks A and B.

---

## Key Results

### Speed (avg over 6 test videos, 100 frames each)

| Framework | Mode | Avg FPS | Avg Decode | Avg Inference |
|---|---|---|---|---|
| A — FFmpeg+PyAV | GPU decode | 16.3 | 20.1 ms | 8.7 ms |
| A — FFmpeg+PyAV | CPU decode | 20.8 | 3.7 ms | 8.7 ms |
| B — GStreamer | VPU decode | 21.9 | 2.5 ms | 8.7 ms |
| C — DeepStream nvinfer | VPU decode | **122.3** | — | inside GStreamer |

### Accuracy (Framework A vs Framework B, same model, same threshold)

| Metric | Framework A | Framework B |
|---|---|---|
| Avg detection rate | 58.6% | 59.5% |
| Total detections (6 videos) | 898 | 910 |
| Avg confidence | 0.579 | 0.577 |
| Frame-level agreement | up to 99% on single-face videos |

Frameworks A and B produce statistically equivalent detection results — confirming that pipeline architecture affects **speed**, not **accuracy**, since both use the identical TensorRT engine and model.

---

## Key Findings

- **GPU decode is not always faster than CPU decode** on Jetson's unified memory architecture — Framework A's software/CPU decode path (20.8 FPS) outperformed its own GPU decode path (16.3 FPS) due to unified memory transfer overhead.
- **DeepStream's zero-copy architecture is dramatically faster** — Framework C achieves 4–6× the throughput of Frameworks A and B because the video frame never crosses from GPU to CPU memory; only detection metadata does.
- **TensorRT engines are GPU-architecture specific** — an engine built on a Windows desktop GPU cannot run on Jetson Orin; it must be rebuilt from ONNX on the target hardware.
- **CUDA context conflicts** occur when GStreamer's hardware decode plugins (`nvv4l2decoder`, `nvvidconv`) and a separately-initialized `pycuda` context both try to use the GPU. Fixed via explicit CUDA context `push()`/`pop()` around every TensorRT inference call.
- **DeepStream's built-in `nvinfer` bbox parser cannot read YOLOv8's raw output** (`1, 20, 8400` tensor) since it expects legacy `coverage`/`bboxes` output layers. Full detection support in Framework C requires a custom C++ bounding-box parser (see [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)) — documented as a next step.

---

## Setup

### Prerequisites
- NVIDIA Jetson Orin (or compatible), Ubuntu 22.04
- CUDA 12.6, TensorRT 10.3, DeepStream SDK 7.1
- Python 3.10, GStreamer 1.20 with Python bindings (`gi`)

### Install dependencies (inside a venv)
```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install pycuda opencv-python numpy av
```

### Build the TensorRT engine (must be done on target hardware)
```bash
python3 -c "
import tensorrt as trt
logger  = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network()
parser  = trt.OnnxParser(network, logger)
with open('models/yolov8n-face.onnx', 'rb') as f:
    parser.parse(f.read())
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
engine = builder.build_serialized_network(network, config)
with open('models/yolov8n-face-jetson.engine', 'wb') as f:
    f.write(engine)
"
```

### Run a benchmark
```bash
# Framework A
cd frameworks/framework_a_ffmpeg
python3 benchmark_offline.py --videos ../../videos --frames 100 --warmup 10

# Framework B
cd frameworks/framework_b_gstreamer
python3 benchmark_b_offline.py --videos ../../videos --frames 100 --warmup 10

# Framework C
cd frameworks/framework_c_deepstream
python3 benchmark_c_offline.py --videos ../../videos --frames 100 --warmup 10
```

---

## Known Limitations / Next Steps

- Framework C currently reports pipeline throughput only; full bounding-box output requires integrating a custom YOLOv8 parser for `nvinfer` (DeepStream-Yolo approach — ONNX re-export + compiled C++ parser plugin).
- No ground-truth-labelled dataset is used; accuracy comparison is framework-to-framework agreement rather than precision/recall against annotated data.
- Benchmarks use offline recorded video (`.dav`) rather than live RTSP for reproducibility; a live RTSP pipeline (`gst_pipeline.py`) is included separately for deployment reference.

---

## License

This project's documentation intentionally omits any organization-identifying information. Model weights, camera credentials, and internal file paths are excluded via `.gitignore`.