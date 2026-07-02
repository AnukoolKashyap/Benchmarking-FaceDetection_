# Framework A — FFmpeg (PyAV) + TensorRT

Face detection pipeline using **FFmpeg** (via the PyAV Python binding) for video decode, and the **TensorRT Python API** for inference on a custom-trained YOLOv8-face model.

Part of the [Real-Time Video Analytics Framework Benchmarking](../../README.md) project — see the main README for full project context, hardware specs, and results across all three frameworks.

---

## How it works

```
.dav video file
    → PyAV opens the container (libavformat)
    → libavcodec decodes each frame
         GPU mode : hevc_cuvid  (NVDEC hardware decode)
         CPU mode : software decode, multi-threaded
    → frame.to_ndarray() — decoded frame lands in CPU RAM as a numpy array
    → OpenCV preprocess (letterbox resize to 640x640, normalize, BGR→RGB)
    → tensor copied to GPU
    → TensorRT Python API (pycuda) runs yolov8n-face.engine
    → raw output (1, 20, 8400) parsed into boxes + confidence + NMS
    → results saved (crops / logs / benchmark stats)
```

The frame crosses from GPU to CPU memory **twice** per frame: once when PyAV decodes and hands the frame to Python, and once when the preprocessed tensor is copied back to GPU for inference. This is the main architectural difference from Framework C (DeepStream), where the frame never leaves GPU memory.

---

## Files

| File | Purpose |
|---|---|
| `benchmark_offline.py` | Runs GPU vs CPU decode benchmark across all `.dav` test videos, records per-frame timing (decode/preprocess/inference/postprocess) and detection counts |
| `save_face_crops_a.py` | Runs detection on all test videos and saves each detected face as a cropped image |
| `config.py` | Engine path, input size, confidence/IoU thresholds, decode resolution |
| `utils/trt_inference.py` | TensorRT 10/11 engine loader and inference wrapper. Handles the TensorRT tensor-address API (`execute_async_v3`) and manual CUDA context management |
| `utils/frame_saver.py` | Helper for writing annotated/cropped frames to disk with rate limiting |

---

## Usage

```bash
cd frameworks/framework_a_ffmpeg
source ../../venv/bin/activate

# Speed benchmark — GPU and CPU decode, both tested automatically
python3 benchmark_offline.py --videos ../../videos --frames 100 --warmup 10

# Save cropped face detections
python3 save_face_crops_a.py --videos ../../videos --max-faces 100
```

Outputs are written to `../../results/` (benchmark CSV/JSON/summary) and `../../results2/framework_a/` (face crops, organized per video).

---

## Notes

- The TensorRT engine (`models/yolov8n-face-jetson.engine`) must be built **on the target Jetson hardware** — engines are GPU-architecture specific and cannot be copied from a desktop build.
- `trt_inference.py` uses explicit CUDA context `push()`/`pop()` around every inference call. This is required on Jetson because GStreamer-based tools (used in Framework B) create their own CUDA context; without explicit push/pop, TensorRT silently returns all-zero output instead of raising an error.
