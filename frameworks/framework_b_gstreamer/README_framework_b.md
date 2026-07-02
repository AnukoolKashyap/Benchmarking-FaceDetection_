# Framework B — GStreamer + TensorRT

Face detection pipeline using **GStreamer** (via PyGObject/`gi`) for video decode on the Jetson's dedicated VPU hardware, and the **TensorRT Python API** for inference on the same YOLOv8-face model used in Framework A.

Part of the [Real-Time Video Analytics Framework Benchmarking](../../README.md) project — see the main README for full project context, hardware specs, and results across all three frameworks.

---

## How it works

```
.dav video file / RTSP stream
    → filesrc / rtspsrc                — GStreamer source element
    → decodebin / rtph265depay+h265parse
    → nvv4l2decoder                    — Jetson VPU hardware decode → NV12 in GPU memory
    → nvvidconv                        — GPU colour convert + resize
    → videoconvert                     — final format conversion
    → appsink                          — frame handed to Python as numpy array
    → OpenCV preprocess (letterbox resize, normalize)
    → tensor copied to GPU
    → TensorRT Python API (pycuda) runs yolov8n-face.engine
    → raw output parsed into boxes + confidence + NMS
    → results saved (crops / logs / benchmark stats)
```

Like Framework A, the frame crosses GPU↔CPU twice per frame — the difference is that decode happens on the Jetson's dedicated VPU chip via `nvv4l2decoder`, which measured faster and more consistent decode times than Framework A's software/NVDEC path.

---

## Files

| File | Purpose |
|---|---|
| `benchmark_b_offline.py` | Runs the GStreamer decode + TensorRT benchmark across all `.dav` test videos |
| `accuracy_comparison.py` | Runs Framework A and Framework B side-by-side on identical frames and reports detection-rate, confidence, and frame-level agreement between them |
| `save_face_crops_b.py` | Runs detection on all test videos and saves each detected face as a cropped image |
| `gst_pipeline.py` | Live RTSP pipeline for camera deployment (as opposed to offline `.dav` file benchmarking) |
| `test_gst.py` | Quick connectivity test — confirms GStreamer can open and decode a source before running the full benchmark |
| `config.py` | RTSP URL / video path, decode resolution, engine path, thresholds |
| `utils/trt_inference.py` | Same TensorRT wrapper as Framework A — shared CUDA-context-safe implementation |
| `utils/frame_saver.py` | Helper for writing annotated/cropped frames to disk |

---

## Usage

```bash
cd frameworks/framework_b_gstreamer
source ../../venv/bin/activate

# Test the pipeline connects and decodes correctly first
python3 test_gst.py --url "rtsp://..."          # for live camera
# or just run the offline benchmark directly for .dav files

# Speed benchmark
python3 benchmark_b_offline.py --videos ../../videos --frames 100 --warmup 10

# Accuracy comparison against Framework A
python3 accuracy_comparison.py --videos ../../videos --frames 200

# Save cropped face detections
python3 save_face_crops_b.py --videos ../../videos --max-faces 100
```

Outputs are written to `../../results/` (benchmark and accuracy CSV/JSON/summary) and `../../results2/framework_b/` (face crops, organized per video).

---

## Important — CUDA context conflict on Jetson

GStreamer's `nvv4l2decoder` and `nvvidconv` plugins create their own CUDA context when the pipeline starts. If TensorRT/pycuda initializes a separate context (e.g. via `pycuda.autoinit`), inference will run without error but **silently return all-zero output** — no crash, no exception, just empty detections.

`utils/trt_inference.py` fixes this by:
1. Creating a single explicit CUDA context (`cuda_device.make_context()`), falling back to `cuda.Context.attach()` if another process already holds the GPU
2. Calling `cuda_context.push()` before every TensorRT inference call and `cuda_context.pop()` after, so GStreamer and TensorRT correctly hand off GPU ownership on every frame

If you see raw TensorRT output with `max confidence: 0.0000` on Jetson, this context conflict is the first thing to check.

---

## Notes

- `decode_width` / `decode_height` in `config.py` must match your camera's native resolution (2560×1440 here). Decoding at a lower resolution before running detection significantly reduces face detection accuracy, since faces become too small after the resize-to-640×640 step.
- The same TensorRT engine and confidence threshold (0.45) are used as Framework A, enabling direct accuracy comparison — see `accuracy_comparison.py` output for frame-level agreement results.
