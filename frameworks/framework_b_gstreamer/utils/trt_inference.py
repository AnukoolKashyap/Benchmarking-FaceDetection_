"""
utils/trt_inference.py — TensorRT inference for YOLOv8-face (Jetson)

Key fix for Jetson: GStreamer GPU plugins and pycuda conflict over CUDA context.
Fix: manually manage CUDA context with push/pop around every TRT inference call.

TensorRT 10/11 API:
    - get_tensor_name()     replaces get_binding_index()
    - get_tensor_mode()     replaces binding_is_input()
    - get_tensor_shape()    replaces get_binding_shape()
    - get_tensor_dtype()    replaces get_binding_dtype()
    - execute_async_v3()    replaces execute_async_v2()
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

log = logging.getLogger("trt_inference")

# ── CUDA context management ───────────────────────────────────────────────────
# On Jetson, GStreamer's nvv4l2decoder creates its own CUDA context.
# We must manually manage our pycuda context to avoid conflicts.
# autoinit is NOT used — we create the context explicitly.

try:
    import tensorrt as trt
    import pycuda.driver as cuda

    cuda.init()
    cuda_device = cuda.Device(0)
    try:
        cuda_context = cuda_device.make_context()
    except cuda.Error:
        # Another process holds context — attach to existing one
        cuda_context = cuda.Context.attach()

    TRT_AVAILABLE = True
    log.info(f"TensorRT {trt.__version__} loaded with manual CUDA context.")

except ImportError:
    TRT_AVAILABLE = False
    cuda_context  = None
    log.warning("TensorRT not found — falling back to Ultralytics CPU inference.")


import atexit

def _cleanup_cuda():
    try:
        if cuda_context:
            cuda_context.pop()
            cuda_context.detach()
    except Exception:
        pass

atexit.register(_cleanup_cuda)


class TRTFaceDetector:
    """
    Loads a TensorRT .engine file and runs YOLOv8-face inference.
    Compatible with TensorRT 10+ and 11+.
    Falls back to Ultralytics if TensorRT is not installed.

    On Jetson: uses CUDA context push/pop to avoid conflict with GStreamer.
    """

    def __init__(
        self,
        engine_path: str,
        conf_threshold: float = 0.45,
        iou_threshold:  float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
    ):
        self.engine_path    = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.input_w, self.input_h = input_size

        if TRT_AVAILABLE:
            self._load_trt_engine()
        else:
            self._load_ultralytics_fallback()

    # ── TensorRT path ─────────────────────────────────────────────────────────

    def _load_trt_engine(self):
        """Deserializes .engine file and allocates GPU memory buffers."""
        log.info(f"Loading TRT engine: {self.engine_path}")

        cuda_context.push()

        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime    = trt.Runtime(trt_logger)

            with open(self.engine_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            self.input_names   = []
            self.output_names  = []
            self.host_inputs   = {}
            self.host_outputs  = {}
            self.device_inputs  = {}
            self.device_outputs = {}
            self.output_shapes  = {}

            num_tensors = self.engine.num_io_tensors

            for i in range(num_tensors):
                name  = self.engine.get_tensor_name(i)
                mode  = self.engine.get_tensor_mode(name)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size  = 1
                for d in shape:
                    size *= abs(d)

                host_mem   = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                    self.host_inputs[name]   = host_mem
                    self.device_inputs[name] = device_mem
                    log.info(f"  Input  '{name}': shape={shape} dtype={dtype}")
                else:
                    self.output_names.append(name)
                    self.host_outputs[name]   = host_mem
                    self.device_outputs[name] = device_mem
                    self.output_shapes[name]  = shape
                    log.info(f"  Output '{name}': shape={shape} dtype={dtype}")

            self.stream = cuda.Stream()
            log.info("TRT engine ready. GPU buffers allocated.")

        finally:
            cuda_context.pop()

        self._infer = self._infer_trt

    def _infer_trt(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """One TRT inference pass with CUDA context push/pop."""
        cuda_context.push()

        try:
            input_name = self.input_names[0]
            np.copyto(self.host_inputs[input_name], input_tensor.ravel())

            cuda.memcpy_htod_async(
                self.device_inputs[input_name],
                self.host_inputs[input_name],
                self.stream
            )

            for name in self.input_names:
                self.context.set_tensor_address(
                    name, int(self.device_inputs[name])
                )
            for name in self.output_names:
                self.context.set_tensor_address(
                    name, int(self.device_outputs[name])
                )

            self.context.execute_async_v3(stream_handle=self.stream.handle)

            for name in self.output_names:
                cuda.memcpy_dtoh_async(
                    self.host_outputs[name],
                    self.device_outputs[name],
                    self.stream
                )

            self.stream.synchronize()

        finally:
            cuda_context.pop()

        return [
            self.host_outputs[name].reshape(self.output_shapes[name])
            for name in self.output_names
        ]

    # ── Ultralytics fallback ──────────────────────────────────────────────────

    def _load_ultralytics_fallback(self):
        from ultralytics import YOLO
        pt_path = self.engine_path.replace(".engine", ".pt")
        log.info(f"Fallback: loading {pt_path} via Ultralytics")
        self._yolo  = YOLO(pt_path)
        self._infer = self._infer_ultralytics

    def _infer_ultralytics(self, _tensor):
        return self._yolo(
            self._raw_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

    # ── Preprocess ────────────────────────────────────────────────────────────

    def _preprocess(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """BGR frame → (1, 3, 640, 640) float32 tensor."""
        h, w  = frame.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        pad_x = (self.input_w  - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        rgb        = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw        = np.transpose(normalized, (2, 0, 1))
        tensor     = np.ascontiguousarray(np.expand_dims(chw, 0))

        return tensor, scale, (pad_x, pad_y)

    # ── Postprocess ───────────────────────────────────────────────────────────

    def _postprocess(
        self,
        raw_outputs,
        orig_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int, float]]:
        """Raw TRT tensors → list of (x1, y1, x2, y2, conf)."""

        if not TRT_AVAILABLE:
            dets = []
            for r in raw_outputs:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    dets.append((x1, y1, x2, y2, float(b.conf[0])))
            return dets

        output = raw_outputs[0][0]

        if output.shape[0] < output.shape[1]:
            output = output.T

        orig_h, orig_w = orig_shape
        pad_x, pad_y   = pad
        dets = []

        for row in output:
            cx, cy, bw, bh, conf = row[:5]
            if conf < self.conf_threshold:
                continue

            x1 = int((cx - bw / 2 - pad_x) / scale)
            y1 = int((cy - bh / 2 - pad_y) / scale)
            x2 = int((cx + bw / 2 - pad_x) / scale)
            y2 = int((cy + bh / 2 - pad_y) / scale)

            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            if x2 > x1 and y2 > y1:
                dets.append((x1, y1, x2, y2, float(conf)))

        return self._nms(dets)

    def _nms(self, dets):
        if not dets:
            return []
        boxes  = np.array(
            [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in dets],
            np.float32
        )
        scores = np.array([d[4] for d in dets], np.float32)
        idx    = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.conf_threshold, self.iou_threshold
        )
        return [dets[i] for i in idx.flatten()] if len(idx) else []

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """BGR frame → list of (x1, y1, x2, y2, confidence)."""
        self._raw_frame = frame
        tensor, scale, pad = self._preprocess(frame)
        raw = self._infer(tensor)
        return self._postprocess(raw, frame.shape[:2], scale, pad)