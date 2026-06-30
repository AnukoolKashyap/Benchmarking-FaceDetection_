"""
config.py — Framework A settings for Jetson offline benchmark
"""

from dataclasses import dataclass


@dataclass
class Config:

    # ── TensorRT model ───────────────────────────────────────────────────────
    engine_path: str = "../../models/yolov8n-face-jetson.engine"
    input_w:     int = 640
    input_h:     int = 640

    # ── Detection thresholds ─────────────────────────────────────────────────
    conf_threshold: float = 0.45
    iou_threshold:  float = 0.45

    # ── Decode resolution ────────────────────────────────────────────────────
    decode_width:  int = 1280
    decode_height: int = 720

    # ── GPU decode codec ─────────────────────────────────────────────────────
    gpu_codec:   str = "hevc_cuvid"
    cuda_device: int = 0

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir:           str  = "../../results/framework_a_captures"
    save_annotated:       bool = True
    max_saves_per_minute: int  = 60