"""
config.py — GStreamer + TensorRT pipeline settings (Jetson)
Edit RTSP URL via .env file — never hardcode credentials here.
"""

from dataclasses import dataclass

import os




@dataclass
class Config:

    # ── Camera ───────────────────────────────────────────────────────────────
    # Loaded from .env file — create .env with:
    # RTSP_URL=rtsp://user:password@IP:port/stream_path
    rtsp_url: str = "rtsp://admin:Admin%231234@192.168.100.164:554/video/live?channel=1&subtype=0"


    # ── GStreamer decode ──────────────────────────────────────────────────────
    # Jetson uses nvv4l2decoder — different from desktop hevc_cuvid
    # camera sends HEVC/H265 so we use rtph265depay + h265parse
    codec:        str = "hevc"          # hevc or h264
    decode_width:  int = 2560
    decode_height: int = 1440

    # ── TensorRT model ────────────────────────────────────────────────────────
    engine_path: str = "../../models/yolov8n-face-jetson.engine"
    input_w:     int = 640
    input_h:     int = 640

    # ── Detection ─────────────────────────────────────────────────────────────
    conf_threshold: float = 0.45
    iou_threshold:  float = 0.45

    # ── Performance ───────────────────────────────────────────────────────────
    process_every_n_frames: int = 1

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:           str  = "../../results/framework_b_captures"
    max_saves_per_minute: int  = 60
    save_annotated:       bool = True
