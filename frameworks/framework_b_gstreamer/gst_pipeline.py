"""
gst_pipeline.py — Framework B: GStreamer + Python TensorRT
Jetson Orin — Ubuntu 22.04 — CUDA 12.6 — DeepStream 7.1

Pipeline:
    IP Camera (RTSP HEVC)
        → rtspsrc           : opens RTSP connection
        → rtph265depay      : strips RTP packet headers
        → h265parse         : parses HEVC stream
        → nvv4l2decoder     : Jetson VPU hardware decode → NV12 in GPU memory
        → nvvidconv         : GPU colour convert + resize
        → video/x-raw,BGR   : frame format Python can read
        → appsink           : hands frame to Python ← GPU→CPU crossing happens here
        → TensorRT          : Python runs inference
        → save detections

Note: This is Framework B — frame crosses GPU→CPU at appsink.
      Compare with Framework C (DeepStream) where frame never leaves GPU.
"""

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp, GLib

import sys
import cv2
import numpy as np
import time
import logging
from datetime import datetime
from pathlib import Path

from config import Config
from utils.trt_inference import TRTFaceDetector
from utils.frame_saver import FrameSaver

# ── Logging ──────────────────────────────────────────────────────────────────
Path("../../results").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"../../results/gst_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger("gst_pipeline")

# ── GStreamer init ────────────────────────────────────────────────────────────
Gst.init(None)


class GStreamerPipeline:
    """
    Framework B — GStreamer + Python TensorRT pipeline.

    GStreamer handles:  RTSP → VPU decode → GPU convert → appsink
    Python handles:     preprocess → TensorRT inference → save

    The handoff point is appsink — where the frame crosses from GPU to CPU RAM.
    """

    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self.pipeline = None
        self.loop     = None
        self.running  = False

        # Stats
        self.frame_count  = 0
        self.fps_frames   = 0
        self.fps_time     = time.time()

        log.info(f"Loading TensorRT engine: {cfg.engine_path}")
        self.detector = TRTFaceDetector(
            engine_path=cfg.engine_path,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            input_size=(cfg.input_w, cfg.input_h),
        )
        log.info("TensorRT engine loaded.")

        self.saver = FrameSaver(
            output_dir=cfg.output_dir,
            max_saves_per_minute=cfg.max_saves_per_minute,
            save_annotated=cfg.save_annotated,
        )

    # ── Build GStreamer pipeline string ───────────────────────────────────────

    def _build_pipeline_string(self) -> str:
        """
        Builds the GStreamer pipeline string for Jetson HEVC camera.

        Plugin chain:
            rtspsrc         — connects to RTSP URL, receives HEVC packets over TCP
            rtph265depay    — removes RTP headers, gives raw HEVC bytestream
            h265parse       — parses HEVC NAL units for decoder
            nvv4l2decoder   — Jetson VPU hardware decode, output is NV12 in GPU memory
            nvvidconv       — converts NV12→BGRx and resizes, stays on GPU
            videoconvert    — final BGRx→BGR conversion
            appsink         — pulls BGR frame into Python as numpy array

        Key Jetson difference vs desktop:
            Desktop uses: hevc_cuvid (FFmpeg) or nvdec (GStreamer desktop)
            Jetson uses:  nvv4l2decoder — talks to Jetson's dedicated VPU chip
        """

        # rtspsrc settings
        src = (
            f"rtspsrc location={self.cfg.rtsp_url} "
            f"latency=200 "
            f"protocols=tcp "           # TCP more stable on LAN
            f"! rtph265depay "
            f"! h265parse "
        )

        # Jetson VPU decode + GPU convert
        decode = (
            f"! nvv4l2decoder "
            f"! nvvidconv "
            f"! video/x-raw,format=BGRx,"
            f"width={self.cfg.decode_width},"
            f"height={self.cfg.decode_height} "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
        )

        # appsink — hands frame to Python
        # max-buffers=1 + drop=true = always latest frame, never queue up
        sink = (
            f"! appsink name=app_sink "
            f"max-buffers=1 "
            f"drop=true "
            f"emit-signals=true "
            f"sync=false"
        )

        pipeline_str = src + decode + sink
        log.info(f"Pipeline string:\n  {pipeline_str}")
        return pipeline_str

    # ── Frame callback ────────────────────────────────────────────────────────

    def _on_new_sample(self, appsink):
        """
        Called by GStreamer every time appsink has a new frame ready.
        This is the handoff point — frame crosses from GPU memory to CPU RAM here.

        Flow inside this function:
            GstSample → GstBuffer → map to bytes → numpy BGR array
            → TensorRT inference → save if faces detected
        """
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf  = sample.get_buffer()
        caps = sample.get_caps()

        # Get frame dimensions from GStreamer caps
        structure = caps.get_structure(0)
        width     = structure.get_value("width")
        height    = structure.get_value("height")

        # Map GStreamer buffer → raw bytes → numpy BGR array
        # This is the GPU→CPU memory crossing
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))   # BGR
            self._process_frame(frame.copy())
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    # ── Inference + save ──────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray):
        """Runs TensorRT detection on frame, saves if faces found."""
        self.frame_count += 1
        self.fps_frames  += 1

        if self.frame_count % self.cfg.process_every_n_frames != 0:
            return

        detections = self.detector.detect(frame)

        # FPS log every 5 seconds
        elapsed = time.time() - self.fps_time
        if elapsed >= 5.0:
            fps = self.fps_frames / elapsed
            log.info(
                f"FPS: {fps:.1f}  |  "
                f"Frames: {self.frame_count}  |  "
                f"Faces this frame: {len(detections)}"
            )
            self.fps_frames = 0
            self.fps_time   = time.time()

        if len(detections) > 0:
            self.saver.save(frame, detections)

    # ── Start / stop ──────────────────────────────────────────────────────────

    def start(self):
        pipeline_str = self._build_pipeline_string()

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            log.error(f"Failed to build pipeline: {e}")
            sys.exit(1)

        # Connect appsink callback
        appsink = self.pipeline.get_by_name("app_sink")
        appsink.connect("new-sample", self._on_new_sample)

        # Bus message handler — catches errors and EOS
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True
        log.info(f"Pipeline PLAYING — {self.cfg.rtsp_url}")

        # GLib main loop keeps GStreamer running
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            self.stop()

    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            log.info("Pipeline stopped.")
        if self.loop and self.loop.is_running():
            self.loop.quit()

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error(f"GStreamer ERROR: {err.message}")
            log.error(f"Debug: {debug}")
            self.stop()
        elif t == Gst.MessageType.EOS:
            log.info("End of stream.")
            self.stop()
        elif t == Gst.MessageType.WARNING:
            w, _ = message.parse_warning()
            log.warning(f"GStreamer WARNING: {w.message}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    log.info("=" * 60)
    log.info("  Framework B — GStreamer + Python TensorRT")
    log.info(f"  Camera  : {cfg.rtsp_url}")
    log.info(f"  Engine  : {cfg.engine_path}")
    log.info("=" * 60)
    app = GStreamerPipeline(cfg)
    app.start()
