"""
save_face_crops_b.py — Framework B: GStreamer + TRT
Saves cropped face images (not full frame) to results2/framework_b/

Each detected face is cropped and saved separately.
Filename format: frame_XXXX_face_N_confXX.jpg

Usage:
    python3 save_face_crops_b.py
    python3 save_face_crops_b.py --videos ../../videos --max-faces 100
"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import cv2
import numpy as np
import logging
import argparse
import threading
from pathlib import Path

from config import Config
from utils.trt_inference import TRTFaceDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("face_crops_b")

Gst.init(None)

OUTPUT_DIR = Path("../../results2/framework_b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_PADDING = 20


def crop_face(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop face region with padding."""
    h, w = frame.shape[:2]
    x1p = max(0, x1 - CROP_PADDING)
    y1p = max(0, y1 - CROP_PADDING)
    x2p = min(w, x2 + CROP_PADDING)
    y2p = min(h, y2 + CROP_PADDING)
    return frame[y1p:y2p, x1p:x2p]


class FaceCropperB:

    def __init__(self, detector: TRTFaceDetector, cfg: Config,
                 video_path: Path, max_faces: int):
        self.detector    = detector
        self.cfg         = cfg
        self.video_path  = video_path
        self.max_faces   = max_faces
        self.frame_count = 0
        self.saved_count = 0
        self.done_event  = threading.Event()
        self.loop        = GLib.MainLoop()

        self.video_folder = OUTPUT_DIR / video_path.stem
        self.video_folder.mkdir(exist_ok=True)

    def run(self):
        pipeline_str = (
            f"filesrc location={self.video_path} "
            f"! decodebin "
            f"! nvvidconv "
            f"! video/x-raw,format=BGRx,"
            f"width={self.cfg.decode_width},height={self.cfg.decode_height} "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink name=crop_sink max-buffers=1 drop=false "
            f"emit-signals=true sync=false"
        )

        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            log.error(f"Pipeline failed: {e}")
            return

        sink = pipeline.get_by_name("crop_sink")
        sink.connect("new-sample", self._on_sample)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus)

        pipeline.set_state(Gst.State.PLAYING)

        t = threading.Thread(target=self.loop.run, daemon=True)
        t.start()
        self.done_event.wait(timeout=300)
        pipeline.set_state(Gst.State.NULL)

        log.info(
            f"  {self.video_path.name}: "
            f"{self.saved_count} face crops → {self.video_folder.name}/"
        )

    def _on_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf  = sample.get_buffer()
        caps = sample.get_caps()
        s    = caps.get_structure(0)
        w    = s.get_value("width")
        h    = s.get_value("height")

        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR

        try:
            frame = np.frombuffer(mi.data, dtype=np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mi)

        self.frame_count += 1
        detections = self.detector.detect(frame)

        for face_idx, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
            crop = crop_face(frame, x1, y1, x2, y2)

            if crop.size == 0:
                continue

            filename  = f"frame_{self.frame_count:04d}_face_{face_idx}_conf{int(conf*100):02d}.jpg"
            save_path = self.video_folder / filename
            cv2.imwrite(str(save_path), crop)
            self.saved_count += 1

            if self.saved_count % 20 == 0:
                log.info(f"  {self.saved_count} faces saved...")

        if self.saved_count >= self.max_faces:
            self.done_event.set()
            GLib.idle_add(self.loop.quit)

        return Gst.FlowReturn.OK

    def _on_bus(self, bus, message):
        if message.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            self.done_event.set()
            GLib.idle_add(self.loop.quit)


def process_videos(videos_dir: str, max_faces: int = 100):
    cfg      = Config()
    detector = TRTFaceDetector(
        engine_path=cfg.engine_path,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        input_size=(cfg.input_w, cfg.input_h),
    )
    log.info("TRT engine loaded.")

    video_files = sorted(Path(videos_dir).glob("*.dav"))
    if not video_files:
        log.error(f"No .dav files found in {videos_dir}")
        return

    log.info(f"Found {len(video_files)} videos — saving face crops to {OUTPUT_DIR}")

    total_saved = 0
    for video_path in video_files:
        log.info(f"\nProcessing: {video_path.name}")
        cropper = FaceCropperB(detector, cfg, video_path, max_faces)
        cropper.run()
        total_saved += cropper.saved_count

    log.info(f"\nDone — {total_saved} face crops saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos",    default="../../videos")
    parser.add_argument("--max-faces", type=int, default=100)
    args = parser.parse_args()

    process_videos(args.videos, args.max_faces)
