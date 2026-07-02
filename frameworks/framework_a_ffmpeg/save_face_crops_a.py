"""
save_face_crops_a.py — Framework A: FFmpeg + PyAV
Saves cropped face images (not full frame) to results2/framework_a/

Each detected face is cropped and saved separately.
Filename format: frame_XXXX_face_N_confXX.jpg

Usage:
    python3 save_face_crops_a.py
    python3 save_face_crops_a.py --videos ../../videos --max-faces 100
"""

import av
import cv2
import logging
import argparse
import numpy as np
from pathlib import Path

from config import Config
from utils.trt_inference import TRTFaceDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("face_crops_a")

OUTPUT_DIR = Path("../../results2/framework_a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Padding around face crop (pixels)
CROP_PADDING = 20


def crop_face(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop face region with padding."""
    h, w = frame.shape[:2]
    x1p = max(0, x1 - CROP_PADDING)
    y1p = max(0, y1 - CROP_PADDING)
    x2p = min(w, x2 + CROP_PADDING)
    y2p = min(h, y2 + CROP_PADDING)
    return frame[y1p:y2p, x1p:x2p]


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

        video_folder = OUTPUT_DIR / video_path.stem
        video_folder.mkdir(exist_ok=True)

        try:
            container    = av.open(str(video_path))
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"
        except Exception as e:
            log.error(f"Could not open {video_path.name}: {e}")
            continue

        frame_count = 0
        saved_count = 0

        for packet in container.demux(video_stream):
            if packet.size == 0:
                continue
            try:
                frames = list(packet.decode())
            except Exception:
                continue
            if not frames:
                continue

            img = frames[0].to_ndarray(format="bgr24")
            frame_count += 1

            detections = detector.detect(img)

            for face_idx, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
                crop = crop_face(img, x1, y1, x2, y2)

                if crop.size == 0:
                    continue

                filename  = f"frame_{frame_count:04d}_face_{face_idx}_conf{int(conf*100):02d}.jpg"
                save_path = video_folder / filename
                cv2.imwrite(str(save_path), crop)
                saved_count += 1

                if saved_count % 20 == 0:
                    log.info(f"  {saved_count} faces saved...")

            if saved_count >= max_faces:
                break

        container.close()
        total_saved += saved_count
        log.info(f"  {video_path.name}: {saved_count} face crops → {video_folder.name}/")

    log.info(f"\nDone — {total_saved} face crops saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos",    default="../../videos")
    parser.add_argument("--max-faces", type=int, default=100,
                        help="Max face crops to save per video (default: 100)")
    args = parser.parse_args()

    process_videos(args.videos, args.max_faces)
