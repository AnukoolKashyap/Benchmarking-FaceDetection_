"""
rtsp_face_capture.py

Captures frames from an RTSP camera, detects faces with a YOLO face model,
and saves any frame containing >=1 face into an output folder, throttled
to a target rate (default 5 FPS). Frames with multiple faces are saved once
(with every box drawn) and each face is additionally cropped into its own
file if --save-crops is passed.

Setup:
    pip install ultralytics opencv-python

    Download YOLO face-detection weights (stock yolov8n.pt is trained on
    COCO and has no "face" class -- you need weights fine-tuned on a face
    dataset, e.g. WIDERFACE). Two drop-in options that load the same way
    as any Ultralytics model:
        https://github.com/akanametov/yolo-face   (yolov8n-face.pt ... yolov12n-face.pt)
        https://github.com/lindevs/yolov8-face
    Place the .pt file next to this script, or pass its path via --model.

Usage:
    python rtsp_face_capture.py --rtsp-url "rtsp://user:pass@192.168.1.50:554/stream1"
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


class RTSPFrameGrabber:
    """
    Reads frames from an RTSP stream in a background thread and exposes
    only the most recent frame. This decouples the camera's native frame
    rate from the (slower) inference loop, and prevents OpenCV's internal
    buffer from piling up stale frames -- the main source of "lag" when
    reading RTSP streams naively in a single loop.
    """

    def __init__(self, rtsp_url: str, reconnect_delay: float = 3.0):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def _connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Best-effort: keeps the FFMPEG-side buffer as small as possible.
        # Not every backend/platform honors this, which is exactly why we
        # also do the "always take the newest frame" trick at the app level.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def _run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print("[grabber] connecting to RTSP stream...")
                if not self._connect():
                    print(f"[grabber] connect failed, retrying in {self.reconnect_delay}s")
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self.cap.read()
            if not ok:
                print("[grabber] stream read failed, reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_delay)
                continue

            with self.lock:
                self.latest_frame = frame

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def get_latest_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()


def save_detection(frame, boxes, output_dir: Path, draw_boxes: bool, save_crops: bool):
    """Persist one frame that contains >=1 detected face, plus a log entry."""
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    face_count = len(boxes)

    annotated = frame.copy()
    box_records = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        conf = float(box.conf[0])
        box_records.append({"xyxy": [x1, y1, x2, y2], "confidence": round(conf, 4)})

        if draw_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{conf:.2f}", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        if save_crops:
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                crop_dir = output_dir / "crops"
                crop_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(crop_dir / f"{ts}_face{i}.jpg"), crop)

    filename = f"{ts}_faces{face_count}.jpg"
    cv2.imwrite(str(output_dir / filename), annotated)

    log_entry = {
        "timestamp": now.isoformat(),
        "filename": filename,
        "face_count": face_count,
        "boxes": box_records,
    }
    with open(output_dir / "detections.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return filename, face_count


def main():
    parser = argparse.ArgumentParser(description="RTSP -> YOLO face detection -> folder capture")
    parser.add_argument("--rtsp-url", required=True, help="RTSP stream URL")
    parser.add_argument("--model", default="yolov8n-face.pt", help="YOLO face-detection weights (.pt)")
    parser.add_argument("--output", default="captured_faces", help="Output folder")
    parser.add_argument("--fps", type=float, default=5.0, help="Target processing rate")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution (long side, px). Higher catches smaller/farther faces at the cost of speed per frame.")
    parser.add_argument("--no-boxes", action="store_true", help="Save raw frame without drawn boxes")
    parser.add_argument("--save-crops", action="store_true", help="Also save each face as a separate crop")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] loading model: {args.model}")
    model = YOLO(args.model)

    grabber = RTSPFrameGrabber(args.rtsp_url)
    grabber.start()

    frame_interval = 1.0 / args.fps
    next_tick = time.time()
    saved = 0

    print(f"[main] running at ~{args.fps} FPS, saving to '{output_dir}'. Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick = now + frame_interval

            frame = grabber.get_latest_frame()
            if frame is None:
                continue  # stream hasn't produced a frame yet

            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                filename, count = save_detection(
                    frame, boxes, output_dir,
                    draw_boxes=not args.no_boxes,
                    save_crops=args.save_crops,
                )
                saved += 1
                print(f"[main] saved {filename} ({count} face{'s' if count != 1 else ''})")

    except KeyboardInterrupt:
        print(f"\n[main] stopping. total frames saved: {saved}")
    finally:
        grabber.stop()


if __name__ == "__main__":
    main()
