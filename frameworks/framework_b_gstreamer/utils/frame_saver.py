"""
utils/frame_saver.py — Saves frames with detected faces + JSON metadata
"""

import cv2
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

log = logging.getLogger("frame_saver")


class FrameSaver:
    def __init__(
        self,
        output_dir: str,
        max_saves_per_minute: int = 60,
        save_annotated: bool = True,
    ):
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_annotated       = save_annotated
        self.max_saves_per_minute = max_saves_per_minute
        self.save_count   = 0
        self._min_saves   = 0
        self._min_start   = time.time()

    def _rate_ok(self) -> bool:
        now = time.time()
        if now - self._min_start >= 60:
            self._min_saves = 0
            self._min_start = now
        if self._min_saves >= self.max_saves_per_minute:
            return False
        self._min_saves += 1
        return True

    def save(self, frame, detections: List[Tuple]):
        if not self._rate_ok():
            return

        self.save_count += 1
        ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stem = f"{ts}_{self.save_count:04d}"

        # Optionally draw bounding boxes
        out = frame.copy()
        if self.save_annotated:
            for x1, y1, x2, y2, conf in detections:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    out, f"{conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

        cv2.imwrite(
            str(self.output_dir / f"{stem}.jpg"), out,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        meta = {
            "timestamp":  datetime.now().isoformat(),
            "frame_index": self.save_count,
            "face_count":  len(detections),
            "detections": [
                {"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3],
                 "conf": round(d[4], 4)}
                for d in detections
            ],
        }
        with open(self.output_dir / f"{stem}.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info(f"Saved {stem}.jpg  ({len(detections)} face(s))")
