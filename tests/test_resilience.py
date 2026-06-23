"""
test_resilience.py

Tests connection-loss resilience of RTSPFrameGrabber and GapLogger
without needing a real RTSP camera.

Simulates:
  0s  — stream starts healthy
  2s  — stream drops (read returns False)
  5s  — stream reconnects
  7s  — stop, session summary written

Run:
    python test_resilience.py
"""

import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rtsp_face_capture import GapLogger, RTSPFrameGrabber, write_session_summary


# ─────────────────────────────────────────────────────────────────────────────
# Shared stream state — a list so threads can mutate it
# ─────────────────────────────────────────────────────────────────────────────

stream_up = [True]   # [0] is read/written by both the test thread and the grabber thread


# ─────────────────────────────────────────────────────────────────────────────
# MockCapture
#
# Key design: each time cv2.VideoCapture() is called (i.e. each _connect()
# attempt), a NEW MockCapture instance is created.  The instance snapshots
# stream_up[0] at the moment it was created — that is what isOpened() returns.
#
# Why this works:
#   • Healthy stream  → new cap created while stream_up=True  → isOpened()=True
#                       → read() returns a frame
#   • After drop      → existing cap still has _connected=True → isOpened()=True
#                       → read() returns (False,None) → _on_drop() fires ✓
#                       → release() marks this instance dead
#   • Reconnect retry → new cap created while stream_up=False → isOpened()=False
#                       → _connect() returns False → grabber keeps retrying ✓
#   • Stream back up  → new cap created while stream_up=True  → isOpened()=True
#                       → read() returns a frame → _on_reconnect() fires ✓
# ─────────────────────────────────────────────────────────────────────────────

class MockCapture:
    def __init__(self, *args, **kwargs):
        # Snapshot stream state at the moment of this "connection attempt"
        self._connected = stream_up[0]
        self._released  = False

    def isOpened(self):
        return self._connected and not self._released

    def set(self, *args):
        pass

    def read(self):
        if not stream_up[0]:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self._released = True


# ─────────────────────────────────────────────────────────────────────────────
# The test
# ─────────────────────────────────────────────────────────────────────────────

def run_test():
    print("=" * 55)
    print("  RESILIENCE TEST — no camera required")
    print("=" * 55)

    tmp = Path(tempfile.mkdtemp())
    print(f"  Output folder : {tmp}\n")

    # Reset shared state
    stream_up[0] = True

    # side_effect=MockCapture means every cv2.VideoCapture() call
    # instantiates a fresh MockCapture — critical for the snapshot logic above
    with patch("rtsp_face_capture.cv2.VideoCapture", side_effect=MockCapture):

        gap_logger = GapLogger(tmp / "session_gaps.jsonl")
        grabber    = RTSPFrameGrabber(
            rtsp_url="rtsp://fake-url",
            gap_logger=gap_logger,
            reconnect_delay=0.3,
        )
        grabber.start()

        # ── Phase 1: healthy stream for 2 seconds ─────────────────────────
        print("  [0s] stream healthy...")
        time.sleep(2)
        assert grabber.is_streaming, "grabber should be STREAMING"
        assert grabber.get_latest_frame() is not None, "should have a frame"
        print("  [2s] frame received OK")

        # ── Phase 2: simulate drop ─────────────────────────────────────────
        print("  [2s] simulating connection drop...")
        stream_up[0] = False
        # Give the grabber thread time to call read(), get False, and
        # transition state.  reconnect_delay=0.3s so 1.5s is plenty.
        time.sleep(1.5)
        assert not grabber.is_streaming, \
            "grabber should NOT be streaming after drop"
        print("  [3.5s] drop detected correctly")

        # ── Phase 3: simulate reconnect ────────────────────────────────────
        print("  [3.5s] simulating reconnect...")
        stream_up[0] = True
        time.sleep(2)
        assert grabber.is_streaming, \
            "grabber should be STREAMING after reconnect"
        print("  [5.5s] reconnect detected correctly")

        # ── Phase 4: run a moment longer then stop ─────────────────────────
        time.sleep(1)
        grabber.stop()

    # ── Session summary ────────────────────────────────────────────────────
    write_session_summary(tmp, datetime.now(), frames_saved=12, faces_total=15)

    # ── Read gap log ───────────────────────────────────────────────────────
    gaps_path = tmp / "session_gaps.jsonl"
    assert gaps_path.exists(), "session_gaps.jsonl missing"

    events = []
    with open(gaps_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    event_types = [e["event"] for e in events]
    print(f"\n  Gap events recorded : {event_types}")

    # ── Read session summary ───────────────────────────────────────────────
    summary_path = tmp / "session_summary.json"
    assert summary_path.exists(), "session_summary.json missing"

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    print(f"  Gap count in summary: {summary['gap_count']}")
    print(f"  Total gap seconds   : {summary['total_gap_s']}s")
    print(f"  Frames saved        : {summary['frames_saved']}")
    print(f"  Faces detected      : {summary['faces_detected']}")

    # ── Assertions ────────────────────────────────────────────────────────
    results = []

    def check(name, condition, detail=""):
        status = "PASSED" if condition else "FAILED"
        results.append(condition)
        print(f"\n  [{status}] {name}")
        if detail:
            print(f"           {detail}")

    check(
        "Gap events logged correctly",
        "connected"    in event_types and
        "disconnected" in event_types and
        "reconnected"  in event_types,
        f"got: {event_types}"
    )

    check(
        "Session summary records the gap",
        summary["gap_count"] == 1,
        f"gap_count = {summary['gap_count']} (expected 1)"
    )

    check(
        "Session summary has correct frame counts",
        summary["frames_saved"] == 12 and summary["faces_detected"] == 15,
        f"frames_saved={summary['frames_saved']}  "
        f"faces_detected={summary['faces_detected']}"
    )

    # ── Cleanup ───────────────────────────────────────────────────────────
    shutil.rmtree(tmp)

    # ── Result ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    if all(results):
        print("  ALL CHECKS PASSED ✓")
    else:
        print(f"  {results.count(False)} CHECK(S) FAILED — see above")
    print("=" * 55)

    return all(results)


if __name__ == "__main__":
    ok = run_test()
    exit(0 if ok else 1)