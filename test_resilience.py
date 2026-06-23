"""
test_resilience.py

Tests the connection-loss resilience of RTSPFrameGrabber and GapLogger
without needing a real RTSP camera.

Simulates:
  0s  — stream starts healthy
  2s  — stream drops (cap.read returns ok=False)
  5s  — stream reconnects
  8s  — script stops, summary written

Run:
    python test_resilience.py

Expected result: PASSED for all three checks.
"""

import json
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ── import the classes we want to test ────────────────────────────────────────
from rtsp_face_capture import GapLogger, RTSPFrameGrabber, write_session_summary


# ─────────────────────────────────────────────────────────────────────────────
# Mock VideoCapture — controls exactly when read() succeeds or fails
# ─────────────────────────────────────────────────────────────────────────────

class MockCapture:
    """
    Pretends to be cv2.VideoCapture.
    stream_up controls whether read() returns a real frame or (False, None).

    isOpened() mirrors stream_up so the grabber's reconnect logic works:
      - stream_up=True  → isOpened()=True  → read() returns a frame
      - stream_up=False → read() returns (False,None) → grabber calls release()
                          then retries _connect() → isOpened()=False → keeps retrying
      - stream_up=True again → _connect() sees isOpened()=True → reconnected
    release() is a no-op so the same object is reused across reconnect cycles.
    """
    def __init__(self):
        self.stream_up = True

    def isOpened(self):
        return self.stream_up

    def set(self, *args):
        pass

    def read(self):
        if not self.stream_up:
            return False, None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        pass  # state is controlled entirely by stream_up


# ─────────────────────────────────────────────────────────────────────────────
# The test
# ─────────────────────────────────────────────────────────────────────────────

def run_test():
    print("=" * 55)
    print("  RESILIENCE TEST — no camera required")
    print("=" * 55)

    tmp = Path(tempfile.mkdtemp())
    print(f"  Output folder : {tmp}\n")

    mock_cap = MockCapture()

    # Patch cv2.VideoCapture so the grabber uses our mock instead
    with patch("rtsp_face_capture.cv2.VideoCapture", return_value=mock_cap):

        gap_logger = GapLogger(tmp / "session_gaps.jsonl")
        grabber    = RTSPFrameGrabber(
            rtsp_url="rtsp://fake-url",
            gap_logger=gap_logger,
            reconnect_delay=0.3,    # short delay so test runs fast
        )
        grabber.start()

        # ── Phase 1: healthy stream for 2 seconds ─────────────────────────
        print("  [0s] stream healthy...")
        time.sleep(2)
        assert grabber.is_streaming, "grabber should be STREAMING after 2s"
        frame = grabber.get_latest_frame()
        assert frame is not None, "should have a frame during healthy stream"
        print("  [2s] frame received OK")

        # ── Phase 2: simulate a drop ───────────────────────────────────────
        print("  [2s] simulating connection drop...")
        mock_cap.stream_up = False
        time.sleep(1.5)
        # grabber should have detected the drop and transitioned to RECONNECTING
        assert not grabber.is_streaming, "grabber should NOT be streaming after drop"
        print("  [3.5s] drop detected correctly")

        # ── Phase 3: simulate reconnect ────────────────────────────────────
        print("  [3.5s] simulating reconnect...")
        mock_cap.stream_up = True
        mock_cap._open     = True
        time.sleep(2)
        assert grabber.is_streaming, "grabber should be STREAMING after reconnect"
        print("  [5.5s] reconnect detected correctly")

        # ── Phase 4: run a bit more, then stop ────────────────────────────
        time.sleep(1)
        grabber.stop()

    # ── Write session summary ──────────────────────────────────────────────
    start_time = datetime.now()
    write_session_summary(tmp, start_time, frames_saved=12, faces_total=15)

    # ── Read and verify the gap log ────────────────────────────────────────
    gaps_path = tmp / "session_gaps.jsonl"
    assert gaps_path.exists(), "session_gaps.jsonl should exist"

    events = []
    with open(gaps_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    event_types = [e["event"] for e in events]
    print(f"\n  Gap events recorded : {event_types}")

    # ── Read and verify the session summary ───────────────────────────────
    summary_path = tmp / "session_summary.json"
    assert summary_path.exists(), "session_summary.json should exist"

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
        "connected" in event_types and
        "disconnected" in event_types and
        "reconnected" in event_types,
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
        f"frames_saved={summary['frames_saved']}  faces_detected={summary['faces_detected']}"
    )

    # ── Cleanup ───────────────────────────────────────────────────────────
    shutil.rmtree(tmp)

    # ── Final result ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    if all(results):
        print("  ALL CHECKS PASSED ✓")
    else:
        failed = results.count(False)
        print(f"  {failed} CHECK(S) FAILED — see above")
    print("=" * 55)

    return all(results)


if __name__ == "__main__":
    ok = run_test()
    exit(0 if ok else 1)
