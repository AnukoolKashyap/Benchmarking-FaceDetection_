# Connection resilience — RTSP face detection pipeline

## The problem in plain words

The pipeline runs for hours against a live camera. At some point the camera
will drop — a network hiccup, a power cycle, someone tripping over a cable.
Before this fix, two things happened silently when that occurred:

1. The grabber thread retried the connection automatically, which worked.
   But **nothing was written down** about the gap. There was no record of
   "stream was offline from 14:32 to 14:37." That five-minute hole just
   disappeared from the data.

2. Frames and detections already on disk stayed safe — they're written
   one at a time as they happen. But you had **no way to know** how many
   gaps occurred, how long they lasted, or what fraction of the run was
   actually active.

The fix adds two things: a **gap logger** that writes every disconnect and
reconnect to disk immediately, and a **session summary** that tells you
the full story of the run when you stop the script.

---

## What the state machine looks like

The grabber thread now moves between three explicit states:

```
CONNECTING  ──── first good frame ───►  STREAMING  ──── read() fails ───►  RECONNECTING
    ▲                                                                              │
    └──────────────────── next good frame ◄────────────────────────────────────────┘
```

Every transition writes one line to `session_gaps.jsonl` immediately — not
buffered, not batched. If the process crashes mid-run, everything up to
that moment is already on disk.

| Transition | What gets written |
|---|---|
| CONNECTING → STREAMING | `{"event": "connected", "iso": "..."}` |
| STREAMING → RECONNECTING | `{"event": "disconnected", "iso": "..."}` |
| RECONNECTING → STREAMING | `{"event": "reconnected", "gap_s": 9.8, "iso": "..."}` |

---

## Output files after a long run

```
captured_faces/
├── 20260623_090042_faces2.jpg     saved frames (unchanged, always were safe)
├── detections.jsonl               one line per detection (unchanged)
├── session_gaps.jsonl             one line per connection event  ← new
└── session_summary.json           written on clean Ctrl+C       ← new
```

`session_gaps.jsonl` looks like:

```json
{"event": "connected",     "iso": "2026-06-23T09:00:01.112"}
{"event": "disconnected",  "iso": "2026-06-23T11:32:44.003"}
{"event": "reconnected",   "gap_s": 9.8, "iso": "2026-06-23T11:32:53.801"}
```

`session_summary.json` looks like:

```json
{
  "session_start":   "2026-06-23T09:00:01",
  "session_end":     "2026-06-23T11:05:22",
  "duration_s":      7521.0,
  "active_stream_s": 7511.2,
  "total_gap_s":     9.8,
  "gap_count":       1,
  "frames_saved":    4821,
  "faces_detected":  5103
}
```

And the terminal prints this on shutdown:

```
=======================================================
  SESSION SUMMARY
=======================================================
  Started          : 2026-06-23 09:00:01
  Ended            : 2026-06-23 11:05:22
  Total duration   : 125.4 min
  Active stream    : 125.2 min
  Connection gaps  : 1  (total 9.8s offline)
  Frames saved     : 4821
  Faces detected   : 5103
  Summary written  : captured_faces/session_summary.json
=======================================================
```

---

## How to run

```bash
python rtsp_face_capture.py --rtsp-url "rtsp://user:pass@ip:554/stream1"
```

Everything else is automatic. There are no new flags — gap logging and
session summary happen on every run.

---

## How to test without a camera

A simulation test is included that verifies the full gap detection flow
without needing a real RTSP stream:

```bash
python test_resilience.py
```

The test runs in about 8 seconds and covers three checks:

| Check | What it verifies |
|---|---|
| Gap events logged correctly | `connected`, `disconnected`, `reconnected` all appear in `session_gaps.jsonl` in the right order |
| Session summary records the gap | `gap_count == 1` in `session_summary.json` |
| Session summary has correct frame counts | `frames_saved` and `faces_detected` match what was passed in |

Expected output:

```
=======================================================
  RESILIENCE TEST — no camera required
=======================================================
  [0s] stream healthy...
  [grabber] connected at 11:17:18
  [2s] frame received OK
  [2s] simulating connection drop...
  [grabber] connection lost at 11:17:20
  [3.5s] drop detected correctly
  [3.5s] simulating reconnect...
  [grabber] reconnected after 1.5s gap
  [5.5s] reconnect detected correctly

  [PASSED] Gap events logged correctly
  [PASSED] Session summary records the gap
  [PASSED] Session summary has correct frame counts

=======================================================
  ALL CHECKS PASSED ✓
=======================================================
```

---

## How the test works (without a camera)

The test replaces `cv2.VideoCapture` with a `MockCapture` class that
accepts the same calls but is controlled by a shared `stream_up` flag.

The key design: each `MockCapture` instance **snapshots** `stream_up` at
the moment it is created. This matters because the grabber's loop checks
`isOpened()` before calling `read()`:

- While stream is healthy, `isOpened()` returns True → `read()` returns
  a real frame → normal operation.
- After `stream_up = False`, the *existing* cap still has `isOpened()=True`
  (it was created while the stream was up) → `read()` now returns
  `(False, None)` → `_on_drop()` fires → state transitions to RECONNECTING.
- Reconnect attempts create *new* caps while `stream_up=False` →
  `isOpened()=False` → `_connect()` fails → grabber keeps retrying.
- After `stream_up = True`, the next new cap has `isOpened()=True` →
  `read()` returns a frame → `_on_reconnect()` fires → back to STREAMING.

This mirrors exactly what happens with a real camera drop — no live
network required.

---

## New classes added

### `GapLogger`

Append-writes one JSON line per event to `session_gaps.jsonl`. Uses a
threading lock so both the grabber thread and the main thread can write
safely. Every write is an immediate `open()` + `write()` — no buffering —
so a crash never loses a gap event.

### `RTSPFrameGrabber` (updated)

Accepts a `GapLogger` and tracks its own connection state. Calls
`gap_logger.log_connect()`, `log_disconnect()`, and `log_reconnect(gap_s)`
on each state transition. The `is_streaming` property lets the main
inference loop skip frames during a gap window rather than processing
a stale frame from before the drop.

### `write_session_summary`

Called in the `finally` block on `Ctrl+C`. Reads `session_gaps.jsonl`,
sums all gap durations, computes active stream time, and writes
`session_summary.json`. Safe to call even if the gaps file is empty
or missing.
