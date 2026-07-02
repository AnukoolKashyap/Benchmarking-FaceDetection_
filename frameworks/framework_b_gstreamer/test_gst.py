"""
test_gst.py — Quick test to confirm GStreamer pipeline works on Jetson
Run this BEFORE benchmark_b.py to confirm:
  1. Camera is reachable
  2. nvv4l2decoder works
  3. appsink receives frames
  4. gi/GStreamer Python binding works

Usage:
    python3 test_gst.py
    python3 test_gst.py --url "rtsp://user:pass@ip/stream"
"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np
import argparse
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_gst")

Gst.init(None)

frame_count = 0
start_time  = None
loop        = None


def on_new_sample(appsink):
    global frame_count, start_time

    sample = appsink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR

    buf  = sample.get_buffer()
    caps = sample.get_caps()
    s    = caps.get_structure(0)
    w    = s.get_value("width")
    h    = s.get_value("height")

    ok, map_info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.ERROR

    try:
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((h, w, 3))
        frame_count += 1

        if start_time is None:
            start_time = time.time()
            log.info(f"First frame received! Shape: {frame.shape}  dtype: {frame.dtype}")

        elapsed = time.time() - start_time
        if elapsed > 0 and frame_count % 25 == 0:
            fps = frame_count / elapsed
            log.info(f"Frames: {frame_count}  FPS: {fps:.1f}  Resolution: {w}x{h}")

        # Stop after 100 frames
        if frame_count >= 100:
            log.info(f"Test complete — {frame_count} frames @ {frame_count/elapsed:.1f} FPS")
            GLib.idle_add(loop.quit)

    finally:
        buf.unmap(map_info)

    return Gst.FlowReturn.OK


def on_bus_message(bus, message):
    if message.type == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        log.error(f"GStreamer ERROR: {err.message}")
        log.error(f"Debug: {dbg}")
        GLib.idle_add(loop.quit)
    elif message.type == Gst.MessageType.WARNING:
        w, _ = message.parse_warning()
        log.warning(f"GStreamer WARNING: {w.message}")


def main(rtsp_url: str, width: int = 1280, height: int = 720):
    global loop

    pipeline_str = (
        f"rtspsrc location={rtsp_url} latency=200 protocols=tcp "
        f"! rtph265depay ! h265parse "
        f"! nvv4l2decoder "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx,width={width},height={height} "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink name=test_sink max-buffers=1 drop=true emit-signals=true sync=false"
    )

    log.info(f"Testing pipeline:")
    log.info(f"  URL: {rtsp_url}")
    log.info(f"  Pipeline: {pipeline_str}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        log.error(f"Failed to build pipeline: {e}")
        sys.exit(1)

    sink = pipeline.get_by_name("test_sink")
    sink.connect("new-sample", on_new_sample)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message)

    pipeline.set_state(Gst.State.PLAYING)
    log.info("Pipeline PLAYING — waiting for frames...")

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="RTSP URL (overrides .env)")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if args.url:
        rtsp_url = args.url
    else:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        rtsp_url = os.getenv("RTSP_URL")
        if not rtsp_url:
            log.error("No RTSP_URL found. Either pass --url or create .env file.")
            sys.exit(1)

    main(rtsp_url, args.width, args.height)
