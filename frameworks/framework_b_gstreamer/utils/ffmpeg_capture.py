"""
utils/ffmpeg_capture.py — RTSP capture using FFmpeg GPU decode (NVDEC via PyAV)

What this does:
    1. Opens RTSP stream using libavformat (AVFormatContext)
    2. Attaches AVHWDeviceContext (CUDA) to libavcodec
    3. Uses h264_cuvid codec — routes H264 decode to NVDEC chip on GPU
    4. Decoded frame lands in VRAM as NV12 format
    5. Transfers NV12 → BGR numpy array for TensorRT / saving

PyAV is the Python wrapper around FFmpeg's libav libraries.
av.open()        → AVFormatContext   (libavformat)
stream.codec_context → AVCodecContext (libavcodec)
frame            → AVFrame           (decoded image)
"""

import av
import av.codec
import numpy as np
import logging
import time
from typing import Optional

log = logging.getLogger("ffmpeg_capture")


class FFmpegCapture:
    """
    Opens an RTSP stream and decodes frames using NVDEC (GPU).

    Usage:
        cap = FFmpegCapture(rtsp_url="rtsp://...", gpu_codec="h264_cuvid")
        cap.open()
        while True:
            frame = cap.read()   # returns BGR numpy array
        cap.close()
    """

    def __init__(
        self,
        rtsp_url: str,
        gpu_codec: str  = "h264_cuvid",   # routes to NVDEC for H264
        cuda_device: int = 0,
        width: int  = 1280,
        height: int = 720,
        reconnect_delay: float = 3.0,
    ):
        self.rtsp_url        = rtsp_url
        self.gpu_codec       = gpu_codec
        self.cuda_device     = cuda_device
        self.width           = width
        self.height          = height
        self.reconnect_delay = reconnect_delay

        self._container = None   # AVFormatContext (libavformat)
        self._stream    = None   # video stream inside the container
        self._open      = False

    # ── Open stream ──────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Opens the RTSP stream with GPU decode.

        Key options explained:
            rtsp_transport=tcp   — force TCP, more stable than UDP on LAN
            hwaccel=cuda         — tells FFmpeg to use CUDA for decode
            hwaccel_device=0     — which GPU (0 = first GPU)
            vcodec=h264_cuvid    — use h264_cuvid decoder = NVDEC for H264
        """
        try:
            log.info(f"Opening RTSP stream: {self.rtsp_url}")
            log.info(f"GPU decode: {self.gpu_codec} on cuda:{self.cuda_device}")

            # These options are passed directly to libavformat / libavcodec
            # Same as: ffmpeg -rtsp_transport tcp -hwaccel cuda -c:v h264_cuvid -i <url>
            options = {
                "rtsp_transport": "tcp",          # stable over LAN
                "stimeout":       "5000000",       # 5 sec connection timeout
                "fflags":         "nobuffer",      # low latency — don't buffer
                "flags":          "low_delay",     # low latency mode
                "hwaccel":        "cuda",          # enable CUDA hardware accel
                "hwaccel_device": str(self.cuda_device),
                "vcodec":         self.gpu_codec,  # h264_cuvid → NVDEC
            }

            # av.open = AVFormatContext — opens the RTSP connection
            self._container = av.open(
                self.rtsp_url,
                options=options,
                timeout=10.0,
            )

            # Get the video stream from the container
            self._stream = self._container.streams.video[0]

            # Tell PyAV we want BGR24 frames out — libswscale converts for us
            # This handles NV12 → BGR conversion automatically
            self._stream.codec_context.pix_fmt = "bgr24"

            log.info(
                f"Stream opened — "
                f"{self._stream.codec_context.width}x{self._stream.codec_context.height} "
                f"@ {float(self._stream.average_rate):.1f} FPS  |  "
                f"codec: {self._stream.codec_context.name}"
            )

            self._open = True
            return True

        except av.AVError as e:
            log.error(f"FFmpeg error opening stream: {e}")
            log.error("Check: RTSP URL correct? Camera reachable? GPU codec available?")
            return False
        except Exception as e:
            log.error(f"Unexpected error opening stream: {e}")
            return False

    # ── Read one frame ────────────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """
        Reads and decodes the next frame from the RTSP stream.

        Flow inside this function:
            libavformat reads next H264 packet from network
            → libavcodec (h264_cuvid) sends packet to NVDEC
            → NVDEC decodes H264 → NV12 in VRAM
            → libswscale converts NV12 → BGR24 in RAM
            → we wrap it as a numpy array and return it

        Returns:
            BGR numpy array of shape (H, W, 3) or None on error
        """
        if not self._open or self._container is None:
            return None

        try:
            # Decode next video frame — PyAV handles the packet/decode loop
            for frame in self._container.decode(self._stream):
                # frame is an AVFrame — convert to numpy BGR array
                # to_ndarray("bgr24") calls libswscale internally
                img = frame.to_ndarray(format="bgr24")
                return img

        except av.AVError as e:
            log.warning(f"FFmpeg decode error: {e} — attempting to continue")
            return None
        except Exception as e:
            log.error(f"Unexpected read error: {e}")
            return None

        return None

    # ── Generator — yields frames continuously ────────────────────────────────

    def frames(self):
        """
        Generator that yields BGR frames continuously.
        Use instead of read() for cleaner loop syntax:

            for frame in capture.frames():
                detections = detector.detect(frame)
        """
        if not self._open:
            return

        try:
            for frame in self._container.decode(self._stream):
                yield frame.to_ndarray(format="bgr24")
        except av.AVError as e:
            log.warning(f"Stream ended or error: {e}")
        except KeyboardInterrupt:
            pass

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        if self._stream:
            return float(self._stream.average_rate)
        return 0.0

    @property
    def resolution(self):
        if self._stream:
            ctx = self._stream.codec_context
            return (ctx.width, ctx.height)
        return (0, 0)

    # ── Close ─────────────────────────────────────────────────────────────────

    def close(self):
        if self._container:
            self._container.close()
            self._container = None
        self._open = False
        log.info("FFmpeg capture closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
