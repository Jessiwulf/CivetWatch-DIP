"""Step 1 · Pre-processing.

* read every camera on one shared clock, 2 frames per real second
* brighten + denoise night (infrared) frames (CLAHE + Non-Local Means)
* judge the observation condition: light, video quality
* FFmpeg helpers to split recordings into 1-hour files and pre-sample at 2 fps
"""
import os
import re
import shutil
import subprocess
from datetime import timedelta

import cv2
import numpy as np


# ---------------------------------------------------------------- reading ---
class CameraReader:
    """Returns the frame a camera showed at a given true time.

    Frames are read sequentially (grab() skips the frames we do not keep), so a
    whole day is decoded once. Files recorded at 2 fps by `civetwatch prepare`
    are read frame by frame without skipping.
    """

    def __init__(self, cam):
        self.cam = cam
        self.offset = timedelta(seconds=cam.get("clock_offset_s", 0))
        self.files = []
        prev_end = None
        for item in sorted(cam["files"], key=lambda f: (f["start"] is None, f["start"] or 0)):
            cap = cv2.VideoCapture(item["path"])
            if not cap.isOpened():
                print(f"[warn] {cam['id']}: cannot open {item['path']}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            start = item["start"] or prev_end
            if start is None:
                raise ValueError(f"{item['path']}: no start time (add 'start:' in the manifest)")
            dur = timedelta(seconds=n / fps if n > 0 else 3600)
            self.files.append({"path": item["path"], "start": start, "end": start + dur, "fps": fps, "frames": n})
            prev_end = start + dur
        self._cap = None
        self._file = None
        self._next = 0        # index of the next frame the decoder will return

    def coverage(self):
        if not self.files:
            return None, None
        return self.files[0]["start"] - self.offset, self.files[-1]["end"] - self.offset

    def _open(self, f):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(f["path"])
        self._file = f
        self._next = 0

    def locate(self, t):
        """(file, frame index) showing true time t, or (None, None)."""
        cam_t = t + self.offset
        for f in self.files:
            if f["start"] <= cam_t < f["end"]:
                idx = int(round((cam_t - f["start"]).total_seconds() * f["fps"]))
                return f, idx
        return None, None

    def read_at(self, t):
        """-> (frame or None, status) with status 'ok' | 'no_video' | 'dropped'."""
        f, idx = self.locate(t)
        if f is None:
            return None, "no_video"
        if f is not self._file:
            self._open(f)
        if idx < self._next or idx - self._next > 10 * f["fps"]:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next = idx
        while self._next < idx:
            if not self._cap.grab():
                return None, "dropped"
            self._next += 1
        ok, frame = self._cap.read()
        self._next += 1
        if not ok or frame is None:
            return None, "dropped"
        return frame, "ok"

    def close(self):
        if self._cap is not None:
            self._cap.release()


def sample_times(start, end, fps):
    step = timedelta(seconds=1.0 / fps)
    t = start
    while t < end:
        yield t
        t += step


# ---------------------------------------------------------- enhancement ---
def resize_width(frame, width):
    h, w = frame.shape[:2]
    if not width or w <= width:
        return frame
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def colorfulness(frame):
    """Hasler-Suesstrunk colourfulness without the mean term, so a uniformly
    tinted (pink) infrared picture still counts as colourless."""
    small = frame[::8, ::8].astype(np.float32)         # a thumbnail is enough and ~50x faster
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))


class Preprocessor:
    def __init__(self, cfg):
        self.p = cfg["preprocess"]
        self.width = cfg["sampling"]["process_width"]
        self.clahe = cv2.createCLAHE(clipLimit=self.p["clahe_clip"],
                                     tileGridSize=(self.p["clahe_grid"], self.p["clahe_grid"]))
        self.fps = cfg["sampling"]["fps"]
        self._prev = {}       # camera id -> previous raw gray frame (frozen-video check)
        self._frozen = {}     # camera id -> identical frames in a row

    def __call__(self, cam_id, frame):
        """-> (enhanced BGR frame, enhanced gray frame, condition dict)."""
        frame = resize_width(frame, self.width)
        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        night = colorfulness(frame) < self.p["night_colorfulness_max"]
        if night:
            gray = self.clahe.apply(raw_gray)
            method = self.p["denoise"]
            if method in ("nlm", True):
                gray = cv2.fastNlMeansDenoising(gray, None, self.p["denoise_strength"],
                                                self.p["nlm_template"], self.p["nlm_search"])
            elif method == "bilateral":
                gray = cv2.bilateralFilter(gray, 7, self.p["denoise_strength"] * 6, 7)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            gray = raw_gray

        reasons = []
        if cv2.Laplacian(gray[::2, ::2], cv2.CV_32F).var() < self.p["blur_min"]:
            reasons.append("blur")
        if gray.mean() < self.p["dark_max"]:
            reasons.append("too dark")
        prev = self._prev.get(cam_id)
        same = (prev is not None and prev.shape == raw_gray.shape
                and cv2.absdiff(prev, raw_gray).mean() < self.p["frozen_diff_max"])
        self._frozen[cam_id] = self._frozen.get(cam_id, 0) + 1 if same else 0
        if self._frozen[cam_id] >= self.p["frozen_min_s"] * self.fps:
            reasons.append("frozen video")
        self._prev[cam_id] = raw_gray

        condition = {
            "light": "night_ir" if night else "day_colour",
            "quality": "poor_image" if reasons else "clear",
            "quality_reason": ", ".join(reasons),
        }
        return frame, gray, condition


# --------------------------------------------------------------- ffmpeg ---
def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("FFmpeg not found. `pip install imageio-ffmpeg` or install ffmpeg.")


def prepare_video(src, out_dir, start, fps=2, segment_s=3600, width=960, hwaccel=None):
    """Split a recording into 1-hour files sampled at `fps`, each named with its
    true start time (cam1_20260720_150000.mp4) so the manifest can glob them."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    stem = re.sub(r"[_\-]?\d{8}[_\-]?\d{6}", "", stem) or "cam"      # the true start is added below
    tmp = os.path.join(out_dir, f"{stem}_part%03d.mp4")
    from .resources import settings, subprocess_flags
    n = int(settings().get("threads") or 0)
    thr = ["-threads", str(n)] if n else []
    hw = ["-hwaccel", hwaccel] if hwaccel else []          # e.g. "cuda": decode on the NVIDIA GPU
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "fatal", *hw, *thr, "-i", src,
           *(["-filter_threads", str(n)] if n else []),
           "-vf", f"fps={fps},scale='min({width},iw)':-2", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           *thr, "-f", "segment", "-segment_time", str(segment_s), "-reset_timestamps", "1", tmp]
    subprocess.run(cmd, check=True, creationflags=subprocess_flags())
    parts = [os.path.join(out_dir, p) for p in sorted(os.listdir(out_dir)) if p.startswith(f"{stem}_part")]
    return name_parts_by_time(parts, start, stem)


def name_parts_by_time(parts, start, stem):
    """FFmpeg cuts segments on keyframes, so they are not exactly one hour:
    each part's true start = recording start + the real length of the parts
    before it. Files are renamed stem_YYYYmmdd_HHMMSS.mp4."""
    outputs, t = [], start
    for path in parts:
        cap = cv2.VideoCapture(path)
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1)
        cap.release()
        dst = os.path.join(os.path.dirname(path), f"{stem}_{t:%Y%m%d_%H%M%S}.mp4")
        if os.path.abspath(dst) != os.path.abspath(path):
            os.replace(path, dst)
        outputs.append(dst)
        t += timedelta(seconds=length)
    return outputs
