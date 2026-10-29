"""Labelled video clips: every number in the sheet links back to the moment it
came from. Clips are cut from the original video (full frame rate) with the
nearest analysed sample drawn on top."""
import contextlib
import json
import os
import sys
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd

from .overlay import draw_record
from .video import CameraReader, Preprocessor
from .zones import ZoneMap

CODECS = [("VP80", ".webm"), ("mp4v", ".mp4")]
_codec_cache = {}


@contextlib.contextmanager
def _quiet_stderr():
    """OpenCV prints 'FFMPEG: tag 0x30385056/'VP80' is not supported with codec id 139
    and format webm' every time a WebM file is opened. It is harmless (WebM files
    simply do not store that tag; the clip is written correctly) and it bypasses
    OpenCV's log settings, so it is hidden here at the file-descriptor level."""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, ValueError, OSError):
        yield
        return
    sys.stderr.flush()
    saved = os.dup(fd)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(saved, fd)
            os.close(saved)


def open_writer(base_path, fps, size):
    """First codec that works here; VP8 (WebM) plays in the browser dashboard."""
    order = [_codec_cache["ok"]] if "ok" in _codec_cache else CODECS
    for fourcc, ext in order:
        path = base_path + ext
        with _quiet_stderr():
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            _codec_cache["ok"] = (fourcc, ext)
            return w, path
        w.release()
        if os.path.exists(path):
            os.remove(path)
    raise RuntimeError("No working video codec for clips")


def _record(row):
    rec = row.to_dict()
    kp = rec.get("kp")
    rec["kp"] = np.array(json.loads(kp), float) if isinstance(kp, str) and kp else None
    return rec


def _tray_polys(rec, zonemap):
    trays = rec.get("trays")
    if isinstance(trays, str) and trays:
        for ztype, box in json.loads(trays).items():
            zonemap.update_dynamic(ztype, [box], rec["t"])
    return zonemap.polygons(rec["t"])


def write_clip(cam, frames, t0, t1, out_base, cfg, fps_out=None):
    """Cut [t0, t1] of one camera; frames = that camera's analysed samples."""
    reader = CameraReader(cam)
    light_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "denoise": "none"}}
    pre = Preprocessor(light_cfg)       # same size and night contrast boost as the analysis
    f, _ = reader.locate(t0)
    if f is None:
        return None
    fps = fps_out or min(f["fps"], 12.5)
    step = timedelta(seconds=1.0 / fps)
    zonemap = ZoneMap(cam.get("zones"))
    times = frames["t"].to_numpy()
    writer, path, t, thumb = None, None, t0, None
    mid = t0 + (t1 - t0) / 2
    while t <= t1:
        frame, status = reader.read_at(t)
        if frame is not None:
            frame, _, _ = pre(cam["id"], frame)
            h, w = frame.shape[:2]
            zonemap.set_size(w, h)
            k = np.searchsorted(times, np.datetime64(t), side="right") - 1
            if k >= 0:
                rec = _record(frames.iloc[k])
                rec["t"] = pd.Timestamp(t).to_pydatetime()
                draw_record(frame, rec, _tray_polys(rec, zonemap))
            if writer is None:
                writer, path = open_writer(out_base, fps, (w, h))
            writer.write(frame)
            if thumb is None and t >= mid:
                thumb = frame.copy()
        t += step
    reader.close()
    if writer is None:
        return None
    writer.release()
    if thumb is not None:
        cv2.imwrite(out_base + ".jpg", thumb)
    return path


def make_clips(events, frames, cams, out_dir, cfg):
    """Clips for every event a keeper should check (low confidence or rare),
    then the remaining frequency events, up to report.max_clips."""
    r = cfg["report"]
    os.makedirs(out_dir, exist_ok=True)
    cam_by_id = {c["id"]: c for c in cams}
    pad = timedelta(seconds=r["clip_pad_s"])
    review = events["needs_review"].astype(bool)
    freq = events["category"] == "frequency"
    stop = events["category"] == "stop"
    order = pd.concat([events[freq & review], events[stop & review], events[freq & ~review]])
    paths = {}
    for idx, ev in order.head(r["max_clips"]).iterrows():
        cam = cam_by_id.get(ev["camera"])
        if cam is None:
            continue
        base = os.path.join(out_dir, f"{ev['start']:%H%M%S}_{ev['behavior']}_{ev['camera']}")
        cam_frames = frames[frames["camera"] == ev["camera"]]
        path = write_clip(cam, cam_frames, ev["start"] - pad, ev["stop"] + pad, base, cfg)
        if path:
            paths[idx] = os.path.relpath(path, os.path.dirname(out_dir)).replace("\\", "/")
    return paths
