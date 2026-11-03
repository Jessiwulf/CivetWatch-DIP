"""One command for a new observation day: prepare the videos, write the
manifest, analyse, report.  Used by the dashboard's "Process new footage".

    python -m civetwatch process --date 2026-04-05 --cam1 "D:/footage/camera 1" ... --cam4 ...

Every step can be stopped and started again: prepared videos and analysed
hours are kept, so it continues where it stopped.
"""
import glob
import json
import os
import shutil
import time
from datetime import datetime, timedelta

import yaml

from . import jobs
from .config import PROJECT_ROOT, load_manifest, timestamp_from_filename


def latest_setup(exclude=None):
    """The most recent earlier day with zones: the cameras do not move, so its
    zones and settings are reused."""
    days = sorted(d for d in glob.glob(os.path.join(PROJECT_ROOT, "data", "*"))
                  if os.path.isdir(os.path.join(d, "zones")) and os.path.basename(d) != exclude)
    return days[-1] if days else None


def process_day(date, cam_folders, cfg, rooms=None, out=None, progress=True):
    day_dir = os.path.join(PROJECT_ROOT, "data", date)
    out = out or os.path.join(PROJECT_ROOT, "runs", date)
    os.makedirs(day_dir, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    est = jobs.estimate(cam_folders)
    total = max(1.0, est["total_s"])
    p_prep = est["prepare_s"] / total
    p_an = est["analyse_s"] / total

    # 1. prepare: 2 fps, 960 px, 1-hour files named by their true start time
    from .video import prepare_video
    sp = cfg["sampling"]
    done_h, t_prep = 0.0, time.time()
    jobs.phase("Preparing the videos (2 frames per second)", 0.0, p_prep)
    for cid, folder in cam_folders.items():
        dst = os.path.join(day_dir, "videos", cid)
        os.makedirs(dst, exist_ok=True)
        for src in jobs.find_videos(folder):
            stem = os.path.splitext(os.path.basename(src))[0]
            marker = os.path.join(dst, f".done_{stem}")
            h = jobs.video_hours(src)
            if not os.path.exists(marker):
                start = timestamp_from_filename(src)
                if start is None:
                    raise SystemExit(f"{src}: no date/time in the file name (e.g. A01_20260405000000.mp4)")
                jobs.status(f"Preparing {cid}: {os.path.basename(src)}", done_h / max(est["camera_hours"], 1e-6))
                if progress:
                    print(f"prepare {cid} {os.path.basename(src)} ({h:.1f} h)", flush=True)
                prepare_video(src, dst, start, fps=sp["fps"], width=sp["process_width"])
                open(marker, "w").close()
            done_h += h
    if est["prepare_s"] and est["camera_hours"]:
        jobs.save_speed(prepare_s_per_cam_hour=(time.time() - t_prep) / est["camera_hours"]
                        if time.time() - t_prep > 60 else 0)

    # 2. manifest, zones and settings (copied from the last day: the cameras do not move)
    prev = latest_setup(exclude=date)
    zones_dir = os.path.join(day_dir, "zones")
    os.makedirs(zones_dir, exist_ok=True)
    for cid in cam_folders:
        z = os.path.join(zones_dir, f"{cid}.json")
        if not os.path.exists(z) and prev and os.path.exists(os.path.join(prev, "zones", f"{cid}.json")):
            shutil.copy(os.path.join(prev, "zones", f"{cid}.json"), z)
    if not os.path.exists(os.path.join(day_dir, "config.yaml")) and prev and \
            os.path.exists(os.path.join(prev, "config.yaml")):
        shutil.copy(os.path.join(prev, "config.yaml"), os.path.join(day_dir, "config.yaml"))
    rooms = rooms or {}
    d0 = datetime.strptime(date, "%Y-%m-%d")
    man = {"date": date, "start": f"{d0:%Y-%m-%d} 00:00:00", "end": f"{d0 + timedelta(days=1):%Y-%m-%d} 00:00:00",
           "cameras": [{"id": cid, "room": rooms.get(cid, f"R{cid[-1]}" if cid[-1].isdigit() else cid),
                        **({"zones": f"zones/{cid}.json"} if os.path.exists(os.path.join(zones_dir, f"{cid}.json")) else {}),
                        "files": f"videos/{cid}/*.mp4"} for cid in cam_folders]}
    manifest = os.path.join(day_dir, "manifest.yaml")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"# Observation day {date} - written by `civetwatch process`\n")
        yaml.safe_dump(man, f, sort_keys=False, allow_unicode=True)
    # keep only the hours this day really has video for
    m = load_manifest(manifest)
    from .video import CameraReader
    cover = [CameraReader(c).coverage() for c in m["cameras"]]
    cover = [c for c in cover if c[0] is not None]
    if not cover:
        raise SystemExit("No readable video was prepared")
    t0, t1 = max(m["start"], min(c[0] for c in cover)), min(m["end"], max(c[1] for c in cover))
    cfg_path = os.path.join(day_dir, "config.yaml")
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"manifest": os.path.relpath(manifest, PROJECT_ROOT).replace("\\", "/"),
                   "config": os.path.relpath(cfg_path, PROJECT_ROOT).replace("\\", "/") if os.path.exists(cfg_path) else None,
                   "sources": cam_folders, "created": f"{datetime.now():%Y-%m-%d %H:%M}"}, f, indent=1)

    # 3. analyse (resumable, hour by hour) + report
    from . import pipeline
    from .config import load_config
    if os.path.exists(cfg_path):
        cfg = load_config(cfg_path)
    t_an = time.time()
    jobs.phase("Finding the civet in every frame", p_prep, p_prep + p_an)
    pipeline.run(manifest, cfg, out, start=t0, end=t1, progress=progress)
    jobs.phase("Done", 1.0, 1.0)
    samples = (t1 - t0).total_seconds() * sp["fps"]
    if time.time() - t_an > 600:
        jobs.save_speed(analyse_samples_per_s=samples / (time.time() - t_an))
    return out
