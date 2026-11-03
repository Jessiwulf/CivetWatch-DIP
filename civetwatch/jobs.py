"""Background jobs started from the dashboard (process new footage, rebuild a
report, train the behavior model, make a spot check), their progress and a
time estimate. One job at a time, so the laptop is never overloaded.

A job is `python -m civetwatch ...` in its own process. It reports progress by
writing a small JSON file named in the CIVETWATCH_STATUS environment variable
(see status()); the dashboard reads that file.
"""
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from .config import PROJECT_ROOT

JOBS_DIR = os.path.join(PROJECT_ROOT, "runs", "_jobs")
SPEED_FILE = os.path.join(JOBS_DIR, "speed.json")
# measured on the 5 Apr 2026 day with the laptop profile (RTX 3050 Ti); replaced by each finished job
DEFAULT_SPEED = {"prepare_s_per_cam_hour": 60.0, "analyse_samples_per_s": 13.0, "report_s": 600.0}

# ------------------------------------------------------------ progress ---
_st = {"stage": "", "lo": 0.0, "hi": 1.0, "base": 0.0, "weight": 0.0, "last": 0.0, "t0": time.time()}


def phase(name, lo, hi):
    """Map the next status() fractions into [lo, hi] of the whole job."""
    _st.update(stage=name, lo=lo, hi=hi, base=0.0, weight=0.0)
    _write(force=True)


def status(stage=None, frac=None, weight=None, sub=None, rate=None):
    """Progress from inside the pipeline. frac = part of the current phase done,
    weight = size of the piece now starting, sub = part of that piece done."""
    if not os.environ.get("CIVETWATCH_STATUS"):
        return
    force = False
    if stage is not None and stage != _st["stage"]:
        _st["stage"], force = stage, True
    if frac is not None:
        _st["base"], _st["weight"] = frac, (weight or 0.0)
    if sub is not None:
        _st["sub"] = sub
    else:
        _st["sub"] = 0.0 if frac is not None else _st.get("sub", 0.0)
    if rate is not None:
        _st["rate"] = rate
    _write(force)


def _write(force=False):
    path = os.environ.get("CIVETWATCH_STATUS")
    now = time.time()
    if not path or (not force and now - _st["last"] < 2):
        return
    _st["last"] = now
    inner = min(1.0, _st["base"] + _st["weight"] * _st.get("sub", 0.0))
    p = _st["lo"] + (_st["hi"] - _st["lo"]) * inner
    data = {"stage": _st["stage"], "progress": round(p, 4), "updated": now, "rate": _st.get("rate")}
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


# ------------------------------------------------------------ estimate ---
def speeds():
    out = dict(DEFAULT_SPEED)
    try:
        out.update(json.load(open(SPEED_FILE, encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return out


def save_speed(**kw):
    os.makedirs(JOBS_DIR, exist_ok=True)
    s = speeds()
    for k, v in kw.items():
        if v and v > 0:
            s[k] = round(0.5 * s.get(k, v) + 0.5 * v, 3)           # smooth over jobs
    with open(SPEED_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)


def video_hours(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return (n / fps) / 3600 if n > 0 else 0.0


VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".dav", ".h264", ".ts", ".m4v")


def find_videos(folder):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(VIDEO_EXT))


def estimate(cam_folders, prepared=False):
    """-> dict with footage hours per camera and the estimated time of each step."""
    sp = speeds()
    cams, total_h, span_h = {}, 0.0, 0.0
    for cid, folder in cam_folders.items():
        files = find_videos(folder)
        h = sum(video_hours(f) for f in files)
        cams[cid] = {"folder": folder, "files": len(files), "hours": round(h, 2)}
        total_h += h
        span_h = max(span_h, h)
    prep = 0.0 if prepared else total_h * sp["prepare_s_per_cam_hour"]
    analyse = min(span_h, 24) * 3600 * 2 / sp["analyse_samples_per_s"]           # 2 samples per second of the day
    report = sp["report_s"]
    disk_gb = total_h * 0.09                                             # ~90 MB per camera-hour at 2 fps
    return {"cameras": cams, "camera_hours": round(total_h, 1), "day_hours": round(span_h, 1),
            "prepare_s": prep, "analyse_s": analyse, "report_s": report,
            "total_s": prep + analyse + report, "disk_gb": round(disk_gb, 1)}


def fmt_duration(s):
    if s is None:
        return "?"
    s = int(s)
    if s < 90:
        return f"{s} s"
    if s < 3600:
        return f"{round(s / 60)} min"
    return f"{s // 3600} h {round((s % 3600) / 60):02d} min"


# ---------------------------------------------------------------- jobs ---
_procs = {}


def _job_path(job_id):
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _alive(pid):
    try:
        import psutil
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def list_jobs(limit=20):
    out = []
    paths = [p for p in glob.glob(os.path.join(JOBS_DIR, "job_*.json")) if not p.endswith(".status.json")]
    for p in sorted(paths, reverse=True)[:limit]:
        try:
            out.append(load(os.path.basename(p)[:-5]))
        except (OSError, ValueError, KeyError):
            continue                                # a damaged job file must not break the page
    return out


def load(job_id):
    with open(_job_path(job_id), encoding="utf-8") as f:
        job = json.load(f)
    if job["state"] == "running":
        proc = _procs.get(job_id)
        code = proc.poll() if proc else (None if _alive(job["pid"]) else -1)
        if code is not None:
            job["state"] = "done" if code == 0 else ("cancelled" if job.get("cancelled") else "failed")
            job["ended"] = time.time()
            _save(job)
    try:
        job["status"] = json.load(open(job["status_file"], encoding="utf-8"))
    except (OSError, ValueError):
        job["status"] = {}
    p = job["status"].get("progress")
    if job["state"] == "done":
        p = 1.0
    job["progress"] = p
    elapsed = (job.get("ended") or time.time()) - job["started"]
    job["elapsed"] = elapsed
    job["eta"] = None
    if job["state"] == "running":
        if p and p > 0.02:
            job["eta"] = elapsed * (1 - p) / p
        elif job.get("estimate_s"):
            job["eta"] = max(0, job["estimate_s"] - elapsed)
    return job


def _save(job):
    keep = {k: v for k, v in job.items() if k not in ("status", "progress", "elapsed", "eta")}
    with open(_job_path(job["id"]), "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=1)


def running():
    for j in list_jobs(5):
        if j["state"] == "running":
            return j
    return None


def log_tail(job, n=12):
    try:
        with open(job["log"], encoding="utf-8", errors="replace") as f:
            lines = f.read().replace("\r", "\n").splitlines()
        return [ln for ln in lines if ln.strip()][-n:]
    except OSError:
        return []


def start(kind, title, args, day=None, estimate_s=None):
    """Start `python -m civetwatch <args>` in the background. Refuses when another job runs."""
    if running():
        raise RuntimeError("Another job is still running. Wait for it or cancel it first.")
    os.makedirs(JOBS_DIR, exist_ok=True)
    job_id = f"job_{datetime.now():%Y%m%d_%H%M%S}"
    log = os.path.join(JOBS_DIR, job_id + ".log")
    status_file = os.path.join(JOBS_DIR, job_id + ".status.json")
    env = {**os.environ, "CIVETWATCH_STATUS": status_file, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    cmd = [sys.executable, "-m", "civetwatch", "--profile", "laptop", *args]
    with open(log, "w", encoding="utf-8") as out:
        out.write("$ " + " ".join(cmd) + "\n")
        out.flush()
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT, env=env,
                                creationflags=flags)
    _procs[job_id] = proc
    job = {"id": job_id, "kind": kind, "title": title, "day": day, "args": args, "pid": proc.pid,
           "started": time.time(), "state": "running", "log": log, "status_file": status_file,
           "estimate_s": estimate_s}
    _save(job)
    return job


def cancel(job_id):
    job = load(job_id)
    if job["state"] != "running":
        return job
    job["cancelled"] = True
    _save(job)
    try:
        import psutil
        p = psutil.Process(job["pid"])
        for c in p.children(recursive=True):            # FFmpeg runs as a child process
            c.kill()
        p.kill()
    except Exception:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(job["pid"])], capture_output=True)
    time.sleep(0.5)
    return load(job_id)
