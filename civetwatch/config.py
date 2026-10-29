"""Load settings (config/default.yaml) and day manifests (which video is which camera)."""
import copy
import glob
import os
import re
from datetime import datetime

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "default.yaml")


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and os.path.abspath(path) != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f))
    return cfg


def resolve(path, base=PROJECT_ROOT):
    """Paths in configs are relative to the project root (or the manifest's folder)."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


_TS_PATTERNS = [
    # 20260720_150000, 20260720-150000, 20260720150000
    (re.compile(r"(\d{4})(\d{2})(\d{2})[_\-T ]?(\d{2})(\d{2})(\d{2})"), None),
    # 2026-07-20_15-00-00, 2026-07-20 15.00.00
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[_\-T ](\d{2})[\-:.](\d{2})[\-:.](\d{2})"), None),
    # 2026-07-20_15 (hour files)
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[_\-T ](\d{2})(?!\d)"), None),
]


def timestamp_from_filename(path):
    name = os.path.basename(path)
    for pattern, _ in _TS_PATTERNS:
        m = pattern.search(name)
        if m:
            parts = [int(x) for x in m.groups()] + [0] * (6 - len(m.groups()))
            try:
                return datetime(*parts)
            except ValueError:
                continue
    return None


def parse_time(value):
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot read time '{value}'. Use YYYY-MM-DD HH:MM:SS")


def load_manifest(path):
    """A manifest lists the 4 cameras of one observation day.

    cameras:
      - id: cam1
        room: R1
        zones: zones/cam1.json
        clock_offset_s: 0          # NVR clock correction for this camera
        files:                     # a list, or a glob string
          - {path: videos/cam1_20260720_150000.mp4, start: "2026-07-20 15:00:00"}
    When 'start' is missing it is read from the file name; when that fails the
    file is assumed to follow the previous one directly.
    """
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        man = yaml.safe_load(f)
    cams = []
    for cam in man["cameras"]:
        files = cam.get("files", [])
        if isinstance(files, str):
            files = [{"path": p} for p in sorted(glob.glob(resolve(files, base)))]
        entries = []
        for item in files:
            item = {"path": item} if isinstance(item, str) else dict(item)
            item["path"] = resolve(item["path"], base)
            if item.get("start"):
                item["start"] = parse_time(item["start"])
            else:
                item["start"] = timestamp_from_filename(item["path"])
            entries.append(item)
        cams.append({
            "id": cam["id"],
            "room": cam.get("room", cam["id"]),
            "zones": resolve(cam.get("zones"), base) if cam.get("zones") else None,
            "background": resolve(cam.get("background"), base) if cam.get("background") else None,
            "clock_offset_s": float(cam.get("clock_offset_s", 0)),
            "files": entries,
        })
    out = {"cameras": cams, "base": base}
    for key in ("start", "end"):
        if man.get(key):
            out[key] = parse_time(man[key])
    out["date"] = man.get("date")
    return out
