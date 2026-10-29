"""Teaching the AI with the zoo's own videos (slide 17):
1 collect day + night frames -> 2 label in CVAT -> 3 train -> 4 test -> 5 compare.
"""
import json
import os
import shutil
import time
from datetime import timedelta

import cv2
import pandas as pd
import yaml

from . import ethogram as E
from .config import PROJECT_ROOT, load_manifest
from .video import CameraReader, colorfulness


def extract_frames(manifest_path, out_dir, every_s=10.0, max_per_cam=None):
    """Save frames from every camera for labelling in CVAT. Night frames are
    kept twice as often because most activity happens in infrared."""
    man = load_manifest(manifest_path)
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for cam in man["cameras"]:
        reader = CameraReader(cam)
        t0, t1 = reader.coverage()
        if t0 is None:
            continue
        t, n = t0, 0
        while t < t1 and (max_per_cam is None or n < max_per_cam):
            frame, _ = reader.read_at(t)
            night = frame is not None and colorfulness(frame) < 8
            if frame is not None:
                cv2.imwrite(os.path.join(out_dir, f"{cam['id']}_{t:%Y%m%d_%H%M%S}.jpg"), frame)
                n += 1
            t += timedelta(seconds=every_s / (2 if night else 1))
        reader.close()
        total += n
        print(f"  {cam['id']}: {n} frames")
    return total


def autolabel_dataset(run_dir, manifest_path, out_dir, every_s=6.0, negatives=0.25, val_frac=0.2,
                      min_conf=0.0, min_motion=0.0, min_size=0, drop=None, seed=0, cfg=None):
    """Bootstrap a YOLO detector dataset from a run made with the motion
    detector: its civet boxes become pseudo-labels (class 0 = civet), frames
    where no camera saw anything become negatives. `drop` = list of
    (camera, start, end) windows judged wrong on the contact sheets.
    Train/val are split by time (the last `val_frac` of each camera).
    `min_motion` keeps only boxes whose pixels changed since the previous
    sample - a moving civet, not a stale patch of background."""
    import numpy as np
    from .video import Preprocessor

    rng = np.random.default_rng(seed)
    frames = pd.read_csv(os.path.join(run_dir, "frames.csv"), parse_dates=["t"])
    man = load_manifest(manifest_path)
    cfg_light = cfg or {"sampling": {"process_width": 960, "fps": 2},
                 "preprocess": {"night_colorfulness_max": 8, "clahe_clip": 3.0, "clahe_grid": 8, "denoise": "none",
                                "denoise_strength": 7, "nlm_template": 5, "nlm_search": 11, "blur_min": 0,
                                "dark_max": 0, "frozen_diff_max": 0, "frozen_min_s": 1e9}}
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    picked = []
    for cam in man["cameras"]:
        cid = cam["id"]
        mine = frames[frames["camera"] == cid].copy()
        size = (mine["x2"] - mine["x1"]).combine(mine["y2"] - mine["y1"], max)
        mine = mine[(mine["det_conf"] >= min_conf) & (mine["motion"] >= min_motion) & (size >= min_size)]
        for c, a, b in drop or []:
            if c == cid:
                mine = mine[~mine["t"].between(pd.Timestamp(a), pd.Timestamp(b))]
        mine = mine[mine["t"].dt.floor(f"{int(every_s * 1000)}ms") == mine["t"]]
        reader = CameraReader(cam)
        t0, t1 = reader.coverage()
        empty = frames[(frames["camera"].isna() | (frames["camera"] == "")) & frames["t"].between(t0, t1)]
        n_neg = int(len(mine) * negatives)
        neg = empty.iloc[rng.choice(len(empty), size=min(n_neg, len(empty)), replace=False)] if len(empty) else empty
        rows = [(r.t, r) for r in mine.itertuples()] + [(r.t, None) for r in neg.itertuples()]
        rows.sort(key=lambda x: x[0])
        split_t = t0 + (t1 - t0) * (1 - val_frac)
        pre = Preprocessor(cfg_light)
        for t, r in rows:
            frame, _ = reader.read_at(t.to_pydatetime())
            if frame is None:
                continue
            img, _, _ = pre(cid, frame)          # same size + night contrast as at run time
            h, w = img.shape[:2]
            split = "val" if t >= split_t else "train"
            name = f"{cid}_{t:%Y%m%d_%H%M%S_%f}"[:-3]
            cv2.imwrite(os.path.join(out_dir, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(out_dir, "labels", split, name + ".txt"), "w") as f:
                if r is not None:
                    cx, cy = (r.x1 + r.x2) / 2 / w, (r.y1 + r.y2) / 2 / h
                    f.write(f"0 {cx:.6f} {cy:.6f} {(r.x2 - r.x1) / w:.6f} {(r.y2 - r.y1) / h:.6f}\n")
            picked.append({"camera": cid, "t": t, "split": split, "image": name, "positive": r is not None,
                           **({"x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2} if r is not None else {})})
        reader.close()
    index = pd.DataFrame(picked)
    index.to_csv(os.path.join(out_dir, "index.csv"), index=False)
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": os.path.abspath(out_dir), "train": "images/train", "val": "images/val",
                        "names": {0: "civet"}}, f, sort_keys=False)
    return index


def contact_sheets(dataset_dir, out_dir, per_sheet=48, cols=8, tile=200):
    """Crops of every pseudo-label, 48 per page, captioned with camera and time,
    for checking the labels by eye before training."""
    import numpy as np
    index = pd.read_csv(os.path.join(dataset_dir, "index.csv"), parse_dates=["t"])
    pos = index[index["positive"]].reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    sheets = []
    for s in range(0, len(pos), per_sheet):
        tiles = []
        for _, r in pos.iloc[s:s + per_sheet].iterrows():
            img = cv2.imread(os.path.join(dataset_dir, "images", r["split"], r["image"] + ".jpg"))
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            pad = int(max(x2 - x1, y2 - y1) * 0.6)
            crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
            cv2.rectangle(crop, (x1 - max(0, x1 - pad), y1 - max(0, y1 - pad)),
                          (x2 - max(0, x1 - pad), y2 - max(0, y1 - pad)), (0, 190, 255), 2)
            crop = cv2.resize(crop, (tile, tile))
            cv2.rectangle(crop, (0, tile - 18), (tile, tile), (0, 0, 0), -1)
            cv2.putText(crop, f"{r['camera']} {r['t']:%H:%M:%S}", (4, tile - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
        while len(tiles) % cols:
            tiles.append(np.zeros((tile, tile, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        path = os.path.join(out_dir, f"sheet_{s // per_sheet:02d}.jpg")
        cv2.imwrite(path, grid)
        sheets.append(path)
    return sheets


def write_dataset_templates(out_dir=os.path.join(PROJECT_ROOT, "datasets")):
    """YOLO dataset yaml files for the detector and the pose model."""
    os.makedirs(out_dir, exist_ok=True)
    det = {"path": "civet_det", "train": "images/train", "val": "images/val",
           "names": {0: "civet", 1: "food_tray", 2: "water_tray"}}
    pose = {"path": "civet_pose", "train": "images/train", "val": "images/val",
            "kpt_shape": [len(E.KEYPOINTS), 3], "flip_idx": E.FLIP_IDX,
            "names": {0: "civet"}, "kpt_names": {0: E.KEYPOINTS}}
    paths = []
    for name, data in (("civet_det.yaml", det), ("civet_pose.yaml", pose)):
        p = os.path.join(out_dir, name)
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        paths.append(p)
    return paths


def _train_yolo(base, data, epochs, imgsz, out_name, device=None, resume=False, **kw):
    """Fine-tune a YOLO model. Uses the resources profile (batch size, data
    workers, a rest between epochs, pause when the GPU is hot). resume=True
    continues an interrupted run from its last saved epoch."""
    from ultralytics import YOLO
    from .resources import Throttle, settings, torch_threads
    torch_threads()
    r = settings()
    project = os.path.join(PROJECT_ROOT, "runs", "training")
    last = os.path.join(project, out_name, "weights", "last.pt")
    throttle = Throttle(check_every_s=0)

    def rest(trainer):
        if r["train_epoch_pause_s"]:
            time.sleep(r["train_epoch_pause_s"])
        throttle.wait_if_hot()

    if resume:
        if not os.path.exists(last):
            raise SystemExit(f"Nothing to resume: {last} does not exist")
        model = YOLO(last)
        model.add_callback("on_train_epoch_end", rest)
        res = model.train(resume=True)
    else:
        model = YOLO(base)
        model.add_callback("on_train_epoch_end", rest)
        kw.setdefault("batch", r["train_batch"])
        kw.setdefault("workers", r["train_workers"])
        res = model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, project=project,
                          name=out_name, exist_ok=True, **kw)
    best = os.path.join(str(res.save_dir), "weights", "best.pt")
    dst = os.path.join(PROJECT_ROOT, "models", f"{out_name}.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(best, dst)
    return dst


def train_detector(data, base="yolo26n.pt", epochs=100, imgsz=960, device=None, resume=False, **kw):
    return _train_yolo(base, data, epochs, imgsz, "civet_det", device, resume=resume, **kw)


def train_pose(data, base="yolo26n-pose.pt", epochs=150, imgsz=640, device=None, resume=False):
    return _train_yolo(base, data, epochs, imgsz, "civet_pose", device, resume=resume)


def export_labels(run_dir, out_csv=None, only_reviewed=False, context_s=10):
    """frames.csv + keeper review decisions -> frames with a `true_label`
    column for GRU training. Corrected events overwrite the AI label; events
    marked 'not this behavior' become active_other."""
    frames = pd.read_csv(os.path.join(run_dir, "frames.csv"), parse_dates=["t"], keep_default_na=False)
    events = pd.read_csv(os.path.join(run_dir, "events.csv"), parse_dates=["start", "stop"])
    review_path = os.path.join(run_dir, "review.json")
    review = json.load(open(review_path, encoding="utf-8")) if os.path.exists(review_path) else {}
    frames["true_label"] = frames["voted"]
    keep = pd.Series(False, index=frames.index)
    for key, decision in review.items():
        ev = events.iloc[int(key)]
        span = (frames["t"] >= ev["start"]) & (frames["t"] < ev["stop"])
        keep |= (frames["t"] >= ev["start"] - timedelta(seconds=context_s)) & \
                (frames["t"] < ev["stop"] + timedelta(seconds=context_s))
        if decision["status"] == "corrected":
            frames.loc[span, "true_label"] = decision["label"]
        elif decision["status"] == "rejected":
            frames.loc[span, "true_label"] = E.ACTIVE_OTHER
        else:                                   # confirmed
            frames.loc[span, "true_label"] = ev["behavior"]
    if only_reviewed:
        frames = frames[keep]
    out_csv = out_csv or os.path.join(run_dir, "frames_labelled.csv")
    frames.to_csv(out_csv, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    return out_csv, len(review)
