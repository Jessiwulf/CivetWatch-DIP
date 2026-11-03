"""Train the stop classifier on YOUR answers from the Behavior check page.

    python -m civetwatch train-behavior --run runs/2026-04-05 --manifest data/2026-04-05/manifest.yaml

Each checked stop (short stops and "busy while staying in place") gives up to
30 crops of the civet from the prepared video. Classes: urination, defecation,
territorial_marking, grooming, nothing. Stops - not single frames - are split into
train and test, so the test score is honest: it is measured on stops the model
never saw. The model is a YOLO26n-cls, started from the MammalNet model.

The result is models/civet_stop_cls.pt. The next report uses it to put a guess
on every stop and to sort the check list (likely behaviors first). It only
counts stops by itself when stops.auto_accept is set.
"""
import json
import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_manifest

CLASSES = ["defecation", "grooming", "nothing", "territorial_marking", "urination"]
ANSWER_CLASS = {"urination": "urination", "defecation": "defecation", "toileting": "urination",
                "territorial_marking": "territorial_marking", "grooming": "grooming"}
RARE = ["urination", "defecation", "territorial_marking", "grooming"]


def checked_stops(run_dir):
    """Stops with a person's answer -> DataFrame with a 'cls' column."""
    ev = pd.read_csv(os.path.join(run_dir, "events.csv"), parse_dates=["start", "stop"], keep_default_na=False)
    path = os.path.join(run_dir, "review.json")
    review = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    from .pipeline import _is_stop_decision
    by_event = {(v["start"][:19], v["camera"]): v for v in review.values() if v.get("start") and _is_stop_decision(v)}
    rows = []
    for _, e in ev[ev["category"] == "stop"].iterrows():
        dec = by_event.get((f"{e['start']:%Y-%m-%d %H:%M:%S}", e["camera"]))
        if not dec or dec.get("label") == "cant_see":
            continue
        cls = ANSWER_CLASS.get(dec.get("label") or "", "nothing") if dec["status"] != "confirmed" else None
        if dec["status"] == "confirmed":
            cls = {"possible_toileting": "urination", "possible_marking": "territorial_marking",
                   "possible_grooming": "grooming"}.get(e["behavior"], "nothing")
        rows.append({**e.to_dict(), "cls": cls})
    return pd.DataFrame(rows)


def build_dataset(run_dir, manifest, cfg, out_dir, val_frac=0.25, seed=0, progress=True):
    from .stops import stop_crops
    from .video import CameraReader
    stops = checked_stops(run_dir)
    if not len(stops):
        raise SystemExit("No checked stops yet: answer some in the dashboard (Behavior check) first")
    counts = stops["cls"].value_counts().to_dict()
    if progress:
        print("Checked stops per class: " + ", ".join(f"{c} {counts.get(c, 0)}" for c in CLASSES))
    for c in RARE:
        if counts.get(c, 0) < 3:
            print(f"  [warn] only {counts.get(c, 0)} '{c}' stop(s): the model cannot learn it well; it is still kept")
    # boxes: events.csv has no box, take the median box of frames.csv inside each stop
    fr = pd.read_csv(os.path.join(run_dir, "frames.csv"), usecols=["t", "camera", "x1", "y1", "x2", "y2"],
                     parse_dates=["t"], low_memory=False)
    fr = fr[fr["camera"].fillna("") != ""].set_index("t").sort_index()
    rng = random.Random(seed)
    split = {}
    for c, grp in stops.groupby("cls"):
        idx = list(grp.index)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac)) if len(idx) >= 3 else 0
        n_val = max(1, n_val) if len(idx) >= 3 and val_frac > 0 else n_val
        for k, i in enumerate(idx):
            split[i] = "val" if k < n_val else "train"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for part in ("train", "val"):
        for c in CLASSES:
            os.makedirs(os.path.join(out_dir, part, c), exist_ok=True)
    cams = {c["id"]: c for c in load_manifest(manifest)["cameras"]}
    readers = {}
    info = []
    stops = stops.sort_values(["camera", "start"])
    for n, (i, e) in enumerate(stops.iterrows()):
        cam = cams.get(e["camera"])
        if cam is None:
            continue
        rd = readers.setdefault(e["camera"], CameraReader(cam))
        seg = fr.loc[e["start"]:e["stop"]]
        seg = seg[seg["camera"] == e["camera"]]
        if not len(seg):
            continue
        box = {k: float(seg[k].median()) for k in ("x1", "y1", "x2", "y2")}
        k = int(np.clip(float(e["duration_s"]) * 2, 4, 30))
        crops = stop_crops(cam, {**e.to_dict(), **box}, cfg, n=k, reader=rd)
        key = f"{e['camera']}_{e['start']:%H%M%S}"
        for q, c in enumerate(crops):
            cv2.imwrite(os.path.join(out_dir, split[i], e["cls"], f"{key}_{q:02d}.jpg"), c)
        info.append({"key": key, "cls": e["cls"], "split": split[i], "crops": len(crops),
                     "start": str(e["start"]), "camera": e["camera"], "kind": e.get("kind", "")})
        if progress and n % 20 == 0:
            print(f"  crops: {n + 1}/{len(stops)} stops", flush=True)
            from .jobs import status
            status("Cutting civet pictures from the checked stops", 0.4 * n / len(stops))
    for rd in readers.values():
        rd.close()
    # balance: copy the rare classes' training crops so each has at least 1/3 as many as 'nothing'
    tr = os.path.join(out_dir, "train")
    n_nothing = len(os.listdir(os.path.join(tr, "nothing")))
    for c in RARE:
        files = os.listdir(os.path.join(tr, c))
        if not files:
            continue
        reps = int(min(8, max(1, np.ceil(n_nothing / 3 / len(files)))))
        for r in range(1, reps):
            for f in files:
                shutil.copy(os.path.join(tr, c, f), os.path.join(tr, c, f"r{r}_{f}"))
    if val_frac == 0:                                   # final model: ultralytics still needs a val folder
        for c in CLASSES:
            for f in os.listdir(os.path.join(tr, c))[:20]:
                shutil.copy(os.path.join(tr, c, f), os.path.join(out_dir, "val", c, f))
    pd.DataFrame(info).to_csv(os.path.join(out_dir, "stops.csv"), index=False)
    return pd.DataFrame(info)


def evaluate(model_path, data_dir, info):
    """Per-stop answer (mean over its crops) on the test stops only."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    rows = []
    for s in info[info["split"] == "val"].to_dict("records"):
        folder = os.path.join(data_dir, "val", s["cls"])
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.startswith(s["key"] + "_")]
        if not files:
            continue
        res = model.predict(files, imgsz=160, verbose=False)
        mean = np.mean([r.probs.data.cpu().numpy() for r in res], axis=0)
        rows.append({**s, "guess": model.names[int(mean.argmax())], "prob": float(mean.max())})
    df = pd.DataFrame(rows)
    if not len(df):
        return df, {}
    out = {"test_stops": len(df), "accuracy": float((df["cls"] == df["guess"]).mean())}
    for c in CLASSES:
        tp = int(((df["cls"] == c) & (df["guess"] == c)).sum())
        out[f"{c}: found"] = f"{tp} of {int((df['cls'] == c).sum())} test stops"
        out[f"{c}: false alarms"] = int(((df["cls"] != c) & (df["guess"] == c)).sum())
    return df, out


def train_behavior(run_dir, manifest, cfg, epochs=30, use_all=False, progress=True):
    from .jobs import phase
    from .training import _train_yolo
    data_dir = os.path.join(PROJECT_ROOT, "datasets", "stop_cls")
    phase("Cutting civet pictures from the checked stops", 0.0, 0.3)
    info = build_dataset(run_dir, manifest, cfg, data_dir, val_frac=0 if use_all else 0.25, progress=progress)
    base = os.path.join(PROJECT_ROOT, "models", "civet_behavior_cls.pt")
    if not os.path.exists(base):
        base = os.path.join(PROJECT_ROOT, "models", "yolo26n-cls.pt")
    import torch
    phase("Training the behavior model", 0.3, 0.95)
    from .jobs import status

    path = _train_yolo(base, data_dir, epochs, 160, "civet_stop_cls", device=0 if torch.cuda.is_available() else "cpu",
                       batch=32, patience=10, plots=False,
                       on_epoch=lambda ep: status(f"Training the behavior model (round {ep} of {epochs})", ep / epochs))
    phase("Testing on stops the model never saw", 0.95, 1.0)
    result = {"model": os.path.relpath(path, PROJECT_ROOT), "train_stops": int((info["split"] == "train").sum()),
              "classes": info["cls"].value_counts().to_dict()}
    if not use_all:
        df, scores = evaluate(path, data_dir, info)
        result.update(scores)
        if len(df):
            df.to_csv(os.path.join(run_dir, "behavior_model_test.csv"), index=False)
    with open(os.path.join(run_dir, "behavior_model.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=str)
    if progress:
        print("\nBehavior model saved: " + result["model"])
        for k, v in result.items():
            if k not in ("model",):
                print(f"  {k}: {v}")
        print("Rebuild the report to use it (dashboard > Rebuild report).")
    return result
