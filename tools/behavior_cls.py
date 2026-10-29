"""Behavior classifier from public MammalNet clips (cats and other small carnivores),
made to look like the zoo cameras, tested on CivetWatch's short stops.

  python tools/behavior_cls.py prepare    # clips -> animal crops (train / val by MammalNet subset)
  python tools/behavior_cls.py train      # YOLO26n-cls, laptop resource profile
  python tools/behavior_cls.py test       # score the 5 Apr 2026 short stops, compare with checked verdicts
"""
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from civetwatch import resources                                      # noqa: E402
from civetwatch.config import load_config                              # noqa: E402

CLIPS = os.path.join(ROOT, "datasets", "mammalnet", "clips")
OUT = os.path.join(ROOT, "datasets", "behavior_cls")
ANIMALS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}                     # COCO bird..giraffe (cat 15, dog 16, bear 21)
CLASSES = ["grooming", "other", "toileting"]


def zoo_look(crop, rng):
    """Make a public video crop look like a zoo CCTV crop: grey, contrast-boosted
    (like the night pipeline), small, a bit blurred and noisy, any rotation."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if rng.random() < 0.7:
        g = cv2.createCLAHE(3.0, (8, 8)).apply(g)
    side = int(rng.integers(90, 200))                                  # civet crops are ~100-250 px
    g = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    g = np.clip(g.astype(np.float32) + rng.normal(0, rng.uniform(0, 8), g.shape), 0, 255).astype(np.uint8)
    angle = float(rng.uniform(0, 360))
    M = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    g = cv2.warpAffine(g, M, (side, side), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        g = cv2.flip(g, 1)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def square_crop(img, box, pad=0.25):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * (0.5 + pad)
    a, b = int(max(0, cy - half)), int(min(h, cy + half))
    c, d = int(max(0, cx - half)), int(min(w, cx + half))
    return img[a:b, c:d]


def prepare(frames_per_clip=10, copies=2):
    from ultralytics import YOLO
    resources.torch_threads()
    det = YOLO(os.path.join(ROOT, "models", "yolo26n.pt"))
    meta = {j["vid"] + "_" + str(j["k"]): j for j in json.load(open(os.path.join(CLIPS, "clips.json")))}
    rng = np.random.default_rng(0)
    counts = {}
    throttle = resources.Throttle(log=None)
    for path in sorted(glob.glob(os.path.join(CLIPS, "*.mp4"))):
        label, key = os.path.basename(path)[:-4].split("__")
        split = "train" if meta.get(key, {}).get("subset") == "training" else "val"
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for q in range(frames_per_clip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((q + 0.5) * n / frames_per_clip))
            ok, f = cap.read()
            if not ok:
                continue
            r = det.predict(f, conf=0.25, verbose=False)[0]
            boxes = [(float(c), b) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist())
                     if k in ANIMALS]
            if not boxes:
                continue
            crop = square_crop(f, max(boxes)[1])
            if crop.size == 0 or min(crop.shape[:2]) < 40:
                continue
            for c in range(copies if split == "train" else 1):
                d = os.path.join(OUT, split, label)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"{key}_{q}_{c}.jpg"), zoo_look(crop, rng))
                counts[(split, label)] = counts.get((split, label), 0) + 1
            throttle.tick()
        cap.release()
    print({f"{s}/{l}": v for (s, l), v in sorted(counts.items())})


def train(epochs=40):
    from civetwatch.training import _train_yolo
    print(_train_yolo("yolo26n-cls.pt", OUT, epochs, 160, "civet_behavior_cls", device=0, batch=32, patience=12))


def stop_crops(cfg, run, n=5):
    """Crops of the civet across each short stop of a run (same processing as the pipeline)."""
    import pandas as pd
    from civetwatch.config import load_manifest
    from civetwatch.video import CameraReader, Preprocessor
    ev = pd.read_csv(os.path.join(run, "events.csv"), parse_dates=["start", "stop"])
    st = ev[ev["category"] == "stop"].reset_index()
    fr = pd.read_csv(os.path.join(run, "frames.csv"), parse_dates=["t"], low_memory=False).set_index("t")
    cams = {c["id"]: c for c in load_manifest(os.path.join(ROOT, "data", "2026-04-05", "manifest.yaml"))["cameras"]}
    light = {**cfg, "preprocess": {**cfg["preprocess"], "denoise": "none"}}
    for _, s in st.iterrows():
        rd, pre = CameraReader(cams[s.camera]), Preprocessor(light)
        crops = []
        for q in range(n):
            t = s.start + (s.stop - s.start) * ((q + 0.5) / n)
            row = fr.iloc[fr.index.get_indexer([t], method="nearest")[0]]
            if row.camera != s.camera:
                continue
            f, _ = rd.read_at(t.to_pydatetime())
            if f is None:
                continue
            img = pre(s.camera, f)[0]
            crop = square_crop(img, (row.x1, row.y1, row.x2, row.y2))
            if crop.size:
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crops.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        rd.close()
        yield s, crops


def test(run=os.path.join(ROOT, "runs", "2026-04-05"), verdicts=None):
    import pandas as pd
    from ultralytics import YOLO
    resources.torch_threads()
    cfg = load_config(os.path.join(ROOT, "data", "2026-04-05", "config.yaml"))
    model = YOLO(os.path.join(ROOT, "models", "civet_behavior_cls.pt"))
    names = model.names
    rows = []
    for s, crops in stop_crops(cfg, run):
        if not crops:
            continue
        res = model.predict(crops, imgsz=160, verbose=False)
        p = np.mean([r.probs.data.cpu().numpy() for r in res], axis=0)
        rows.append({"start": f"{s.start:%Y-%m-%d %H:%M:%S}", "camera": s.camera, "room": s.room,
                     "behavior": s.behavior, **{names[i]: float(p[i]) for i in range(len(p))}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run, "stops_behavior_cls.csv"), index=False)
    if verdicts and os.path.exists(verdicts):
        v = pd.read_csv(verdicts, parse_dates=["start"])
        v["start"] = v["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df = df.merge(v[["start", "verdict", "kind"]], on="start", how="left")
        df["verdict"] = df["verdict"].fillna("unjudged")
        judged = df[df.verdict != "unjudged"]
        for cls in ("toileting", "grooming"):
            o = judged.sort_values(cls, ascending=False).reset_index(drop=True)
            ranks = [i + 1 for i, r in o.iterrows() if r.verdict == "real" and r.kind == cls]
            print(f"{cls}: rank of the real {cls} stops among {len(o)} judged stops: {ranks}")
            print("  top 10:", [f"{r.start[11:19]} {r.verdict}/{r.kind}" for _, r in o.head(10).iterrows()])
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    resources.apply(load_config())
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"prepare": prepare, "train": train, "test": lambda: test(verdicts=sys.argv[2] if len(sys.argv) > 2 else None)}[cmd]()
