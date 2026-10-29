"""Success check without a keeper sheet: people check random moments themselves.

`make_spotcheck` picks random moments of an analysed day and saves, for each,
all cameras at t-1 s, t, t+1 s - *without* CivetWatch's answer, so the
checker is not influenced. The checker answers (in the dashboard, or in
answers.json): which room is the civet in, and what is it doing.
`score` then compares the answers with CivetWatch.
"""
import json
import os
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd

from . import ethogram as E
from .config import load_manifest
from .video import CameraReader, Preprocessor

CANT_TELL = "cant_tell"
NO_CIVET = "none"
ANSWER_LABELS = E.INACTIVE_LABELS + [E.WALKING, E.ACTIVE_OTHER] + E.FREQUENCY_LABELS + [CANT_TELL]


def _dir(run_dir):
    return os.path.join(run_dir, "spotcheck")


def _prediction(frames, t):
    row = frames.iloc[(frames["t"] - t).abs().argmin()]
    seen = bool(row.get("camera")) or row["label"] == E.SLEEP_IN_BOX
    return {"room": row["room"] if seen and row["room"] else NO_CIVET,
            "label": row["voted"], "state": row["state"], "camera": row.get("camera") or ""}


def make_spotcheck(run_dir, manifest_path, cfg, n=120, seen_share=0.6, seed=0, tile=(320, 180)):
    """Half (seen_share) of the moments are drawn from times CivetWatch saw the
    civet, the rest from the whole day, at least 2 minutes apart."""
    frames = pd.read_csv(os.path.join(run_dir, "frames.csv"), parse_dates=["t"], low_memory=False)
    frames["camera"] = frames["camera"].fillna("")
    frames["room"] = frames["room"].fillna("")
    rng = np.random.default_rng(seed)
    seen_t = frames.loc[frames["camera"] != "", "t"].to_numpy()
    all_t = frames["t"].to_numpy()
    picked = []
    for pool, k in ((seen_t, int(n * seen_share)), (all_t, n - int(n * seen_share))):
        tries = 0
        while k > 0 and tries < 50 * n and len(pool):
            t = pd.Timestamp(pool[rng.integers(len(pool))])
            tries += 1
            if all(abs((t - p).total_seconds()) >= 120 for p in picked):
                picked.append(t)
                k -= 1
    picked.sort()

    man = load_manifest(manifest_path)
    cams = man["cameras"]
    light_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "denoise": "none"}}
    pre = Preprocessor(light_cfg)
    out = _dir(run_dir)
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    readers = {c["id"]: CameraReader(c) for c in cams}
    items = []
    for i, t in enumerate(picked):
        rows = []
        for cam in cams:
            tiles = []
            for dt in (-1, 0, 1):
                frame, _ = readers[cam["id"]].read_at((t + timedelta(seconds=dt)).to_pydatetime())
                if frame is None:
                    img = np.zeros((tile[1], tile[0], 3), np.uint8)
                else:
                    img = cv2.resize(pre(cam["id"], frame)[0], tile)
                label = f"{cam['room']}  {dt:+d}s" if dt else f"{cam['room']}  t"
                cv2.rectangle(img, (0, 0), (110, 22), (30, 30, 30), -1)
                cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(img)
            rows.append(np.hstack(tiles))
        name = f"{i:03d}.jpg"
        cv2.imwrite(os.path.join(out, "img", name), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append({"id": i, "t": t.isoformat(), "image": f"spotcheck/img/{name}", "pred": _prediction(frames, t)})
    for r in readers.values():
        r.close()
    with open(os.path.join(out, "items.json"), "w", encoding="utf-8") as f:
        json.dump({"rooms": [c["room"] for c in cams], "items": items}, f, indent=1, default=str)
    return items


def load(run_dir, answers_name="answers.json"):
    d = _dir(run_dir)
    with open(os.path.join(d, "items.json"), encoding="utf-8") as f:
        items = json.load(f)
    p = os.path.join(d, answers_name)
    answers = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return items, answers


def save_answer(run_dir, item_id, room, label, checker=""):
    items, answers = load(run_dir)
    answers[str(item_id)] = {"room": room, "label": label, "checker": checker}
    with open(os.path.join(_dir(run_dir), "answers.json"), "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=1)


def _state(label):
    return E.state_of(label) or ""


def score(run_dir, answers_name="answers.json"):
    """-> (summary dict, per-item DataFrame, behavior confusion DataFrame)."""
    items, answers = load(run_dir, answers_name)
    rows = []
    for it in items["items"]:
        a = answers.get(str(it["id"]))
        if not a:
            continue
        p = it["pred"]
        rows.append({"id": it["id"], "t": it["t"], "true_room": a["room"], "pred_room": p["room"],
                     "true_label": a["label"], "pred_label": p["label"], "pred_state": p["state"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"checked": 0}, df, pd.DataFrame()
    present = df["true_room"] != NO_CIVET
    summary = {"checked": len(df)}
    # 1 detection: is the civet in view, and in which room
    summary["found when in view"] = (df.loc[present, "pred_room"] != NO_CIVET).mean() if present.any() else np.nan
    summary["no false civet when out of view"] = (df.loc[~present, "pred_room"] == NO_CIVET).mean() if (~present).any() else np.nan
    both = present & (df["pred_room"] != NO_CIVET)
    summary["right room"] = (df.loc[both, "true_room"] == df.loc[both, "pred_room"]).mean() if both.any() else np.nan
    # 2 state and behavior, where the checker could tell
    judged = both & (df["true_label"] != CANT_TELL)
    st = df.loc[judged]
    summary["Inactive/Active right"] = (st["true_label"].map(_state) == st["pred_label"].map(_state)).mean() if len(st) else np.nan
    summary["behavior right"] = (st["true_label"] == st["pred_label"]).mean() if len(st) else np.nan
    confusion = pd.crosstab(st["true_label"].map(lambda l: E.DISPLAY.get(l, l)),
                            st["pred_label"].map(lambda l: E.DISPLAY.get(l, l)),
                            rownames=["checker said"], colnames=["CivetWatch said"]) if len(st) else pd.DataFrame()
    return summary, df, confusion
