"""Short-stop finder + vision-language pre-filter for the rare behaviors.

Toileting, scent marking and grooming all happen while the civet stands still
for a few seconds between walking: it arrives, stops 3-60 s, leaves. Step 1
finds every such stop (away from the food / water). Step 2 shows each stop to
a small vision-language model (Qwen2.5-VL-3B, 4-bit, runs on a 4 GB GPU) as a
2x2 grid of frames and asks what the civet is doing.

Tested on 5 Apr 2026 (70 stops, checked by eye): the model found 3 of 4
toileting stops with ~11 false alarms, and no grooming. So its answer is only
a pre-filter: stops it calls toileting / marking / grooming are flagged
"please check" with a clip; the rest are listed but not flagged. None of them
are counted in the observation sheet until a person confirms them.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

from . import ethogram as E
from .resources import Throttle

POSSIBLE = {"A": E.URINATION, "B": E.MARKING, "C": E.GROOMING}      # A = toileting (urine or feces)
VLM_LABEL = {"A": "toileting", "B": "marking", "C": "grooming", "D": "drinking/eating",
             "E": "standing/sniffing", "F": "no animal"}
PROMPT = (
    "These 4 frames (left to right, top to bottom, a few seconds apart) are from a zoo CCTV camera looking down "
    "at a large-spotted civet (a cat-like animal with a long ringed tail). Night frames are black and white infrared. "
    "What is the civet doing in these frames? Choose one:\n"
    "A) squatting or crouching with the hindquarters lowered and the tail lifted up (urinating or defecating)\n"
    "B) rubbing or pressing its rear end against an object or wall (scent marking)\n"
    "C) licking or scratching its own body, head turned back to its body (grooming)\n"
    "D) drinking or eating from a bowl\n"
    "E) standing, sniffing, looking around or walking, with the tail hanging down\n"
    "F) no animal is visible\n"
    "Answer with the letter first, then a few words why.")


# ------------------------------------------------------------- step 1 ---
def mark_doorway(stops, frames, cams, cfg):
    """Stops in a doorway with a camera change within a few seconds are the
    civet passing between rooms, not a behavior (5 Apr 2026: removed 23 of 127
    flagged stops, none of the real toileting / grooming ones)."""
    from .zones import ZoneMap
    c = cfg["stops"]
    if not len(stops):
        return stops
    zms = {}
    for cam in cams:
        z = ZoneMap(cam.get("zones"))
        z.set_size(cfg["sampling"]["process_width"], int(round(cfg["sampling"]["process_width"] * 9 / 16)))
        zms[cam["id"]] = z
    f = frames[["t", "camera"]].copy()
    f["camera"] = f["camera"].fillna("")
    f = f.set_index("t").sort_index()
    pad = pd.Timedelta(seconds=c.get("doorway_switch_s", 4))
    flags = []
    for s in stops.itertuples():
        z = zms.get(s.camera)
        door = z.box_overlap((s.x1, s.y1, s.x2, s.y2)).get("door", 0.0) if z else 0.0
        near = f.loc[s.start - pad:s.stop + pad, "camera"]
        switch = bool(((near != s.camera) & (near != "")).any())
        flags.append(door >= c.get("doorway_overlap", 0.1) and switch)
    stops = stops.copy()
    stops["doorway"] = flags
    return stops


def find_stops(frames, cfg):
    """Stops of 3-60 s with walking shortly before and after, in one camera,
    not at the food / water."""
    c = cfg["stops"]
    b = cfg["behavior"]
    dt = 1.0 / cfg["sampling"]["fps"]
    df = frames.sort_values("t").reset_index(drop=True)
    cam = df["camera"].fillna("").to_numpy()
    speed = df["speed"].fillna(0).to_numpy()
    visible = cam != ""
    # stillness from how far the box centre moves over 2 s (body lengths / s): the box itself
    # jitters as the tail moves, so a half-second speed is too noisy for a standing civet
    k = max(1, int(round(1.0 / dt)))
    cx = ((df["x1"] + df["x2"]) / 2).to_numpy()
    cy = ((df["y1"] + df["y2"]) / 2).to_numpy()
    size = np.maximum(df["x2"] - df["x1"], df["y2"] - df["y1"]).to_numpy()
    disp = np.full(len(df), np.inf)
    for i in range(len(df)):
        a_, b_ = max(0, i - k), min(len(df) - 1, i + k)
        if visible[i] and cam[a_] == cam[i] == cam[b_]:
            span = (b_ - a_) * dt
            disp[i] = np.hypot(cx[b_] - cx[a_], cy[b_] - cy[a_]) / max(1.0, np.nanmedian(size[a_:b_ + 1])) / max(span, dt)
    still = disp < c["still_speed"]
    out, i, n = [], 0, len(df)
    win = int(c["moving_within_s"] / dt)
    while i < n:
        if not (visible[i] and still[i]):
            i += 1
            continue
        j = i
        while j < n and visible[j] and still[j] and cam[j] == cam[i]:
            j += 1
        dur = (j - i) * dt
        seg = df.iloc[i:j]
        # real walking before and after, measured the same steady way (2 s displacement)
        before = disp[max(0, i - win):i][cam[max(0, i - win):i] == cam[i]]
        after = disp[j:j + win][cam[j:j + win] == cam[i]]
        before, after = before[np.isfinite(before)], after[np.isfinite(after)]
        # coming from / going to another camera's room is movement too
        r = i
        while r > 0 and cam[r - 1] == cam[i]:
            r -= 1
        e = j
        while e < n and cam[e] == cam[i]:
            e += 1
        # entered this camera's view shortly before, and was not in it for win samples before that
        # (near a doorway the civet flickers between two cameras: that is not an arrival)
        rooms = c.get("count_room_changes", True)
        arrived = rooms and r > 0 and i - r <= win and not (cam[max(0, r - win):r] == cam[i]).any()
        left = rooms and e < n and e - j <= win and not (cam[e:e + win] == cam[i]).any()
        moved = (arrived or (before.max() if len(before) else 0) >= c["walk_speed"]) and                 (left or (after.max() if len(after) else 0) >= c["walk_speed"])
        tray = max(seg.get("in_food", pd.Series([0])).mean(), seg.get("in_water", pd.Series([0])).mean())
        if c["min_s"] <= dur <= c["max_s"] and moved and tray < b["tray_overlap"]:
            out.append({"camera": cam[i], "room": seg["room"].iloc[0], "start": df["t"].iloc[i],
                        "stop": df["t"].iloc[j - 1] + pd.Timedelta(seconds=dt), "duration_s": round(dur, 1),
                        "x1": seg["x1"].median(), "y1": seg["y1"].median(), "x2": seg["x2"].median(), "y2": seg["y2"].median(),
                        "det_conf": float(seg["det_conf"].mean()), "light": seg["light"].mode().iloc[0]})
        i = j
    return pd.DataFrame(out)


# ------------------------------------------------------------- step 2 ---
class StopJudge:
    """Qwen2.5-VL-3B in 4-bit. Loaded only when there are stops to judge."""

    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from .resources import torch_threads
        torch_threads()
        import gc
        gc.collect()
        torch.cuda.empty_cache()                 # the detector / pose models of the analysis are done
        name = cfg["stops"]["vlm_model"]
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        kw = dict(quantization_config=q, device_map="cuda:0")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, local_files_only=True, **kw)
            self.proc = AutoProcessor.from_pretrained(name, local_files_only=True,
                                                      min_pixels=256 * 28 * 28, max_pixels=576 * 28 * 28)
        except OSError:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kw)
            self.proc = AutoProcessor.from_pretrained(name, min_pixels=256 * 28 * 28, max_pixels=576 * 28 * 28)
        self.model.eval()
        self.torch = torch

    def __call__(self, image):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image], return_tensors="pt").to("cuda:0")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        ans = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        letter = ans.lstrip(" *(").upper()[:1]
        return (letter if letter in VLM_LABEL else "?"), ans

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def stop_grid(cam, stop, cfg, size=336):
    """2x2 grid of the civet across the stop, from the prepared video."""
    from PIL import Image
    from .video import CameraReader, Preprocessor
    light_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "denoise": "none"}}
    pre, reader = Preprocessor(light_cfg), CameraReader(cam)
    x1, y1, x2, y2 = stop["x1"], stop["y1"], stop["x2"], stop["y2"]
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.75
    tiles = []
    for q in range(4):
        t = (stop["start"] + (stop["stop"] - stop["start"]) * (q / 3)).to_pydatetime()
        frame, _ = reader.read_at(t)
        if frame is None:
            tiles.append(np.zeros((size, size, 3), np.uint8))
            continue
        f = pre(cam["id"], frame)[0]
        h, w = f.shape[:2]
        crop = f[int(max(0, cy - half)):int(min(h, cy + half)), int(max(0, cx - half)):int(min(w, cx + half))]
        tiles.append(cv2.resize(crop, (size, size)))
    reader.close()
    g = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    return Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)), g


def judge_stops(stops, cams, cfg, out_dir, progress=True):
    """Adds vlm_letter / vlm_answer to each stop. Answers are cached in
    out_dir/stops_vlm.json, so rebuilding a report does not ask again."""
    cache_path = os.path.join(out_dir, "stops_vlm.json")
    use_vlm = cfg["stops"].get("vlm", False)
    cache = json.load(open(cache_path, encoding="utf-8")) if use_vlm and os.path.exists(cache_path) else {}
    cam_by_id = {c["id"]: c for c in cams}
    grid_dir = os.path.join(out_dir, "stops")
    os.makedirs(grid_dir, exist_ok=True)
    keys = [f"{s.camera}_{s.start:%Y%m%d_%H%M%S}" for s in stops.itertuples()]
    todo = [k for k in keys if k not in cache]
    judge = None
    if todo and use_vlm:
        try:
            if progress:
                print(f"Asking the vision-language model about {len(todo)} short stops ...")
            judge = StopJudge(cfg)
        except Exception as e:                      # no GPU / library / model: every stop is flagged instead
            print(f"[stops] pre-filter not available ({e.__class__.__name__}: {str(e)[:80]}) - all stops flagged")
    throttle = Throttle(log=print if progress else None)
    letters, answers, grids = [], [], []
    for key, s in zip(keys, stops.to_dict("records")):
        cam = cam_by_id.get(s["camera"])
        gpath = os.path.join(grid_dir, key + ".jpg")
        image = None
        if cam is not None and not os.path.exists(gpath):
            image, arr = stop_grid(cam, s, cfg)
            cv2.imwrite(gpath, arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if key not in cache and judge is not None:
            if image is None:
                from PIL import Image
                image = Image.open(gpath).convert("RGB")
            cache[key] = judge(image)
            throttle.tick()
        letter, ans = cache.get(key, ("?", ""))
        letters.append(letter)
        answers.append(ans)
        grids.append(os.path.relpath(gpath, out_dir).replace("\\", "/"))
    if judge is not None:
        judge.close()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1, ensure_ascii=False)
    stops = stops.copy()
    stops["vlm_letter"], stops["vlm_answer"], stops["grid"] = letters, answers, grids
    return stops


def score_stops(stops, cams, cfg, progress=True):
    """Order hint for Quick check, never a decision: a small classifier trained
    on public MammalNet clips (cats and other small carnivores) scores how much
    each stop looks like toileting. Tested on 5 Apr 2026 it put 3 of the 5 real
    toileting stops in its top 25 of 127 (random order: about 1); its grooming
    score was no better than random, so only the toileting score is used."""
    from .config import resolve
    path = resolve(cfg["stops"].get("order_model", "models/civet_behavior_cls.pt"))
    stops = stops.copy()
    stops["toilet_score"] = 0.0
    if not len(stops) or not os.path.exists(path):
        return stops
    from ultralytics import YOLO
    from .video import CameraReader, Preprocessor
    model = YOLO(path)
    k = [i for i, n in model.names.items() if n == "toileting"]
    if not k:
        return stops
    light = {**cfg, "preprocess": {**cfg["preprocess"], "denoise": "none"}}
    cam_by_id = {c["id"]: c for c in cams}
    if progress:
        print(f"Ordering {len(stops)} short stops for Quick check ...")
    scores = []
    for s in stops.to_dict("records"):
        cam = cam_by_id.get(s["camera"])
        crops = []
        if cam is not None:
            rd, pre = CameraReader(cam), Preprocessor(light)
            x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
            cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.75
            for q in range(5):
                f, _ = rd.read_at((s["start"] + (s["stop"] - s["start"]) * ((q + 0.5) / 5)).to_pydatetime())
                if f is None:
                    continue
                img = pre(s["camera"], f)[0]
                h, w = img.shape[:2]
                c = img[int(max(0, cy - half)):int(min(h, cy + half)), int(max(0, cx - half)):int(min(w, cx + half))]
                if c.size:
                    crops.append(cv2.cvtColor(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
            rd.close()
        if crops:
            res = model.predict(crops, imgsz=160, verbose=False)
            scores.append(float(np.mean([r.probs.data[k[0]].item() for r in res])))
        else:
            scores.append(0.0)
    stops["toilet_score"] = scores
    return stops


def stop_events(stops):
    """Rows for events.csv: category 'stop', never counted in the sheet.
    Flagged for checking when the model says toileting / marking / grooming,
    or when no model answer is available."""
    rows = []
    for s in stops.to_dict("records"):
        letter = s["vlm_letter"]
        if letter == "F":
            continue                                            # model sees no animal
        likely = None if s.get("doorway") else POSSIBLE.get(letter)
        rows.append({
            "start": s["start"], "stop": s["stop"], "duration_s": s["duration_s"],
            "behavior": "doorway_pass" if s.get("doorway") else ("possible_" + VLM_LABEL[letter] if likely else "short_stop"),
            "category": "stop", "room": s["room"], "camera": s["camera"], "confidence": round(s["det_conf"], 3),
            "light": s["light"], "visibility": "visible", "quality": "clear",
            "note": "passing through a doorway" if s.get("doorway") else (
                "stood still a few seconds - what was it doing?" if letter == "?" else f"model: {VLM_LABEL.get(letter)}"),
            "needs_review": bool(likely) or (letter == "?" and not s.get("doorway")),
            "grid": s["grid"], "vlm_answer": s["vlm_answer"], "toilet_score": round(s.get("toilet_score", 0.0), 3),
        })
    return pd.DataFrame(rows)
