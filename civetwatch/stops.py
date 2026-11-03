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
    visible = cam != ""
    disp = _displacement(df, dt)
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
    out = pd.DataFrame(out)
    if len(out):
        out["kind"] = "short_stop"
    busy = find_busy(df, disp, cfg, out)
    return pd.concat([out, busy], ignore_index=True).sort_values("start").reset_index(drop=True) if len(busy) else out


def _displacement(df, dt):
    """How far the box centre moves over 2 s (body lengths / s). The box itself
    jitters as the tail moves, so a half-second speed is too noisy for a
    standing civet."""
    cam = df["camera"].fillna("").to_numpy()
    visible = cam != ""
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
    return disp


def find_busy(df, disp, cfg, stops):
    """'Busy while staying in place': the civet's box does not move but its body
    does (motion inside the box). Grooming often happens like this while it sits
    or lies, for longer than a short stop, so the short-stop finder misses it."""
    c = cfg["stops"].get("busy", {})
    if not c.get("enabled", True) or not len(df):
        return pd.DataFrame()
    dt = 1.0 / cfg["sampling"]["fps"]
    cam = df["camera"].fillna("").to_numpy()
    motion = df["motion"].fillna(0).to_numpy()
    food = df.get("in_food", pd.Series(0, index=df.index)).fillna(0).to_numpy()
    water = df.get("in_water", pd.Series(0, index=df.index)).fillna(0).to_numpy()
    hit = (cam != "") & (disp < cfg["stops"]["still_speed"]) & (motion >= c.get("motion_min", 0.04)) \
        & (np.maximum(food, water) < cfg["behavior"]["tray_overlap"])
    gap = int(c.get("gap_s", 2) / dt)
    taken = [(s["start"], s["stop"]) for s in stops.to_dict("records")] if len(stops) else []
    out, i, n = [], 0, len(df)
    while i < n:
        if not hit[i]:
            i += 1
            continue
        j, last = i, i
        while j < n and cam[j] in (cam[i], "") and (hit[j] or j - last <= gap):
            if hit[j]:
                last = j
            j += 1
        seg = df.iloc[i:last + 1]
        dur = (last + 1 - i) * dt
        start, stop = df["t"].iloc[i], df["t"].iloc[last] + pd.Timedelta(seconds=dt)
        overlaps = any(a < stop and start < b for a, b in taken)
        if dur >= c.get("min_s", 4) and not overlaps:
            seg = seg[seg["camera"] == cam[i]]
            out.append({"camera": cam[i], "room": seg["room"].iloc[0], "start": start, "stop": stop,
                        "duration_s": round(dur, 1), "x1": seg["x1"].median(), "y1": seg["y1"].median(),
                        "x2": seg["x2"].median(), "y2": seg["y2"].median(), "det_conf": float(seg["det_conf"].mean()),
                        "light": seg["light"].mode().iloc[0], "kind": "busy_still",
                        "busy": float(seg["motion"].mean()) * min(dur, 120)})
        i = last + 1
    out = pd.DataFrame(out)
    keep = c.get("max_per_day")
    if keep and len(out) > keep:                       # the busiest ones first
        out = out.nlargest(keep, "busy")
    return out.drop(columns=["busy"], errors="ignore")


def find_box_visits(frames, cams, cfg):
    """Every visit of the civet to a 'box_top' zone (the top of the sleeping box
    on cam2, where it usually marks, urinates and defecates). Unlike short stops,
    no walking before / after is needed: each visit is checked."""
    from .zones import ZoneMap
    c = cfg["stops"].get("box_visits", {})
    if not c.get("enabled", True) or not len(frames):
        return pd.DataFrame()
    dt = 1.0 / cfg["sampling"]["fps"]
    W = cfg["sampling"]["process_width"]
    H = int(round(W * 9 / 16))
    df = frames.sort_values("t").reset_index(drop=True)
    out = []
    for cam in cams:
        z = ZoneMap(cam.get("zones"))
        z.set_size(W, H)
        polys = [p for zt, _, p in z.polygons() if zt == "box_top"]
        if not polys:
            continue
        zx1, zy1 = np.min([p.min(0) for p in polys], axis=0)
        zx2, zy2 = np.max([p.max(0) for p in polys], axis=0)
        mine = (df["camera"].fillna("") == cam["id"]).to_numpy()
        on = np.zeros(len(df), bool)
        for i in np.flatnonzero(mine):
            r = df.iloc[i]
            on[i] = z.box_overlap((r["x1"], r["y1"], r["x2"], r["y2"])).get("box_top", 0) >= c.get("overlap", 0.3)
        # 'clear' = the civet is not on / next to the box top (another camera, out of view, or elsewhere in cam2)
        cw, ch = (zx2 - zx1) * 0.15, (zy2 - zy1) * 0.6
        bx = df[["x1", "y1", "x2", "y2"]].to_numpy()
        near = mine & ~((bx[:, 2] < zx1 - cw) | (bx[:, 0] > zx2 + cw) | (bx[:, 1] > zy2 + ch) | (bx[:, 3] < zy1 - ch))
        # positive evidence only: seen in another camera, or in this camera away from the box top, for 2 s.
        # (On the box top the civet is half out of the picture, so 'not detected' does not mean 'gone'.)
        seen = df["camera"].fillna("").to_numpy() != ""
        clear0 = seen & ~near
        k2 = max(1, int(2 / dt))
        clear = np.array([clear0[k:k + k2].all() for k in range(len(df))])
        tt = df["t"]
        far = int(c.get("clear_search_s", 600) / dt)
        gap = int(c.get("gap_s", 3) / dt)
        i, n = 0, len(df)
        while i < n:
            if not on[i]:
                i += 1
                continue
            j, last = i, i
            while j < n and (on[j] or j - last <= gap):
                if on[j]:
                    last = j
                j += 1
            dur = (last + 1 - i) * dt
            if dur >= c.get("min_s", 2):
                seg = df.iloc[i:last + 1]
                seg = seg[seg["camera"] == cam["id"]]
                out.append({"camera": cam["id"], "room": seg["room"].iloc[0], "start": df["t"].iloc[i],
                            "stop": df["t"].iloc[last] + pd.Timedelta(seconds=dt), "duration_s": round(dur, 1),
                            "x1": seg["x1"].median(), "y1": seg["y1"].median(), "x2": seg["x2"].median(),
                            "y2": seg["y2"].median(), "det_conf": float(seg["det_conf"].mean()),
                            "light": seg["light"].mode().iloc[0], "kind": "box_top",
                            "clear_before": next((tt.iloc[k] for k in range(i - 1, max(-1, i - far), -1) if clear[k]), None),
                            "clear_after": next((tt.iloc[k] for k in range(last + 1, min(n, last + far)) if clear[k]), None),
                            "zx1": float(zx1), "zy1": float(zy1), "zx2": float(zx2), "zy2": float(zy2)})
            i = last + 1
    out = pd.DataFrame(out)
    if len(out):
        # visits with no sighting elsewhere in between are one session on the box top: one item to check
        sess, cur = [], None
        for r in out.sort_values(["camera", "start"]).to_dict("records"):
            same = cur is not None and r["camera"] == cur["camera"] and cur["clear_after"] is not None \
                and not pd.isna(cur["clear_after"]) and r["start"] <= cur["clear_after"]
            if same:
                cur["stop"] = max(cur["stop"], r["stop"])
                cur["clear_after"] = r["clear_after"]
            else:
                cur = dict(r)
                sess.append(cur)
        out = pd.DataFrame(sess)
        out["duration_s"] = (out["stop"] - out["start"]).dt.total_seconds().round(1)
    return out


def visit_grid(cam, visit, cfg, size=(480, 200)):
    """Before / during / during / after pictures of the box top, and how much
    the box top changed between before and after (new droppings, a wet patch).
    Before / after are taken a few seconds outside the visit, when the civet has gone."""
    from .video import CameraReader, Preprocessor
    light = {**cfg, "preprocess": {**cfg["preprocess"], "denoise": "none"}}
    pre, rd = Preprocessor(light), CameraReader(cam)
    from .zones import ZoneMap
    pad = pd.Timedelta(seconds=cfg["stops"].get("box_visits", {}).get("before_after_s", 4))
    x1, y1, x2, y2 = visit["zx1"], visit["zy1"], visit["zx2"], visit["zy2"]
    mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.6
    a, b = visit["start"], visit["stop"]
    cb, ca = visit.get("clear_before"), visit.get("clear_after")
    before = cb - pad / 4 if cb is not None and not pd.isna(cb) else a - pad
    after = ca + pad / 4 if ca is not None and not pd.isna(ca) else b + pad
    times = [before, a + (b - a) / 3, a + (b - a) * 2 / 3, after]
    W = cfg["sampling"]["process_width"]
    zm = ZoneMap(cam.get("zones"))
    zm.set_size(W, int(round(W * 9 / 16)))
    zmask = np.zeros((int(round(W * 9 / 16)), W), np.uint8)
    for zt, _, poly in zm.polygons():
        if zt == "box_top":
            cv2.fillPoly(zmask, [poly], 1)
    for zt, _, poly in zm.polygons():
        if zt == "ignore":
            cv2.fillPoly(zmask, [poly], 0)
    cy1, cy2, cx1, cx2 = int(max(0, y1 - my)), int(min(zmask.shape[0], y2 + my)), int(max(0, x1 - mx)), int(min(W, x2 + mx))
    zmask = cv2.resize(zmask[cy1:cy2, cx1:cx2], size, interpolation=cv2.INTER_NEAREST).astype(bool)
    names = ["before", "during", "during", "after"]
    tiles, greys = [], []
    for t, nm in zip(times, names):
        f, _ = rd.read_at(t.to_pydatetime())
        if f is None:
            tile = np.zeros((size[1], size[0], 3), np.uint8)
        else:
            img = pre(cam["id"], f)[0]
            h, w = img.shape[:2]
            tile = cv2.resize(img[cy1:cy2, cx1:cx2], size)
        greys.append(cv2.GaussianBlur(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY), (7, 7), 0))
        cv2.putText(tile, f"{nm} {t:%H:%M:%S}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    rd.close()
    # change between before and after, ignoring an overall brightness shift (IR, clouds)
    d = cv2.absdiff(greys[0], greys[3]).astype(np.float32)
    d -= np.median(d)
    hit = (d > 25) & zmask
    change = float(hit.sum() / max(1, zmask.sum()))
    t3 = tiles[3].astype(np.float32)
    t3[hit] = 0.5 * t3[hit] + np.array([0, 0, 127], np.float32)            # changed pixels in red
    tiles[3] = t3.astype(np.uint8)
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), change


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
    letters, answers, grids, changes = [], [], [], []
    for key, s in zip(keys, stops.to_dict("records")):
        cam = cam_by_id.get(s["camera"])
        gpath = os.path.join(grid_dir, key + ".jpg")
        image = None
        if s.get("kind") == "box_top":
            gpath = os.path.join(grid_dir, key + "_box.jpg")
            cpath = gpath + ".change"
            if cam is not None and not (os.path.exists(gpath) and os.path.exists(cpath)):
                arr, ch = visit_grid(cam, s, cfg)
                cv2.imwrite(gpath, arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with open(cpath, "w") as fh:
                    fh.write(f"{ch:.5f}")
            changes.append(float(open(cpath).read()) if os.path.exists(cpath) else 0.0)
        else:
            changes.append(0.0)
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
    stops["box_change"] = changes
    return stops


def stop_crops(cam, stop, cfg, n=5, reader=None):
    """n grey crops of the civet spread across a stop (same look by day and by night)."""
    from .video import CameraReader, Preprocessor
    light = {**cfg, "preprocess": {**cfg["preprocess"], "denoise": "none"}}
    rd, pre = reader or CameraReader(cam), Preprocessor(light)
    x1, y1, x2, y2 = stop["x1"], stop["y1"], stop["x2"], stop["y2"]
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.75
    crops = []
    for q in range(n):
        f, _ = rd.read_at((stop["start"] + (stop["stop"] - stop["start"]) * ((q + 0.5) / n)).to_pydatetime())
        if f is None:
            continue
        img = pre(cam["id"], f)[0]
        h, w = img.shape[:2]
        c = img[int(max(0, cy - half)):int(min(h, cy + half)), int(max(0, cx - half)):int(min(w, cx + half))]
        if c.size:
            crops.append(cv2.cvtColor(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
    if reader is None:
        rd.close()
    return crops


def score_stops(stops, cams, cfg, progress=True):
    """Hints for the Behavior check list, never a decision on their own.

    - toilet_score: a small classifier trained on public MammalNet clips (cats and
      other small carnivores). On 5 Apr 2026 it put 3 of the 5 real toileting
      stops in its top 25 of 127 (random order: about 1).
    - guess / guess_prob: the classifier trained on your own Behavior check
      answers (civetwatch train-behavior), when models/civet_stop_cls.pt exists."""
    from .config import resolve
    stops = stops.copy()
    stops["toilet_score"], stops["guess"], stops["guess_prob"] = 0.0, "", 0.0
    models = {}
    for key, default in (("order_model", "models/civet_behavior_cls.pt"), ("stop_model", "models/civet_stop_cls.pt")):
        path = resolve(cfg["stops"].get(key) or default)
        if os.path.exists(path):
            from ultralytics import YOLO
            models[key] = YOLO(path)
    if not len(stops) or not models:
        return stops
    cam_by_id = {c["id"]: c for c in cams}
    if progress:
        print(f"Scoring {len(stops)} stops for the Behavior check list ({', '.join(models)}) ...")
    order = models.get("order_model")
    k_toilet = [i for i, nm in order.names.items() if nm == "toileting"] if order else []
    own = models.get("stop_model")
    tscore, guess, prob = [], [], []
    for s in stops.to_dict("records"):
        cam = cam_by_id.get(s["camera"])
        crops = stop_crops(cam, s, cfg) if cam is not None else []
        t, g, p = 0.0, "", 0.0
        if crops and k_toilet:
            res = order.predict(crops, imgsz=160, verbose=False)
            t = float(np.mean([r.probs.data[k_toilet[0]].item() for r in res]))
        if crops and own is not None:
            res = own.predict(crops, imgsz=160, verbose=False)
            mean = np.mean([r.probs.data.cpu().numpy() for r in res], axis=0)
            g, p = own.names[int(mean.argmax())], float(mean.max())
            names = {v: k for k, v in own.names.items()}
            t = max([t] + [float(mean[names[c]]) for c in ("urination", "defecation") if c in names])
        tscore.append(t), guess.append(g), prob.append(p)
    stops["toilet_score"], stops["guess"], stops["guess_prob"] = tscore, guess, prob
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
            "behavior": "box_top_visit" if s.get("kind") == "box_top" else (
                "doorway_pass" if s.get("doorway") else ("possible_" + VLM_LABEL[letter] if likely else "short_stop")),
            "category": "stop", "room": s["room"], "camera": s["camera"], "confidence": round(s["det_conf"], 3),
            "light": s["light"], "visibility": "visible", "quality": "clear",
            "note": "passing through a doorway" if s.get("doorway") else (
                ("on the sleeping box top: marking, urination or defecation?" if s.get("kind") == "box_top" else
                 "moved its body while staying in place - grooming?" if s.get("kind") == "busy_still" else
                 "stood still a few seconds - what was it doing?") if letter == "?" else f"model: {VLM_LABEL.get(letter)}"),
            "needs_review": bool(likely) or (letter == "?" and not s.get("doorway")),
            "box_change": round(float(s.get("box_change", 0.0) or 0.0), 4),
            "grid": s["grid"], "vlm_answer": s["vlm_answer"], "toilet_score": round(s.get("toilet_score", 0.0), 3),
            "kind": s.get("kind", "short_stop"), "guess": s.get("guess", ""),
            "guess_prob": round(float(s.get("guess_prob", 0.0) or 0.0), 3),
        })
    return pd.DataFrame(rows)
