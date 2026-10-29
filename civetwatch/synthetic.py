"""Synthetic 4-room / 4-camera recording with a scripted civet, for trying the
whole pipeline before real footage and trained models are available.

Writes videos, zone maps, a manifest and ground_truth.csv (the scripted
behavior every 0.5 s) so `civetwatch compare` can score the result.
"""
import json
import math
import os
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import yaml

from . import ethogram as E

W, H, FPS = 640, 360, 5
BODY = 70

ROOMS = {
    "R1": {"doors": {"right": (600, 130, 640, 240)}, "tyre": (200, 255, 48)},
    "R2": {"doors": {"left": (0, 130, 40, 240), "right": (600, 130, 640, 240)},
           "food": (180, 285, 30), "water": (400, 262, 480, 312)},
    "R3": {"doors": {"left": (0, 130, 40, 240), "right": (600, 130, 640, 240)}, "log": (300, 70, 520, 108)},
    "R4": {"doors": {"left": (0, 130, 40, 240)}, "box": (400, 60, 570, 175)},
}
CAMS = [("cam1", "R1", "night"), ("cam2", "R2", "night"), ("cam3", "R3", "night_pink"), ("cam4", "R4", "day")]


def scenario():
    """(room, kind, seconds, waypoints, truth label). Walking moves along the
    waypoints; other kinds stay at the first one."""
    L, R = (15, 185), (625, 185)
    return [
        ("R1", "curl", 150, [(200, 255)], E.CURLING_UP),
        ("R1", "walk", 7, [(200, 255), (420, 200), R], E.WALKING),
        ("R2", "walk", 5, [L, (150, 240)], E.WALKING),
        ("R2", "feed", 30, [(150, 240)], E.FEEDING),
        ("R2", "stand", 5, [(150, 240)], E.ACTIVE_OTHER),
        ("R2", "feed", 25, [(150, 240)], E.FEEDING),
        ("R2", "walk", 4, [(150, 240), (300, 200), (410, 240)], E.WALKING),
        ("R2", "drink", 10, [(410, 240)], E.DRINKING),
        ("R2", "walk", 3, [(410, 240), (330, 170)], E.WALKING),
        ("R2", "groom", 20, [(330, 170)], E.GROOMING),
        ("R2", "walk", 4, [(330, 170), R], E.WALKING),
        ("R3", "walk", 5, [L, (380, 220)], E.WALKING),
        ("R3", "stretch", 100, [(380, 220)], E.STRETCHED_OUT),
        ("R3", "walk", 5, [(380, 220), R], E.WALKING),
        ("R4", "walk", 5, [L, (300, 220), (440, 190), (470, 140)], E.WALKING),
        ("R4", "hidden_box", 90, [(480, 120)], E.SLEEP_IN_BOX),
        ("R4", "walk", 6, [(470, 140), (440, 200), (250, 230), L], E.WALKING),
        ("R3", "walk", 3, [R, (420, 60), (420, -60)], E.WALKING),
        ("R3", "hidden", 40, [(420, -60)], E.NOT_VISIBLE),
        ("R3", "walk", 4, [(420, -60), (300, 150), L], E.WALKING),
        ("R2", "walk", 5, [R, (400, 200), (150, 240)], E.WALKING),
        ("R2", "feed", 20, [(150, 240)], E.FEEDING),
        ("R2", "walk", 5, [(150, 240), L], E.WALKING),
        ("R1", "walk", 5, [R, (380, 200), (260, 170)], E.WALKING),
        ("R1", "lie", 80, [(260, 170)], E.LYING_DOWN),
    ]


def _along(points, frac):
    pts = np.array(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if seg.sum() == 0:
        return pts[0], 0.0
    d = frac * seg.sum()
    for i, s in enumerate(seg):
        if d <= s or i == len(seg) - 1:
            p = pts[i] + (pts[i + 1] - pts[i]) * (d / s if s else 0)
            v = pts[i + 1] - pts[i]
            return p, math.atan2(v[1], v[0])
        d -= s


def timeline():
    """Per video frame: (room, kind, x, y, heading, truth)."""
    out, heading = [], 0.0
    for room, kind, secs, pts, truth in scenario():
        n = int(secs * FPS)
        for k in range(n):
            if kind == "walk":
                p, heading = _along(pts, k / max(1, n - 1))
            else:
                p = np.array(pts[0], float)
            out.append((room, kind, float(p[0]), float(p[1]), heading, truth))
    return out


def _background(room, seed):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (150, 168, 182), np.uint8)
    noise = rng.normal(0, 9, (H, W, 1))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.rectangle(img, (0, 0), (W, 40), (95, 105, 115), -1)                      # back wall
    for x in range(0, W, 24):                                                  # fence mesh
        cv2.line(img, (x, 0), (x + 20, 40), (70, 80, 85), 1)
    r = ROOMS[room]
    for x1, y1, x2, y2 in r["doors"].values():
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 70, 80), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 45, 50), 3)
    if "tyre" in r:
        x, y, rad = r["tyre"]
        cv2.circle(img, (x, y), rad, (55, 55, 60), 14)
    if "food" in r:
        x, y, rad = r["food"]
        cv2.circle(img, (x, y), rad, (200, 200, 205), -1)
        cv2.circle(img, (x, y), rad, (120, 120, 125), 3)
        for k in range(5):
            cv2.circle(img, (x - 12 + k * 6, y - 4 + (k % 2) * 8), 5, (60, 140, 220), -1)
    if "water" in r:
        x1, y1, x2, y2 = r["water"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (190, 190, 195), -1)
        cv2.rectangle(img, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), (170, 140, 110), -1)
    if "log" in r:
        x1, y1, x2, y2 = r["log"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 90, 120), -1)
        cv2.ellipse(img, (x1, (y1 + y2) // 2), (8, (y2 - y1) // 2), 0, 0, 360, (90, 130, 160), -1)
    return img


def _draw_box(img, room):
    if "box" not in ROOMS[room]:
        return
    x1, y1, x2, y2 = ROOMS[room]["box"]
    cv2.rectangle(img, (x1, y1), (x2, y2), (120, 170, 200), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 110, 140), 3)
    cv2.rectangle(img, (x1 + 50, y2 - 50), (x1 + 110, y2), (40, 50, 60), -1)    # opening


def _rot(heading, dx, dy):
    c, s = math.cos(heading), math.sin(heading)
    return dx * c - dy * s, dx * s + dy * c


def draw_civet(img, x, y, heading, kind, phase):
    grey, dark = (135, 140, 145), (25, 25, 28)
    if kind in ("hidden", "hidden_box"):
        return

    def P(dx, dy):
        ox, oy = _rot(heading, dx, dy)
        return int(x + ox), int(y + oy)

    if kind == "curl":
        cv2.circle(img, (int(x), int(y)), 25, grey, -1)
        for a in range(0, 360, 40):
            cv2.circle(img, (int(x + 14 * math.cos(math.radians(a))), int(y + 14 * math.sin(math.radians(a)))), 4, dark, -1)
        cv2.ellipse(img, (int(x), int(y)), (28, 28), 0, 20, 200, dark, 6)
        return
    half = {"stretch": 52, "lie": 26}.get(kind, 34)
    width = {"stretch": 10, "lie": 17}.get(kind, 13)
    # tail (ringed): straight out behind, or folded along the body when lying down
    for k in range(7):
        if kind == "lie":
            cx, cy = P(-half + k * 7, width + 6)
        else:
            cx, cy = P(-half - 6 - k * 7, 3 * math.sin(phase * 0.5 + k * 0.6))
        cv2.circle(img, (cx, cy), 5, dark if k % 2 == 0 else (215, 215, 215), -1)
    # legs
    swing = 6 * math.sin(phase * 2.5) if kind == "walk" else 0
    spread = 16 if kind == "stretch" else 12
    for lx, sgn in ((half * 0.6, 1), (-half * 0.6, -1)):
        for side in (-1, 1):
            off = swing * sgn * side
            if kind == "groom" and lx < 0 and side == 1:
                off = 16 * math.sin(phase * 9)                 # fast scratching hind paw
            cv2.line(img, P(lx, side * width * 0.6), P(lx + off, side * (width + spread)), dark, 4)
    # body + spots + crest
    cx, cy = P(0, 0)
    cv2.ellipse(img, (cx, cy), (half, width), math.degrees(heading), 0, 360, grey, -1)
    for k in range(-3, 4):
        for side in (-1, 1):
            cv2.circle(img, P(k * half / 4.2, side * width * 0.45), 3, dark, -1)
    cv2.line(img, P(-half * 0.8, 0), P(half * 0.8, 0), dark, 2)
    # head
    hx, hy = half + 10, 0
    if kind in ("feed", "drink"):
        hx += 6 + 3 * math.sin(phase * (8 if kind == "drink" else 5))   # chewing / lapping
    if kind == "groom":
        hx, hy = half * 0.2 + 22 * math.sin(phase * 4), width + 10     # head turned back to lick
    cv2.circle(img, P(hx, hy), 11, (110, 115, 120), -1)
    cv2.circle(img, P(hx + 8, hy), 4, dark, -1)


def render(kind_light, img, rng, grain, blur=False):
    """Night = dark infrared grey with sensor grain (mostly fixed-pattern so the
    files stay small, plus a little noise that changes every frame)."""
    if blur:
        img = cv2.GaussianBlur(img, (31, 31), 0)
    if kind_light == "day":
        return np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g * 0.35 + 6 + grain + rng.normal(0, 1.5, g.shape)
    g = np.clip(g, 0, 255)
    out = cv2.merge([g, g, g])
    if kind_light == "night_pink":
        out = out * np.array([0.97, 0.9, 1.06])
    return np.clip(out, 0, 255).astype(np.uint8)


def zones_for(room):
    r, zs = ROOMS[room], []
    n = lambda pts: [[round(px / W, 4), round(py / H, 4)] for px, py in pts]
    rect = lambda x1, y1, x2, y2: [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for x1, y1, x2, y2 in r["doors"].values():
        zs.append({"type": "door", "name": "DOOR", "points": n(rect(x1 - 5, y1, x2 + 5, y2))})
    zs.append({"type": "fence", "name": "FENCE", "points": n(rect(0, 0, W, 40))})
    if "tyre" in r:
        x, y, rad = r["tyre"]
        zs.append({"type": "tyre", "name": "TYRE", "points": n(rect(x - rad - 10, y - rad - 10, x + rad + 10, y + rad + 10))})
    if "food" in r:
        x, y, rad = r["food"]
        zs.append({"type": "food", "name": "FOOD", "points": n(rect(x - rad - 25, y - rad - 30, x + rad + 25, y + rad + 10))})
    if "water" in r:
        x1, y1, x2, y2 = r["water"]
        zs.append({"type": "water", "name": "WATER", "points": n(rect(x1 - 25, y1 - 35, x2 + 20, y2 + 5))})
    if "log" in r:
        zs.append({"type": "log", "name": "LOG", "points": n(rect(*r["log"]))})
    if "box" in r:
        x1, y1, x2, y2 = r["box"]
        zs.append({"type": "rest_box", "name": "REST BOX", "points": n(rect(x1 - 10, y1 - 10, x2 + 10, y2 + 10))})
    return zs


def generate(out_dir, start=datetime(2026, 7, 20, 20, 0, 0), seed=7, progress=True):
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "zones"), exist_ok=True)
    tl = timeline()
    rng = np.random.default_rng(seed)
    blur_frames = range(int(262 * FPS), int(264 * FPS))        # 2 s of a smeared lens on cam2
    manifest = {"date": start.strftime("%Y-%m-%d"), "cameras": []}
    for ci, (cid, room, light) in enumerate(CAMS):
        bg = _background(room, seed + ci)
        grain = rng.normal(0, 5, (H, W)).astype(np.float32)
        name = f"{cid}_{start:%Y%m%d_%H%M%S}.mp4"
        path = os.path.join(out_dir, "videos", name)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for i, (r, kind, x, y, heading, _) in enumerate(tl):
            img = bg.copy()
            if r == room:
                draw_civet(img, x, y, heading, kind, i / FPS * 2 * math.pi)
            _draw_box(img, room)
            writer.write(render(light, img, rng, grain, blur=(cid == "cam2" and i in blur_frames)))
        writer.release()
        zpath = os.path.join(out_dir, "zones", f"{cid}.json")
        with open(zpath, "w", encoding="utf-8") as f:
            json.dump({"camera": cid, "zones": zones_for(room)}, f, indent=2)
        manifest["cameras"].append({"id": cid, "room": room, "zones": f"zones/{cid}.json",
                                    "files": [{"path": f"videos/{name}", "start": start.strftime("%Y-%m-%d %H:%M:%S")}]})
        if progress:
            print(f"  wrote {path}")
    with open(os.path.join(out_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    truth = []
    for k in range(int(len(tl) / FPS * 2)):                    # one row per 0.5 s sample
        i = int(k * FPS / 2)
        truth.append({"t": start + timedelta(seconds=k / 2), "room": tl[i][0], "true_label": tl[i][5]})
    truth = pd.DataFrame(truth)
    truth.to_csv(os.path.join(out_dir, "ground_truth.csv"), index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    return os.path.join(out_dir, "manifest.yaml")
