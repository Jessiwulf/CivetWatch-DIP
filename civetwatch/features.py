"""Turn box + body dots + zones into numbers per sample (slide 13):
where it is, how fast it moves, body/head angle, limb distance, nose zone."""
from collections import deque

import cv2
import numpy as np

from .ethogram import KP, MARKING_OBJECTS

ZONE_FEATURES = ["in_food", "in_water", "in_rest_box", "in_door", "in_log", "in_tyre", "in_fence",
                 "in_latrine"]   # not 'ignore'
FEATURES = [
    "visible", "speed", "motion", "aspect", "compactness",
    "head_drop", "tail_raise", "hip_drop", "paw_spread", "hindpaw_head", "nose_hips",
    "nose_motion", "paw_motion", "kp_frac",
    "rear_obj", "nose_obj", "rear_latrine",       # where the rear end / nose are (marking, toileting)
    "at_edge",                                    # box touches the picture edge: body partly out of view
] + ZONE_FEATURES


def zone_features(box, kp, center, zonemap, t=None):
    """in_<zone>: for food / water the nose dot when the pose model sees it,
    otherwise 1 if the body centre is inside, else the share of the box inside."""
    nose = _pt(kp, "nose")
    overlap = zonemap.box_overlap(box, t)
    nose_zones = zonemap.zones_at(nose, t) if nose is not None else None
    center_zones = zonemap.zones_at(center, t)
    out = {}
    for zf in ZONE_FEATURES:
        zt = zf[3:]
        if zt in ("food", "water") and nose_zones is not None:
            out[zf] = 1.0 if zt in nose_zones else 0.0
        elif zt in center_zones:
            out[zf] = 1.0
        else:
            out[zf] = overlap.get(zt, 0.0)
    return out


def at_edge(box, shape, margin=4):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return 1.0 if x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin else 0.0


def body_point_zones(kp, zonemap, t=None):
    """rear_obj: tail base (or hips) on a log / door / hollow log; nose_obj: the
    nose on one too; rear_latrine: the rear end in a latrine zone."""
    out = {"rear_obj": 0.0, "nose_obj": 0.0, "rear_latrine": 0.0}
    if kp is None:
        return out
    rear = _pt(kp, "tail_base")
    if rear is None:
        rear = _pt(kp, "hips")
    nose = _pt(kp, "nose")
    rz = zonemap.zones_at(rear, t) if rear is not None else set()
    nz = zonemap.zones_at(nose, t) if nose is not None else set()
    out["rear_obj"] = 1.0 if rz & set(MARKING_OBJECTS) else 0.0
    out["nose_obj"] = 1.0 if nz & set(MARKING_OBJECTS) else 0.0
    out["rear_latrine"] = 1.0 if "latrine" in rz else 0.0
    return out


def _pt(kp, name):
    if kp is None:
        return None
    x, y, c = kp[KP[name]]
    return None if np.isnan(x) or np.isnan(c) else np.array([x, y], np.float32)


def _dist(a, b):
    return float(np.linalg.norm(a - b)) if a is not None and b is not None else None


class FeatureExtractor:
    def __init__(self, cfg):
        self.fps = cfg["sampling"]["fps"]
        self.hist = deque(maxlen=4)                  # last visible samples
        self.lengths = {}                            # cam -> deque of body lengths
        self.diff_thr = 18

    def typical_length(self, cam, length):
        d = self.lengths.setdefault(cam, deque(maxlen=7200))
        d.append(length)
        return float(np.percentile(d, 80)) if len(d) >= 5 else length

    def missing(self):
        self.hist.clear()
        return {f: 0.0 for f in FEATURES}

    def __call__(self, t, cam, det, kp, gray, prev_gray, zonemap):
        x1, y1, x2, y2 = det.box
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        length = max(w, h)
        typ = self.typical_length(cam, length)
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
        if kp is not None:
            body = [p for p in (_pt(kp, "withers"), _pt(kp, "hips")) if p is not None]
            if body:
                center = np.mean(body, axis=0)

        f = {k: 0.0 for k in FEATURES}
        f["visible"] = 1.0
        f["aspect"] = det.elongation or float(max(w, h) / min(w, h))
        f["compactness"] = float(length / typ) if typ else 1.0
        f["at_edge"] = at_edge(det.box, gray.shape)

        # speed in body lengths per second, over ~1 s in the same room
        prev = next((p for p in reversed(self.hist) if p["cam"] == cam
                     and 0 < (t - p["t"]).total_seconds() <= 1.6), None)
        if prev is not None:
            dt = (t - prev["t"]).total_seconds()
            f["speed"] = float(np.linalg.norm(center - prev["center"]) / typ / dt)

        # changed pixels inside the box since the last sample (chewing, lapping, scratching)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            xi1, yi1, xi2, yi2 = int(max(0, x1)), int(max(0, y1)), int(x2), int(y2)
            roi = cv2.absdiff(gray[yi1:yi2, xi1:xi2], prev_gray[yi1:yi2, xi1:xi2])
            f["motion"] = float((roi > self.diff_thr).mean()) if roi.size else 0.0

        nose = _pt(kp, "nose")
        rel = {}
        if kp is not None:
            f["kp_frac"] = float(np.mean(~np.isnan(kp[:, 2])))
            withers, hips = _pt(kp, "withers"), _pt(kp, "hips")
            tb, tt = _pt(kp, "tail_base"), _pt(kp, "tail_tip")
            hl, hr = _pt(kp, "hind_left_paw"), _pt(kp, "hind_right_paw")
            ears = [p for p in (_pt(kp, "left_ear"), _pt(kp, "right_ear")) if p is not None]
            if nose is not None and withers is not None:
                f["head_drop"] = float((nose[1] - withers[1]) / typ)
            if tb is not None and tt is not None:
                f["tail_raise"] = float((tb[1] - tt[1]) / typ)
            if hips is not None and withers is not None:
                f["hip_drop"] = float((hips[1] - withers[1]) / typ)
            if hl is not None and hr is not None:
                f["paw_spread"] = _dist(hl, hr) / typ
            head_pts = ([nose] if nose is not None else []) + ears
            paw_d = [_dist(p, q) for p in (hl, hr) if p is not None for q in head_pts]
            f["hindpaw_head"] = min(paw_d) / typ if paw_d else 1.0
            nh = _dist(nose, hips)
            f["nose_hips"] = nh / typ if nh is not None else 1.0
            for name, p in (("nose", nose), ("hl", hl), ("hr", hr)):
                if p is not None:
                    rel[name] = p - center
            if prev is not None:
                dt = (t - prev["t"]).total_seconds()
                if "nose" in rel and "nose" in prev["rel"]:
                    f["nose_motion"] = float(np.linalg.norm(rel["nose"] - prev["rel"]["nose"]) / typ / dt)
                pm = [np.linalg.norm(rel[k] - prev["rel"][k]) for k in ("hl", "hr") if k in rel and k in prev["rel"]]
                if pm:
                    f["paw_motion"] = float(max(pm) / typ / dt)

        f.update(zone_features(det.box, kp, center, zonemap, t))

        f.update(body_point_zones(kp, zonemap, t))
        self.hist.append({"t": t, "cam": cam, "center": center, "rel": rel})
        return f
