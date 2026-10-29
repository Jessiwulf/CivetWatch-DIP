"""Draw the 4 layers on a frame: zones, civet box + ID, body dots, behavior."""
import cv2
import numpy as np

from . import ethogram as E
from .zones import ZONE_COLORS

BOX = (0, 190, 255)            # amber, like the deck
LIMB = (230, 210, 40)
KP_COLORS = {"nose": (255, 255, 255), "left_ear": (255, 255, 255), "right_ear": (255, 255, 255),
             "withers": (255, 255, 255), "hips": (255, 255, 255),
             "front_left_paw": (90, 220, 90), "front_right_paw": (90, 220, 90),
             "hind_left_paw": (90, 220, 90), "hind_right_paw": (90, 220, 90),
             "tail_base": (80, 80, 240), "tail_tip": (80, 80, 240)}


def label_box(img, text, org, bg, fg=(20, 20, 20), scale=0.5):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    y = max(h + 4, y)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + base - 2), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def draw_zones(img, polygons, alpha=0.25):
    polygons = [p for p in polygons if p[0] != "ignore"]     # keep the burned-in clock readable
    layer = img.copy()
    for ztype, name, poly in polygons:
        cv2.fillPoly(layer, [poly], ZONE_COLORS.get(ztype, (200, 200, 200)))
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)
    for ztype, name, poly in polygons:
        color = ZONE_COLORS.get(ztype, (200, 200, 200))
        cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
        x, y = poly[:, 0].min(), poly[:, 1].min()
        label_box(img, name.upper().replace("_", " "), (x, y), color, (255, 255, 255), 0.42)


def draw_keypoints(img, kp):
    if kp is None:
        return
    kp = np.asarray(kp, float)
    for a, b in E.SKELETON:
        pa, pb = kp[E.KP[a]], kp[E.KP[b]]
        if not (np.isnan(pa[:2]).any() or np.isnan(pb[:2]).any()):
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), LIMB, 2, cv2.LINE_AA)
    for name, i in E.KP.items():
        x, y = kp[i][:2]
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(img, (int(x), int(y)), 4, KP_COLORS[name], -1, cv2.LINE_AA)


def draw_record(img, rec, polygons=None):
    """rec: one row of frames.csv (dict-like)."""
    if polygons:
        draw_zones(img, polygons)
    if rec.get("x1") == rec.get("x1") and rec.get("x1") is not None:     # not NaN
        x1, y1, x2, y2 = [int(rec[k]) for k in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX, 2)
        label_box(img, f"civet ID 1 {rec.get('det_conf', 0):.2f}", (x1, y1), BOX)
        draw_keypoints(img, rec.get("kp"))
    lab = rec.get("voted") or rec.get("label")
    if lab:
        st = E.state_of(lab)
        text = E.DISPLAY.get(lab, lab).upper() + (f" · {st}" if st else "")
        h = img.shape[0]
        label_box(img, text.replace("·", "-"), (10, h - 10), (60, 160, 60), (255, 255, 255), 0.6)
    t = rec.get("t")
    if t is not None:
        label_box(img, f"{rec.get('room', '')} {rec.get('camera', '')}  {t:%Y-%m-%d %H:%M:%S}",
                  (10, 24), (40, 60, 30), (255, 255, 255), 0.55)
    return img
