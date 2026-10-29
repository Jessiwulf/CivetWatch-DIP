"""Keepers draw the fixed zones once per camera.

Left-click adds a point, right-click (or Enter) closes the polygon and asks
for its type with a key:  b = rest box   d = door   f = fence   l = log
t = tyre   o = food   w = water   x = latrine   i = ignore (clock text, lens dirt).   u = undo last point / zone,  s = save,
q = quit.
"""
import cv2
import numpy as np

from .overlay import draw_zones
from .video import resize_width
from .zones import load_zones, save_zones

KEYS = {"b": "rest_box", "d": "door", "f": "fence", "l": "log", "t": "tyre", "o": "food", "w": "water", "x": "latrine", "i": "ignore"}


def first_frame(video, at_s=5.0):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read a frame from {video}")
    return frame


def edit(image_or_video, out_path, camera, width=1100):
    img = cv2.imread(image_or_video)
    if img is None:
        img = first_frame(image_or_video)
    img = resize_width(img, width)
    h, w = img.shape[:2]
    zones = load_zones(out_path)
    points, waiting = [], False
    win = f"CivetWatch zones - {camera}"

    def on_mouse(event, x, y, flags, param):
        nonlocal waiting
        if waiting:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
            waiting = True

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = img.copy()
        polys = [(z["type"], z.get("name", z["type"]),
                  (np.array(z["points"]) * [w, h]).astype(np.int32)) for z in zones]
        draw_zones(view, polys)
        for i, p in enumerate(points):
            cv2.circle(view, p, 4, (0, 255, 255), -1)
            if i:
                cv2.line(view, points[i - 1], p, (0, 255, 255), 2)
        msg = ("zone type?  b=rest box d=door f=fence l=log t=tyre o=food w=water x=latrine i=ignore" if waiting else
               "click points, right-click/Enter = close   u=undo  s=save  q=quit")
        cv2.rectangle(view, (0, h - 28), (w, h), (30, 30, 30), -1)
        cv2.putText(view, msg, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        ch = chr(key).lower() if key < 128 else ""
        if waiting and ch in KEYS:
            zones.append({"type": KEYS[ch], "name": KEYS[ch].upper().replace("_", " "),
                          "points": [[round(x / w, 4), round(y / h, 4)] for x, y in points]})
            points.clear()
            waiting = False
        elif waiting and key == 27:
            waiting = False
        elif key == 13 and len(points) >= 3:
            waiting = True
        elif ch == "u":
            if points:
                points.pop()
            elif zones:
                zones.pop()
        elif ch == "s":
            save_zones(out_path, camera, zones)
            print(f"saved {len(zones)} zones -> {out_path}")
        elif ch == "q":
            break
    cv2.destroyWindow(win)
    return zones
