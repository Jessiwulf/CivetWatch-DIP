"""Map the enclosure: fixed zones drawn once per camera + the movable tray zone.

Zones are saved as JSON with coordinates normalised to 0..1, so they work at
any processing resolution:

{"camera": "cam1", "zones": [{"type": "rest_box", "name": "REST BOX",
                              "points": [[0.42, 0.10], [0.70, 0.12], ...]}]}
"""
import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .ethogram import ZONE_TYPES

ZONE_COLORS = {          # BGR, matching the deck
    "rest_box": (90, 200, 120),
    "door": (220, 140, 60),
    "fence": (200, 80, 220),
    "log": (60, 120, 180),
    "tyre": (160, 160, 160),
    "food": (40, 140, 240),
    "water": (230, 180, 40),
    "latrine": (60, 60, 160),
    "ignore": (90, 90, 90),
}
DYNAMIC_HOLD_S = 600     # keep the last tray position while the civet's head hides it


class ZoneMap:
    def __init__(self, path=None, zones=None):
        self.zones = zones if zones is not None else (load_zones(path) if path else [])
        self.size = None
        self._px = []
        self.dynamic = {}       # type -> (polygon px, last_seen_t)

    def set_size(self, w, h):
        if self.size == (w, h):
            return
        self.size = (w, h)
        self._px = [(z["type"], z.get("name", z["type"]),
                     (np.array(z["points"], np.float32) * [w, h]).astype(np.int32))
                    for z in self.zones]

    def update_dynamic(self, ztype, boxes, t, grow=0.15):
        """Tray found by the detector this frame -> its zone follows it."""
        if not boxes:
            return
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
        poly = np.array([[x1 - gx, y1 - gy], [x2 + gx, y1 - gy], [x2 + gx, y2 + gy], [x1 - gx, y2 + gy]], np.int32)
        self.dynamic[ztype] = (poly, t)

    def polygons(self, t=None):
        """All active zones as (type, name, polygon px). A tray seen by the
        detector replaces the hand-drawn tray zone of the same type."""
        live = {}
        for ztype, (poly, seen) in self.dynamic.items():
            if t is None or (t - seen).total_seconds() <= DYNAMIC_HOLD_S:
                live[ztype] = poly
        out = [(zt, name, poly) for zt, name, poly in self._px if zt not in live]
        out += [(zt, zt.upper() + " (detected)", poly) for zt, poly in live.items()]
        return out

    def zones_at(self, point, t=None):
        if point is None:
            return set()
        p = (float(point[0]), float(point[1]))
        return {zt for zt, _, poly in self.polygons(t) if cv2.pointPolygonTest(poly, p, False) >= 0}

    def box_overlap(self, box, t=None):
        """Fraction of the box area inside each zone type."""
        x1, y1, x2, y2 = [int(v) for v in box]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        out = {}
        for zt, _, poly in self.polygons(t):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [poly - [x1, y1]], 1)
            frac = mask.mean()
            out[zt] = max(out.get(zt, 0.0), float(frac))
        return out


def load_zones(path):
    if not path or not os.path.exists(path):
        return []
    if path.lower().endswith(".xml"):
        return import_cvat(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("zones", data if isinstance(data, list) else [])


def save_zones(path, camera, zones):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"camera": camera, "zones": zones}, f, indent=2)


def _zone_type(label):
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"restbox": "rest_box", "sleeping_box": "rest_box", "box": "rest_box",
               "food_water": "food", "food_tray": "food", "water_tray": "water", "door_frame": "door"}
    key = aliases.get(key, key)
    return key if key in ZONE_TYPES else None


def import_cvat(xml_path, image_name=None):
    """Read polygons from a CVAT 'CVAT for images 1.1' export (first image, or
    the named one). Labels such as 'REST BOX', 'door', 'fence' become zone types."""
    root = ET.parse(xml_path).getroot()
    for img in root.iter("image"):
        if image_name and os.path.basename(img.get("name")) != image_name:
            continue
        w, h = float(img.get("width")), float(img.get("height"))
        zones = []
        for shape in img:
            zt = _zone_type(shape.get("label", ""))
            if zt is None:
                continue
            if shape.tag == "polygon":
                pts = [[float(a) / w, float(b) / h] for a, b in
                       (p.split(",") for p in shape.get("points").split(";"))]
            elif shape.tag == "box":
                x1, y1 = float(shape.get("xtl")) / w, float(shape.get("ytl")) / h
                x2, y2 = float(shape.get("xbr")) / w, float(shape.get("ybr")) / h
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                continue
            zones.append({"type": zt, "name": shape.get("label"), "points": pts})
        return zones
    return []
