"""Layer 4 · Name the behavior.

Zone + timer rules handle rest, walk, eat and drink (slides 8-9). A small
sequence model reads 4 s of body dots for grooming, toileting and marking:
the trained GRU when models/civet_gru.pt exists, else hand-written cues.
"""
import os

import numpy as np

from . import ethogram as E
from .config import resolve
from .features import FEATURES


class FrameRules:
    """One sample -> (label, confidence). Durations are enforced later
    (min_duration_s in postprocess), so these are per-frame clues."""

    def __init__(self, cfg):
        self.b = cfg["behavior"]

    def __call__(self, f, det_conf):
        b = self.b
        pose = f["kp_frac"] > 0
        still = f["speed"] < b["still_speed"]
        if f["speed"] >= b["walk_speed"]:
            return E.WALKING, min(0.99, det_conf * (0.7 + min(f["speed"], 1.0) * 0.3))

        # head_down only means something from a side view; the zoo cameras look down, so it is optional
        head_down = (f["head_drop"] > b["head_down"]) if pose and b.get("head_down") is not None else True
        for zone, label in (("in_food", E.FEEDING), ("in_water", E.DRINKING)):
            inside = f[zone] >= (1.0 if pose else b["tray_overlap"])
            if inside and head_down and f["speed"] < b["walk_speed"] and f["motion"] > b["still_motion"] * 0.5:
                return label, det_conf * (0.95 if pose else 0.8)

        if still and f["motion"] < b["still_motion"]:
            return self.rest_type(f, pose), det_conf * 0.9
        return E.ACTIVE_OTHER, det_conf * 0.75

    def rest_type(self, f, pose):
        b = self.b
        if f["in_rest_box"] >= 0.5:
            return E.SLEEP_IN_BOX
        if f["in_tyre"] >= 0.5 or f["compactness"] < b["curl_compactness"] or (pose and f["nose_hips"] < 0.35):
            return E.CURLING_UP
        if (f["compactness"] > b["stretch_compactness"] or f["aspect"] > b["stretch_aspect"]
                or (pose and f["paw_spread"] > 0.6)):
            return E.STRETCHED_OUT
        return E.LYING_DOWN


class HeuristicSequence:
    """Hand-written cues over the last 4 s, used until the GRU is trained.

    grooming   body stays in place, lots of movement inside the box, nose
               turned back to its own body (needs body points; box-only only
               with groom_without_pose)
    marking    rear end (tail base) on a log / door / hollow log while the
               nose is away from it, small rubbing movement
    toileting  rear end in a 'latrine' zone, body still; postprocess splits
               short stops (urination) from long ones (defecation)
    Rare behaviors are always flagged for a keeper's check.
    """

    name = "heuristic"

    def __init__(self, cfg):
        self.b = cfg["behavior"]
        self.idx = {k: i for i, k in enumerate(FEATURES)}

    def __call__(self, window):
        """window: (W, F) array, oldest first -> (label or None, prob)."""
        b, i = self.b, self.idx
        vis = window[:, i["visible"]]
        if vis.mean() < 0.75:
            return None, 0.0
        w = window[vis > 0]
        col = lambda k: w[:, i[k]]
        posed = col("kp_frac") > 0
        pose = posed.mean() >= 0.5
        speed, motion = col("speed").mean(), col("motion").mean()
        in_tray = max(col("in_food").mean(), col("in_water").mean())
        scores = {}
        if col("at_edge").mean() > 0.25:          # half out of the picture: body points are guesses
            return None, 0.0
        if not b.get("pose_rare_rules", True) and not b.get("groom_without_pose"):
            return None, 0.0

        # toileting: rear end in a latrine, body nearly still
        if pose and col("rear_latrine")[posed].mean() >= 0.75 and speed < b["walk_speed"]:
            scores[E.URINATION] = 0.7

        # marking: rear on an object, nose away from it, small movement
        if pose and speed < b["walk_speed"] and motion > b["still_motion"]:
            rear_on = col("rear_obj")[posed].mean()
            nose_off = 1.0 - col("nose_obj")[posed].mean()
            if rear_on >= 0.6 and nose_off >= 0.6:
                scores[E.MARKING] = 0.6 + 0.3 * rear_on * nose_off

        # grooming: in place, busy inside the box, head turned back to the body
        if speed < b["walk_speed"] and motion > b["groom_motion"] and in_tray < b["tray_overlap"]:
            s = 0.62 if b.get("groom_without_pose") else 0.0
            if pose:
                nose_body = col("nose_hips")[posed]
                if (nose_body < b["groom_nose_body"]).mean() >= 0.5:
                    s = 0.75 + min(0.2, col("nose_motion")[posed].mean() * 0.2)
            if s:
                scores[E.GROOMING] = s
        if not scores:
            return None, 0.0
        label = max(scores, key=scores.get)
        return label, float(scores[label])


class GRUSequence:
    """Trained GRU (see civetwatch.gru)."""

    name = "gru"

    def __init__(self, path):
        from .gru import load_model
        self.model, self.meta = load_model(path)
        self.order = [FEATURES.index(f) for f in self.meta["features"]]

    def __call__(self, window):
        from .gru import predict
        probs = predict(self.model, self.meta, window[:, self.order])
        k = int(np.argmax(probs))
        label = self.meta["classes"][k]
        return (None if label == "other" else label), float(probs[k])


def build_sequence_model(cfg):
    path = resolve(cfg["behavior"]["gru_weights"])
    if os.path.exists(path):
        return GRUSequence(path)
    return HeuristicSequence(cfg)


class BehaviorNamer:
    """Combines the frame rules with the sequence model."""

    def __init__(self, cfg):
        self.rules = FrameRules(cfg)
        self.seq = build_sequence_model(cfg)
        self.window = cfg["behavior"]["gru_window"]
        self.min_prob = cfg["behavior"]["gru_min_prob"]
        self.buf = []

    def skip(self):
        """A sample where the civet is not seen: keeps the 4 s window honest."""
        self.buf.append([0.0] * len(FEATURES))
        self.buf = self.buf[-self.window:]

    def __call__(self, f, det_conf):
        """-> (label, conf, seq_label, seq_prob)."""
        self.buf.append([f[k] for k in FEATURES])
        self.buf = self.buf[-self.window:]
        label, conf = self.rules(f, det_conf)
        seq_label, seq_prob = None, 0.0
        if len(self.buf) == self.window:
            seq_label, seq_prob = self.seq(np.asarray(self.buf, np.float32))
        if seq_label and seq_prob >= self.min_prob:
            walking_ok = seq_label == E.URINATION      # urination can happen while walking
            if label not in (E.FEEDING, E.DRINKING) and (label != E.WALKING or walking_ok):
                label, conf = seq_label, seq_prob * max(det_conf, 0.5)
        return label, float(conf), seq_label, float(seq_prob)
