"""python -m pytest tests  (or python tests/test_postprocess.py)"""
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from civetwatch import ethogram as E
from civetwatch.config import load_config
from civetwatch.features import FEATURES
from civetwatch.postprocess import majority_vote, min_duration_rule, postprocess

T0 = datetime(2026, 7, 20, 15, 30)


def frames_from(script):
    """script: [(label, seconds)] -> per-sample frames at 2 fps."""
    rows, t = [], T0
    for label, secs in script:
        for _ in range(int(secs * 2)):
            rows.append({"t": t, "label": label, "conf": 0.9, "room": "R1", "camera": "cam1",
                         "light": "night_ir", "visibility": "visible", "quality": "clear",
                         **{f: 0.0 for f in FEATURES}})
            t += timedelta(seconds=0.5)
    return pd.DataFrame(rows)


def test_vote_ignores_one_odd_frame():
    labels = [E.FEEDING] * 3 + [E.WALKING] + [E.FEEDING] * 2
    voted = majority_vote(labels, [0.92, 0.88, 0.95, 0.51, 0.9, 0.93], 6)
    assert list(voted) == [E.FEEDING] * 6


def test_one_minute_rule_absorbs_short_stripes():
    times = [T0 + timedelta(seconds=i * 0.5) for i in range(600)]
    states = [E.INACTIVE] * 200 + [E.ACTIVE] * 10 + [E.INACTIVE] * 190 + [E.ACTIVE] * 200
    out = min_duration_rule(states, times, 60, 0.5)
    assert list(out[:400]) == [E.INACTIVE] * 400
    assert list(out[400:]) == [E.ACTIVE] * 200


def test_feeding_bouts_merge_short_pauses():
    cfg = load_config()
    df = frames_from([(E.LYING_DOWN, 120), (E.FEEDING, 40), (E.ACTIVE_OTHER, 5), (E.FEEDING, 30),
                      (E.WALKING, 180), (E.FEEDING, 20), (E.WALKING, 60)])
    res = postprocess(df, cfg)
    feeding = res["events"][res["events"]["behavior"] == E.FEEDING]
    assert len(feeding) == 2                       # eat 40 s, look up 5 s, eat 30 s = ONE bout
    assert res["hourly"]["Feeding"].sum() == 2
    states = list(res["blocks"]["state"])
    assert states == [E.INACTIVE, E.ACTIVE]


def test_not_visible_rows_need_10_seconds():
    cfg = load_config()
    df = frames_from([(E.WALKING, 90), (E.NOT_VISIBLE, 5), (E.WALKING, 30), (E.NOT_VISIBLE, 25), (E.WALKING, 60)])
    res = postprocess(df, cfg)
    nv = res["events"][res["events"]["behavior"] == E.NOT_VISIBLE]
    assert len(nv) == 1 and 20 <= nv.iloc[0]["duration_s"] <= 30


def _window(**vals):
    import numpy as np
    from civetwatch.rules import HeuristicSequence
    row = {f: 0.0 for f in FEATURES}
    row.update({"visible": 1.0, "kp_frac": 0.8, "speed": 0.02, "nose_hips": 1.0})
    row.update(vals)
    cfg = load_config()
    cfg["behavior"]["pose_rare_rules"] = True             # the rules themselves (off by default)
    return HeuristicSequence(cfg)(np.array([[row[f] for f in FEATURES]] * 8, np.float32))


def test_sequence_rules():
    assert _window(motion=0.15, nose_hips=0.3)[0] == E.GROOMING          # head turned to body, busy
    assert _window(motion=0.15)[0] is None                              # busy but head away: not grooming
    assert _window(motion=0.08, rear_obj=1.0)[0] == E.MARKING           # rear on the log, nose away
    assert _window(motion=0.08, rear_obj=1.0, nose_obj=1.0)[0] is None  # sniffing the log
    assert _window(rear_latrine=1.0)[0] == E.URINATION                  # still in the latrine
    assert _window(rear_latrine=1.0, kp_frac=0.0)[0] is None            # no body points: no toileting


def test_toilet_split_by_duration():
    cfg = load_config()
    df = frames_from([(E.WALKING, 60), (E.URINATION, 3), (E.WALKING, 60), (E.URINATION, 12), (E.WALKING, 60),
                      (E.MARKING, 300), (E.WALKING, 60)])
    df.loc[df["label"] == E.MARKING, "motion"] = 0.0
    ev = postprocess(df, cfg)["events"]
    counts = ev[ev["category"] == "frequency"]["behavior"].value_counts().to_dict()
    assert counts == {E.URINATION: 1, E.DEFECATION: 1}, counts           # 5-min "marking" is resting


def test_stop_finder():
    from civetwatch.stops import find_stops, stop_events
    cfg = load_config()
    df = frames_from([(E.WALKING, 20), (E.ACTIVE_OTHER, 8), (E.WALKING, 20),        # 8 s stop between walks: found
                      (E.ACTIVE_OTHER, 120), (E.WALKING, 20),                       # 2 min: resting, not a stop
                      (E.ACTIVE_OTHER, 6), (E.WALKING, 20),                         # at the water bowl: drinking
                      (E.ACTIVE_OTHER, 1), (E.WALKING, 20)])                        # 1 s: too short
    df["speed"] = [0.5 if l == E.WALKING else 0.0 for l in df["label"]]
    df[["x1", "y1", "x2", "y2", "det_conf"]] = [100, 100, 200, 160, 0.8]
    df["x1"] = df["x1"] + df["speed"].cumsum() * 50 * 0.5        # the box moves while walking
    df["x2"] = df["x1"] + 100
    seg = (df["label"] != df["label"].shift()).cumsum()
    df.loc[seg == seg.unique()[5], "in_water"] = 1.0
    stops = find_stops(df, cfg)
    assert len(stops) == 1 and 7 <= stops.iloc[0]["duration_s"] <= 9, stops
    stops["vlm_letter"], stops["vlm_answer"], stops["grid"] = "A", "A) squatting", "stops/x.jpg"
    ev = stop_events(stops)
    assert ev.iloc[0]["behavior"] == "possible_toileting" and ev.iloc[0]["needs_review"]
    stops["vlm_letter"] = "E"
    assert not stop_events(stops).iloc[0]["needs_review"]                        # standing: listed, not flagged
    stops["vlm_letter"] = "F"
    assert len(stop_events(stops)) == 0                                           # no animal: dropped
    res = postprocess(df, cfg)                                                    # stops never reach the sheet
    assert res["hourly"][["Urine", "Feces"]].to_numpy().sum() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
