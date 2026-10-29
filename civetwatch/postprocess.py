"""Step 3 · Post-processing (slide 14).

1. AI label per frame  -> majority vote over a 3 s window (6 frames)
2. 1-minute rule       -> clean Inactive / Active blocks = Start / Stop rows
3. count bouts         -> 1 trip to the tray = 1 count; pauses < 30 s stay in the bout
plus Not-visible rows, per-hour counts, summary numbers and keeper alerts.
"""
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd

from . import ethogram as E


# ------------------------------------------------------------ helpers ---
def runs(values):
    """Run-length encode -> list of (value, start_idx, end_idx_exclusive)."""
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((values[start], start, i))
            start = i
    return out


def _span_s(times, a, b, dt):
    return (times[b - 1] - times[a]).total_seconds() + dt


def _mode(series, default=""):
    s = series.dropna()
    s = s[s != ""]
    return s.mode().iloc[0] if len(s) else default


def majority_vote(labels, confs, window):
    """Each frame takes the conf-weighted majority label of the `window`
    frames around it, so one odd frame (walk 51%) is ignored."""
    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0:
        return labels
    names = sorted(set(labels))
    idx = np.array([names.index(l) for l in labels])
    w = np.clip(np.nan_to_num(np.asarray(confs, float)), 0.05, 1.0)
    onehot = np.zeros((len(labels), len(names)))
    onehot[np.arange(len(labels)), idx] = w
    kernel = np.ones(window)
    pad_l, pad_r = window // 2, window - 1 - window // 2
    padded = np.pad(onehot, ((pad_l, pad_r), (0, 0)))
    score = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(len(names))], axis=1)
    best = score.argmax(1)
    # keep the frame's own label on ties
    own = score[np.arange(len(labels)), idx]
    best = np.where(own >= score.max(1) - 1e-9, idx, best)
    return np.array([names[k] for k in best], dtype=object)


def min_duration_rule(states, times, min_s, dt):
    """The state only changes if the new state lasts at least `min_s`;
    shorter stripes are absorbed into the current state."""
    rs = runs(list(states))
    if not rs:
        return np.array([], dtype=object)
    long_runs = [r for r in rs if _span_s(times, r[1], r[2], dt) >= min_s]
    cur = long_runs[0][0] if long_runs else Counter(states).most_common(1)[0][0]
    out = np.empty(len(states), dtype=object)
    for val, a, b in rs:
        if val != cur and _span_s(times, a, b, dt) >= min_s:
            cur = val
        out[a:b] = cur
    return out


def at_place(seg, b, beh):
    """Feeding / drinking only when the civet stays at the bowl: its box moves
    less than max_travel body lengths over the whole bout (walking past the
    basin is not drinking), and for drinking the pose model must put the nose
    in the zone in at least min_pose of the samples."""
    rule = b.get("bout_filter", {}).get(beh)
    if not rule or "x1" not in seg:
        return True
    box = seg[["x1", "y1", "x2", "y2"]].astype(float).dropna()
    if len(box):
        cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
        size = max(1.0, float(np.maximum(box.x2 - box.x1, box.y2 - box.y1).median()))
        if np.hypot(cx.max() - cx.min(), cy.max() - cy.min()) / size > rule.get("max_travel", 9e9):
            return False
    if rule.get("min_pose") and "kp" in seg:
        if (seg["kp"].fillna("").astype(str) != "").mean() < rule["min_pose"]:
            return False
    return True


def split_rare(df, times, b, dt):
    """Toileting stops longer than defecation_min_s are defecation, shorter
    ones urination. A rare behavior that lasts far longer than it can (a civet
    resting with its rear on the log is not marking for 10 minutes) is put
    back to resting / active."""
    voted = df["voted"].to_numpy(dtype=object).copy()
    longest = b.get("max_duration_s", {})
    for lab, a, z in runs(list(voted)):
        span = _span_s(times, a, z, dt)
        if lab == E.URINATION and span >= b.get("defecation_min_s", 5):
            lab = E.DEFECATION
            voted[a:z] = lab
        if lab in longest and span > longest[lab]:
            still = df["motion"].iloc[a:z].mean() < b["still_motion"]
            voted[a:z] = E.LYING_DOWN if still else E.ACTIVE_OTHER
    return voted


# --------------------------------------------------------------- main ---
def postprocess(df, cfg):
    """df = per-sample records from the pipeline. Returns a dict of tables."""
    b, p, r = cfg["behavior"], cfg["postprocess"], cfg["report"]
    dt = 1.0 / cfg["sampling"]["fps"]
    df = df.sort_values("t").reset_index(drop=True)
    times = list(df["t"])
    df["voted"] = majority_vote(df["label"], df["conf"], b["vote_window"])
    df["voted"] = split_rare(df, times, b, dt)

    # ---- Inactive / Active with the 1-minute rule ----
    # short gaps (out of view, poor image) keep the last state; long ones are
    # 'Not visible' - never counted as resting or as active
    state = pd.Series([E.state_of(l) for l in df["voted"]], dtype=object)
    for val, a, z in runs([s is None for s in state]):
        if val and _span_s(times, a, z, dt) >= p["hold_max_s"]:
            state.iloc[a:z] = E.OUT_OF_VIEW
    state = state.ffill().bfill().fillna(E.OUT_OF_VIEW)
    df["state"] = min_duration_rule(state.tolist(), times, p["min_state_s"], dt)

    # rest type inside each inactive block, also with the 1-minute rule
    df["rest_type"] = ""
    for st, a, z in runs(list(df["state"])):
        if st != E.INACTIVE:
            continue
        rest = pd.Series([l if l in E.INACTIVE_LABELS else None for l in df["voted"].iloc[a:z]], dtype=object)
        rest = rest.ffill().bfill().fillna(E.LYING_DOWN)
        df.loc[a:z - 1, "rest_type"] = min_duration_rule(rest.tolist(), times[a:z], p["min_state_s"], dt)

    blocks = []
    for st, a, z in runs(list(df["state"])):
        seg = df.iloc[a:z]
        agree = seg[[E.state_of(l) == st for l in seg["voted"]]]
        blocks.append({
            "state": st, "start": times[a], "stop": times[z - 1] + timedelta(seconds=dt),
            "duration_s": _span_s(times, a, z, dt),
            "room": _mode(seg["room"]), "camera": _mode(seg["camera"]),
            "confidence": float(agree["conf"].mean()) if len(agree) else 0.0,
            "light": _mode(seg["light"]), "visibility": _mode(seg["visibility"]),
            "quality": _mode(seg["quality"]),
        })
    blocks = pd.DataFrame(blocks)

    # ---- duration events: rest types inside Inactive blocks, walking for Active ----
    duration_events = []
    for _, blk in blocks.iterrows():
        seg = df[(df["t"] >= blk["start"]) & (df["t"] < blk["stop"])]
        if blk["state"] == E.OUT_OF_VIEW:
            continue
        if blk["state"] == E.INACTIVE:
            labels = list(seg["rest_type"])
            for lab, a, z in runs(labels):
                sub = seg.iloc[a:z]
                duration_events.append(_event(sub, lab, "duration", dt, sub[sub["voted"] == lab]["conf"]))
        else:
            seen = seg[seg["camera"].fillna("") != ""]
            duration_events.append(_event(seg, E.WALKING, "duration", dt, seen["conf"]))

    # ---- Not visible rows: no civet in any camera for >= 10 s ----
    nv = (df["label"] == E.NOT_VISIBLE).to_numpy()
    nv_runs = [(a, z) for v, a, z in runs(list(nv)) if v]
    merged = []
    for a, z in nv_runs:
        if merged and (times[a] - times[merged[-1][1] - 1]).total_seconds() - dt <= p["not_visible_merge_s"]:
            merged[-1] = (merged[-1][0], z)
        else:
            merged.append((a, z))
    for a, z in merged:
        if _span_s(times, a, z, dt) >= p["not_visible_min_s"]:
            seg = df.iloc[a:z]
            duration_events.append(_event(seg, E.NOT_VISIBLE, "not_visible", dt, pd.Series([1.0])))

    # ---- frequency bouts ----
    freq_events = []
    for beh in E.FREQUENCY_LABELS:
        mask = (df["voted"] == beh).to_numpy()
        kept = [(a, z) for v, a, z in runs(list(mask))
                if v and _span_s(times, a, z, dt) >= b["min_duration_s"][beh] and at_place(df.iloc[a:z], b, beh)]
        bouts = []
        for a, z in kept:
            if bouts and (times[a] - times[bouts[-1][1] - 1]).total_seconds() - dt < b["bout_gap_s"][beh]:
                bouts[-1] = (bouts[-1][0], z)
            else:
                bouts.append((a, z))
        for a, z in bouts:
            seg = df.iloc[a:z]
            ev = _event(seg, beh, "frequency", dt, seg[seg["voted"] == beh]["conf"])
            freq_events.append(ev)

    events = pd.DataFrame(duration_events + freq_events)
    if len(events):
        events = events.sort_values("start").reset_index(drop=True)
        always = set(r["always_review"])
        events["needs_review"] = [
            (row["category"] == "frequency" and (row["confidence"] < r["review_conf"] or row["behavior"] in always))
            for _, row in events.iterrows()]
    else:
        events = pd.DataFrame(columns=["start", "stop", "duration_s", "behavior", "category", "room", "camera",
                                       "confidence", "light", "visibility", "quality", "note", "needs_review"])

    hourly = hourly_counts(events, df["t"].min(), df["t"].max(), r["sheet_start_hour"])
    summary = summarise(df, blocks, events, dt)
    alerts = make_alerts(df, events, summary, cfg)
    return {"frames": df, "blocks": blocks, "events": events, "hourly": hourly,
            "summary": summary, "alerts": alerts}


def _event(seg, behavior, category, dt, confs):
    times = list(seg["t"])
    return {
        "start": times[0], "stop": times[-1] + timedelta(seconds=dt),
        "duration_s": round(_span_s(times, 0, len(times), dt), 1),
        "behavior": behavior, "category": category,
        "room": _mode(seg["room"]), "camera": _mode(seg["camera"]),
        "confidence": round(float(confs.mean()) if len(confs) else 0.0, 3),
        "light": _mode(seg["light"]), "visibility": _mode(seg["visibility"]),
        "quality": _mode(seg["quality"]), "note": "",
    }


def hour_label(h, first):
    nh = (h + 1) % 24
    return f"{h}.00-{nh}.00" if first else f"{h}.01-{nh}.00"


def hourly_counts(events, t0, t1, start_hour):
    """Rows = hours of the observation, columns = the zoo sheet's frequency
    columns. Like the paper sheet, the first row and the start of the active
    period read 'h.00-h+1.00', later rows 'h.01-h+1.00'."""
    cols = ["Feeding", "Drinking", "Territorial marking", "Grooming", "Urine", "Feces"]
    rows = []
    cur = t0.replace(minute=0, second=0, microsecond=0)
    while cur <= t1:
        first = not rows or cur.hour == start_hour
        rows.append({"hour": hour_label(cur.hour, first), "start": cur, **{c: 0 for c in cols}})
        cur += timedelta(hours=1)
    table = pd.DataFrame(rows)
    colmap = {E.FEEDING: "Feeding", E.DRINKING: "Drinking", E.MARKING: "Territorial marking",
              E.GROOMING: "Grooming", E.URINATION: "Urine", E.DEFECATION: "Feces"}
    for _, ev in events[events["category"] == "frequency"].iterrows():
        col = colmap.get(ev["behavior"])
        hstart = ev["start"].replace(minute=0, second=0, microsecond=0)
        hit = table.index[table["start"] == hstart]
        if col and len(hit):
            table.loc[hit[0], col] += 1
    table["Total"] = table[cols].sum(axis=1)
    return table


def summarise(df, blocks, events, dt):
    total = len(df) * dt
    act = float(blocks.loc[blocks["state"] == E.ACTIVE, "duration_s"].sum()) if len(blocks) else 0.0
    inact = float(blocks.loc[blocks["state"] == E.INACTIVE, "duration_s"].sum()) if len(blocks) else 0.0
    seen = act + inact
    in_view = (df["camera"].fillna("") != "") | (df["label"] == E.SLEEP_IN_BOX)
    rooms = df["room"].where(in_view, "Not seen").replace("", "Not seen").fillna("Not seen")
    room_share = (rooms.value_counts() / max(1, len(rooms))).sort_index()
    freq = events[events["category"] == "frequency"] if len(events) else events
    counts = freq["behavior"].value_counts().to_dict() if len(freq) else {}
    rest = df.loc[df["state"] == E.INACTIVE, "rest_type"].value_counts() * dt
    return {
        "start": df["t"].min(), "end": df["t"].max() + timedelta(seconds=dt),
        "observed_s": total, "active_s": act, "inactive_s": inact,
        "active_frac": act / seen if seen else 0.0,        # share of the time the civet was seen
        "unseen_s": total - seen,
        "room_share": {k: float(v) for k, v in room_share.items()},
        "counts": {k: int(v) for k, v in counts.items()},
        "rest_types_s": {k: float(v) for k, v in rest.items()},
        "not_visible_s": float((df["label"] == E.NOT_VISIBLE).sum() * dt),
        "poor_image_frac": float((df["quality"] == "poor_image").mean()) if len(df) else 0.0,
        "night_frac": float((df["light"] == "night_ir").mean()) if len(df) else 0.0,
        "to_review": int(events["needs_review"].sum()) if len(events) else 0,
    }


def make_alerts(df, events, summary, cfg):
    r = cfg["report"]
    t0, t1 = summary["start"], summary["end"]
    alerts = []
    freq = events[events["category"] == "frequency"] if len(events) else events
    for rule in r.get("alerts", []):
        hh, mm = [int(x) for x in rule["after"].split(":")]
        thr = t0.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if thr < t0:
            thr += timedelta(days=1)
        if thr >= t1:
            continue
        later = freq[(freq["behavior"] == rule["behavior"]) & (freq["start"] >= thr)] if len(freq) else freq
        if len(later) == 0:
            alerts.append(rule["message"])
    if summary["not_visible_s"] / 60 >= r["not_visible_alert_min"]:
        alerts.append(f"Civet not visible for {summary['not_visible_s'] / 60:.0f} min in total, please check the cameras")
    if summary["poor_image_frac"] >= r["poor_image_alert_frac"]:
        alerts.append(f"{summary['poor_image_frac']:.0%} of frames had a poor image (blur / dark / frozen)")
    if summary["to_review"]:
        alerts.append(f"{summary['to_review']} event(s) to check by a person")
    return alerts
