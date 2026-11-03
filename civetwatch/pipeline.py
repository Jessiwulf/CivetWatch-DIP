"""From 4 camera recordings to data and charts (slide 11):

0 video in -> 1 prepare -> 2 find + follow -> 3 read body -> 4 name behavior
-> 5 clean + count -> events.csv · sheet.xlsx · charts.png · clips/
"""
import json
import os
import time
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd

from . import ethogram as E
from .charts import make_charts
from .clips import make_clips
from .config import load_manifest
from .detect import MotionDetector, build_detector, build_pose
from .features import FEATURES, FeatureExtractor, at_edge, body_point_zones, zone_features
from .fusion import CivetFusion
from .jobs import status
from .postprocess import postprocess
from .resources import Throttle, settings as resource_settings
from .report import write_csvs, write_sheet, write_summary
from .rules import BehaviorNamer
from .video import CameraReader, Preprocessor, sample_times
from .zones import ZoneMap


def _prime_backgrounds(detector, cams, cfg, t0, t1):
    """Empty-room background per camera and light mode for the motion fallback:
    the median of frames spread over the day (the civet moves, the room doesn't)."""
    n = cfg["detector"]["motion"]["background_samples"]
    pre = Preprocessor(cfg)
    for cam in cams:
        reader = CameraReader(cam)
        span = (t1 - t0).total_seconds()
        grays = {}
        for k in range(n):
            frame, _ = reader.read_at(t0 + timedelta(seconds=span * (k + 0.5) / n))
            if frame is not None:
                _, gray, cond = pre(cam["id"] + "#bg", frame)
                grays.setdefault(cond["light"], []).append(gray)
        reader.close()
        for light, frames in grays.items():
            if len(frames) >= 3:
                detector.set_background(cam["id"], light, frames)
        if cam.get("background"):
            import cv2
            img = cv2.imread(cam["background"])
            if img is not None:
                _, gray, cond = pre(cam["id"] + "#bgimg", img)
                detector.set_background(cam["id"], cond["light"], [gray])


def analyse(manifest_path, cfg, start=None, end=None, progress=True):
    """Steps 1-4: one record per sample time (2 per second)."""
    man = load_manifest(manifest_path)
    cams = man["cameras"]
    readers = {c["id"]: CameraReader(c) for c in cams}
    cover = [r.coverage() for r in readers.values() if r.files]
    if not cover:
        raise ValueError("No readable video in the manifest")
    t0 = start or man.get("start") or min(c[0] for c in cover)
    t1 = end or man.get("end") or max(c[1] for c in cover)

    ids = [c["id"] for c in cams]
    rooms = {c["id"]: c["room"] for c in cams}
    zonemaps = {c["id"]: ZoneMap(c["zones"]) for c in cams}
    detector = build_detector(cfg, ids)
    if isinstance(detector, MotionDetector):
        _prime_backgrounds(detector, cams, cfg, t0, t1)
    pose = build_pose(cfg)
    pre = Preprocessor(cfg)
    fusion = CivetFusion(cfg)
    feats = FeatureExtractor(cfg)
    namer = BehaviorNamer(cfg)
    info = {"detector": detector.name, "pose": pose.name if pose else "none",
            "sequence_model": namer.seq.name, "cameras": ids}
    if progress:
        print(f"Analysing {t0:%Y-%m-%d %H:%M:%S} -> {t1:%H:%M:%S}  |  detector={info['detector']}  "
              f"pose={info['pose']}  sequence={info['sequence_model']}")

    rows, prev_gray = [], {}
    last_room, last_cam, last_zone_rest = "", None, False
    throttle = Throttle(log=print if progress else None)
    tick, n_total = time.time(), int((t1 - t0).total_seconds() * cfg["sampling"]["fps"])
    for n, t in enumerate(sample_times(t0, t1, cfg["sampling"]["fps"])):
        per_cam, conds, grays, frames = {}, {}, {}, {}
        batched = getattr(detector, "batch", False)
        for cid in ids:
            frame, status = readers[cid].read_at(t)
            if frame is None:
                conds[cid] = {"light": "", "quality": "poor_image" if status == "dropped" else "no_video",
                              "quality_reason": status}
                continue
            pf, gray, cond = pre(cid, frame)
            h, w = gray.shape
            zonemaps[cid].set_size(w, h)
            if hasattr(detector, "set_ignore") and cid not in detector.ignore:
                detector.set_ignore(cid, [p for zt, _, p in zonemaps[cid].polygons() if zt == "ignore"], gray.shape)
            conds[cid], grays[cid], frames[cid] = cond, gray, pf
            if not batched:
                per_cam[cid] = detector(cid, pf, gray, cond["light"])
        if batched:
            per_cam = detector.detect_many(frames)
        for cid, dets in per_cam.items():
            for ztype in ("food", "water"):
                zonemaps[cid].update_dynamic(ztype, [d.box for d in dets if d.cls == ztype], t)

        cid, det = fusion.select(per_cam)
        row = {"t": t}
        if det is not None:
            cond = conds[cid]
            kp = pose(frames[cid], det.box) if pose else None
            f = feats(t, cid, det, kp, grays[cid], prev_gray.get(cid), zonemaps[cid])
            if cond["quality"] == "poor_image":
                namer.skip()
                label, conf, seq_label, seq_prob = E.UNKNOWN, 0.0, None, 0.0
            else:
                label, conf, seq_label, seq_prob = namer(f, det.conf)
            last_room, last_cam, last_zone_rest = rooms[cid], cid, f["in_rest_box"] >= cfg["behavior"]["rest_box_enter"]
            visibility = "part_hidden" if kp is not None and f["kp_frac"] < 0.5 else "visible"
            trays = {zt: [round(float(v), 1) for v in poly.min(0).tolist() + poly.max(0).tolist()]
                     for zt, (poly, _) in zonemaps[cid].dynamic.items()}
            row.update({
                "camera": cid, "room": rooms[cid], "det_conf": round(det.conf, 3),
                "x1": round(det.box[0], 1), "y1": round(det.box[1], 1),
                "x2": round(det.box[2], 1), "y2": round(det.box[3], 1),
                "kp": json.dumps(np.round(kp, 1).tolist()) if kp is not None else "",
                "trays": json.dumps(trays) if trays else "",
                "zones": ",".join(sorted(z[3:] for z in f if z.startswith("in_") and f[z] >= 0.5)),
                "light": cond["light"], "quality": cond["quality"], "quality_reason": cond["quality_reason"],
                "visibility": visibility, "label": label, "conf": round(conf, 3),
                "seq_label": seq_label or "", "seq_prob": round(seq_prob, 3),
            })
            row.update({k: round(v, 4) for k, v in f.items()})
        else:
            f = feats.missing()
            namer.skip()
            lights = [c["light"] for c in conds.values() if c["light"]]
            # a poor picture where the civet was last seen may be hiding it
            watch = [conds[last_cam]] if last_cam in conds else list(conds.values())
            any_poor = any(c["quality"] == "poor_image" for c in watch)
            if last_zone_rest:
                label, vis = E.SLEEP_IN_BOX, "in_rest_box"     # last seen going into the box
            elif any_poor:
                label, vis = E.UNKNOWN, "poor_image"          # cannot tell: never 'resting'
            else:
                label, vis = E.NOT_VISIBLE, "out_of_view"
            row.update({
                "camera": "", "room": last_room, "det_conf": 0.0, "kp": "", "trays": "", "zones": "",
                "light": Counter(lights).most_common(1)[0][0] if lights else "",
                "quality": "poor_image" if any_poor else "clear",
                "quality_reason": ";".join(sorted({c["quality_reason"] for c in conds.values() if c["quality_reason"]})),
                "visibility": vis, "label": label, "conf": 0.5 if label == E.SLEEP_IN_BOX else 0.0,
                "seq_label": "", "seq_prob": 0.0,
            })
            row.update(f)
        prev_gray = grays
        rows.append(row)
        throttle.tick()
        if n % 120 == 0:
            status(None, None, sub=(n + 1) / max(1, n_total), rate=(n + 1) / max(1e-6, time.time() - tick))
        if progress and (n % 600 == 0 or n == n_total - 1):
            rate = (n + 1) / max(1e-6, time.time() - tick)
            eta = (n_total - n - 1) / max(rate, 1e-6)
            print(f"  {t:%H:%M:%S}  {n + 1}/{n_total} samples  {rate:.1f} samples/s  ETA {eta / 60:.1f} min"
                  f"  (cooling pauses {throttle.paused_s / 60:.0f} min)", flush=True)

    for r in readers.values():
        r.close()
    df = pd.DataFrame(rows)
    for col in FEATURES:
        df[col] = df[col].astype(float)
    return df, cams, info


def relabel(frames, cfg, cams):
    """Re-apply the current zones and behavior rules to an analysed day
    (boxes, body dots and motion are kept from frames.csv), so zones and
    thresholds can be tuned in minutes instead of re-reading the video."""
    zonemaps = {}
    for cam in cams:
        reader = CameraReader(cam)
        t0, _ = reader.coverage()
        frame, _ = reader.read_at(t0 + timedelta(seconds=1)) if t0 else (None, None)
        reader.close()
        if frame is None:
            continue
        h, w = Preprocessor(cfg)(cam["id"], frame)[1].shape
        zm = ZoneMap(cam["zones"])
        zm.set_size(w, h)
        zonemaps[cam["id"]] = zm
    namer = BehaviorNamer(cfg)
    frames = frames.copy()
    for i, row in frames.iterrows():
        cid = row.get("camera") or ""
        if not cid or cid not in zonemaps:
            namer.skip()
            continue
        zm = zonemaps[cid]
        if isinstance(row.get("trays"), str) and row["trays"]:
            for ztype, box in json.loads(row["trays"]).items():
                zm.update_dynamic(ztype, [box], row["t"])
        box = (row["x1"], row["y1"], row["x2"], row["y2"])
        kp = row.get("kp")
        kp = np.array(json.loads(kp), np.float32) if isinstance(kp, str) and kp else None
        if kp is not None and kp.shape[0] != len(E.KEYPOINTS):
            kp = None
        body = [kp[E.KP[n], :2] for n in ("withers", "hips")] if kp is not None else []
        body = [p for p in body if not np.isnan(p).any()]
        center = np.mean(body, axis=0) if body else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        f = {k: float(row.get(k, 0.0) or 0.0) for k in FEATURES}
        f.update(zone_features(box, kp, center, zm, row["t"]))
        f.update(body_point_zones(kp, zm, row["t"]))
        f["at_edge"] = at_edge(box, (zm.size[1], zm.size[0]))
        for k in list(zone_features(box, None, center, zm, row["t"])) + ["rear_obj", "nose_obj", "rear_latrine", "at_edge"]:
            frames.at[i, k] = f[k]
        if row.get("quality") == "poor_image":
            namer.skip()
            continue
        label, conf, seq_label, seq_prob = namer(f, float(row["det_conf"]))
        frames.at[i, "label"], frames.at[i, "conf"] = label, round(conf, 3)
        frames.at[i, "seq_label"], frames.at[i, "seq_prob"] = seq_label or "", round(seq_prob, 3)
    return frames


def _span(manifest_path, start, end):
    man = load_manifest(manifest_path)
    cover = [CameraReader(c).coverage() for c in man["cameras"]]
    cover = [c for c in cover if c[0] is not None]
    if not cover:
        raise ValueError("No readable video in the manifest")
    t0 = start or man.get("start") or min(c[0] for c in cover)
    t1 = end or man.get("end") or max(c[1] for c in cover)
    return t0, t1


def run(manifest_path, cfg, out_dir, start=None, end=None, clips=True, progress=True, fresh=False):
    """Analyses the day in chunks (resources.chunk_min, 1 hour) and saves each
    one in out_dir/parts/. Run the same command again after a crash or Ctrl+C
    and it continues from the first unfinished hour."""
    os.makedirs(out_dir, exist_ok=True)
    t0, t1 = _span(manifest_path, start, end)
    step = timedelta(minutes=resource_settings()["chunk_min"])
    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    pieces, info, cams = [], None, load_manifest(manifest_path)["cameras"]
    a = t0
    n_chunks = max(1, int(np.ceil((t1 - t0) / step)))
    while a < t1:
        b = min(a + step, t1)
        k = len(pieces)
        status(f"Analysing {a:%H:%M}-{b:%H:%M} (part {k + 1} of {n_chunks})", k / n_chunks, weight=1 / n_chunks)
        path = os.path.join(parts_dir, f"frames_{a:%Y%m%d_%H%M%S}_{b:%H%M%S}.pkl")
        if os.path.exists(path) and not fresh:
            if progress:
                print(f"{a:%H:%M}-{b:%H:%M} already analysed (saved) - skipping")
            df = pd.read_pickle(path)
        else:
            df, cams, info = analyse(manifest_path, cfg, a, b, progress)
            df.to_pickle(path + ".tmp")
            os.replace(path + ".tmp", path)          # only complete chunks are ever saved
            with open(os.path.join(parts_dir, "info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f)
        pieces.append(df)
        a = b
    status("Analysis done", 1.0)
    if info is None:
        with open(os.path.join(parts_dir, "info.json"), encoding="utf-8") as f:
            info = json.load(f)
    frames = pd.concat(pieces, ignore_index=True)
    return report(frames, cams, info, cfg, out_dir, clips=clips, progress=progress)


def report(frames, cams, info, cfg, out_dir, clips=True, progress=True):
    """Step 5 and all output files, from per-sample records."""
    os.makedirs(out_dir, exist_ok=True)
    result = postprocess(frames, cfg)
    if cfg.get("stops", {}).get("enabled") and cams and len(frames):
        status("Finding short stops and making pictures", None)
        add_stop_events(result, cams, cfg, out_dir, progress)
    apply_event_review(result, out_dir, cfg, progress)
    status("Writing the sheet and charts", None)
    events = result["events"]
    events["clip"] = ""
    if clips and cams and len(events):
        if progress:
            print("Cutting labelled clips ...")
        for idx, path in make_clips(events, result["frames"], cams, os.path.join(out_dir, "clips"), cfg).items():
            events.loc[idx, "clip"] = path
    write_csvs(result, out_dir)
    ids = info.get("cameras", [c["id"] for c in cams])
    write_sheet(result, os.path.join(out_dir, "sheet.xlsx"), cfg, ids)
    s = result["summary"]
    make_charts(result, os.path.join(out_dir, "charts.png"),
                f"CivetWatch · {cfg['civet_name']} · {s['start']:%d %b %Y %H:%M} – {s['end']:%d %b %H:%M}")
    write_summary(result, out_dir, {"models": info, "civet": cfg["civet_name"]})
    if progress:
        print_summary(result, out_dir)
    return result


def _is_stop_decision(v):
    b = v.get("behavior", "")
    return v.get("category", "stop" if b.startswith("possible_") or b in ("short_stop", "doorway_pass") else "event") == "stop"


def apply_stop_review(ev, out_dir, cfg):
    """Stops a person marked Correct (or 'It was <behavior>') in the dashboard
    become counted frequency events; checked stops are no longer flagged."""
    path = os.path.join(out_dir, "review.json")
    if not len(ev) or not os.path.exists(path):
        return ev, pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        review = json.load(f)
    by_event = {(v["start"][:19], v["camera"]): v for v in review.values() if v.get("start") and _is_stop_decision(v)}
    ev = ev.copy()
    rows = []
    for i, e in ev.iterrows():
        dec = by_event.get((f"{e['start']:%Y-%m-%d %H:%M:%S}", e["camera"]))
        if not dec:
            continue
        ev.at[i, "needs_review"] = False
        label = None
        if dec["status"] == "confirmed":
            label = {"possible_toileting": E.URINATION, "possible_marking": E.MARKING,
                     "possible_grooming": E.GROOMING}.get(e["behavior"])
        elif dec["status"] == "corrected" and dec.get("label") == "toileting":
            label = E.URINATION                     # not sure which: decided by length below
        elif dec["status"] == "corrected" and dec.get("label") in E.FREQUENCY_LABELS:
            label = dec["label"]
        if label == E.URINATION and dec.get("label") in (None, "toileting") and \
                e["duration_s"] >= cfg["behavior"].get("defecation_min_s", 5):
            label = E.DEFECATION
        who = dec.get("checker") or "a person"
        if label:
            ev.at[i, "note"] = f"checked by {who}: {E.DISPLAY[label]}"
            rows.append({**e.to_dict(), "behavior": label, "category": "frequency", "needs_review": False,
                         "note": f"checked by {who}"})
        elif dec.get("label") in E.DISPLAY and dec.get("label") != "cant_see":
            ev.at[i, "note"] = f"checked by {who}: {E.DISPLAY[dec['label']]}"      # resting / sitting: not counted
        elif dec.get("checker"):
            ev.at[i, "note"] = f"checked by {who}"
    return ev, merge_bouts(pd.DataFrame(rows), cfg)


def merge_bouts(rows, cfg):
    """One grooming / marking bout seen as several stops (the civet pauses, moves a
    little, goes on) is one occurrence in the sheet: confirmed events of the same
    behavior less than behavior.bout_gap_s apart are merged."""
    if not len(rows):
        return rows
    gaps = cfg["behavior"].get("bout_gap_s", {})
    gaps = gaps if isinstance(gaps, dict) else {}
    rows = rows.sort_values("start").reset_index(drop=True)
    keep, last = [], {}
    for r in rows.to_dict("records"):
        p = last.get(r["behavior"])
        gap = pd.Timedelta(seconds=gaps.get(r["behavior"], 60))
        if p is not None and r["start"] - p["stop"] <= gap:
            p["stop"] = max(p["stop"], r["stop"])
            p["duration_s"] = (p["stop"] - p["start"]).total_seconds()
            p["note"] = p["note"] + " (one bout, several stops)" if "one bout" not in p["note"] else p["note"]
            continue
        keep.append(r)
        last[r["behavior"]] = r
    return pd.DataFrame(keep)


GUESS_LABEL = {"urination": E.URINATION, "defecation": E.DEFECATION, "territorial_marking": E.MARKING,
               "grooming": E.GROOMING}


def auto_accept_stops(ev, cfg):
    """Stops the own-footage model (train-behavior) is very sure about are
    counted without a person when stops.auto_accept is set. Off by default."""
    thr = cfg["stops"].get("auto_accept")
    if not thr or not len(ev) or "guess" not in ev:
        return ev, pd.DataFrame()
    ev, rows = ev.copy(), []
    for i, e in ev.iterrows():
        label = GUESS_LABEL.get(e.get("guess"))
        if not label or not e["needs_review"] or float(e.get("guess_prob") or 0) < thr:
            continue
        ev.at[i, "note"] = f"auto: {E.DISPLAY[label]} ({float(e['guess_prob']):.0%} sure), not checked"
        rows.append({**e.to_dict(), "behavior": label, "category": "frequency", "needs_review": False,
                     "note": f"auto (model {float(e['guess_prob']):.0%} sure), not checked by a person"})
    return ev, pd.DataFrame(rows)


def _recount(result, cfg):
    events, s = result["events"], result["summary"]
    from .postprocess import hourly_counts
    freq = events[events["category"] == "frequency"]
    result["hourly"] = hourly_counts(events, s["start"], s["end"], cfg["report"]["sheet_start_hour"])
    s["counts"] = {k: int(v) for k, v in freq["behavior"].value_counts().items()}


def apply_event_review(result, out_dir, cfg, progress=True):
    """Answers given in the dashboard for feeding / drinking / other events:
    Correct keeps it, 'Not this behavior' removes it from the counts, 'It was ...'
    changes it. Stops are handled in apply_stop_review."""
    path = os.path.join(out_dir, "review.json")
    ev = result["events"]
    if not len(ev) or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        review = json.load(f)
    by_event = {(v["start"][:19], v["camera"]): v for v in review.values() if v.get("start") and not _is_stop_decision(v)}
    changed = 0
    for i, e in ev.iterrows():
        if e["category"] in ("stop", "not_visible"):
            continue
        dec = by_event.get((f"{e['start']:%Y-%m-%d %H:%M:%S}", e["camera"]))
        if not dec or dec.get("behavior") not in (None, e["behavior"]):
            continue
        ev.at[i, "needs_review"] = False
        if dec["status"] == "confirmed":
            ev.at[i, "note"] = "checked: correct"
            continue
        label = dec.get("label") if dec["status"] == "corrected" else None
        if e["category"] == "frequency":
            changed += 1
            if label in E.FREQUENCY_LABELS:
                ev.at[i, "behavior"], ev.at[i, "note"] = label, f"checked: it was {E.DISPLAY[label]}"
            else:
                ev.at[i, "category"] = "removed"
                ev.at[i, "note"] = f"removed by a person: it was {E.DISPLAY.get(label, 'something else')}"                     if label else "removed by a person: not this behavior"
        else:
            ev.at[i, "note"] = f"checked: it was {E.DISPLAY.get(label, label)}" if label else "checked: not this behavior"
    result["events"] = ev
    _recount(result, cfg)
    s = result["summary"]
    s["to_review"] = int(ev["needs_review"].astype(bool).sum())
    if progress and changed:
        print(f"{changed} counted event(s) changed or removed by a person's check")


def add_stop_events(result, cams, cfg, out_dir, progress=True):
    """Short stops (possible toileting / marking / grooming) -> events, flagged
    for checking by the vision-language pre-filter. Not counted in the sheet."""
    from .stops import find_box_visits, find_stops, judge_stops, mark_doorway, score_stops, stop_events
    stops = mark_doorway(find_stops(result["frames"], cfg), result["frames"], cams, cfg)
    visits = find_box_visits(result["frames"], cams, cfg)
    if len(visits):
        # a box-top visit replaces the short stops inside it (one thing to check, not three)
        if len(stops):
            inside = [any(v.camera == st.camera and v.start < st.stop and st.start < v.stop for v in visits.itertuples())
                      for st in stops.itertuples()]
            stops = stops[[not x for x in inside]]
        visits["doorway"] = False
        stops = pd.concat([stops, visits], ignore_index=True).sort_values("start").reset_index(drop=True)
        if progress:
            print(f"{len(visits)} visit(s) to the sleeping box top")
    if not len(stops):
        return
    stops = judge_stops(stops, cams, cfg, out_dir, progress)
    stops = score_stops(stops, cams, cfg, progress)
    ev = stop_events(stops)
    flagged = int(ev["needs_review"].sum()) if len(ev) else 0
    s_ = result["summary"]
    s_["busy_stops"] = int((ev.get("kind", pd.Series(dtype=str)) == "busy_still").sum()) if len(ev) else 0
    if progress:
        print(f"{len(stops)} short stops: {flagged} to check in the dashboard (Quick check), "
              f"{len(stops) - flagged} were the civet passing between rooms")
    ev, confirmed = apply_stop_review(ev, out_dir, cfg)
    ev, auto = auto_accept_stops(ev, cfg)
    events = pd.concat([result["events"], ev, confirmed, auto], ignore_index=True).sort_values("start").reset_index(drop=True)
    result["events"] = events
    s = result["summary"]
    if len(confirmed) or len(auto):
        _recount(result, cfg)
        if progress:
            print(f"{len(confirmed)} checked stop(s) and {len(auto)} auto-accepted stop(s) added to the counts")
    s["to_review"] = int(events["needs_review"].astype(bool).sum())
    s["short_stops"], s["stops_flagged"] = len(stops), flagged
    result["alerts"] = [a for a in result["alerts"] if "to check by a person" not in a]
    if s["to_review"]:
        nr = events["needs_review"].astype(bool)
        n_stops = int((nr & (events["category"] == "stop")).sum())
        result["alerts"].append(f"{s['to_review']} thing(s) to check by a person: {n_stops} in Behavior check "
                                f"and {s['to_review'] - n_stops} unsure events")


def print_summary(result, out_dir):
    s = result["summary"]
    from .report import fmt_hm
    print(f"\nDone. {fmt_hm(s['observed_s'])} observed | active {fmt_hm(s['active_s'])} ({s['active_frac']:.0%}) | "
          f"inactive {fmt_hm(s['inactive_s'])}")
    if s["counts"]:
        print("Counts: " + ", ".join(f"{E.DISPLAY.get(k, k)} {v}" for k, v in s["counts"].items()))
    for a in result["alerts"]:
        print(f"  ! {a}")
    print(f"Outputs in {out_dir}: events.csv, frames.csv, sheet.xlsx, charts.png, summary.json, clips/")
