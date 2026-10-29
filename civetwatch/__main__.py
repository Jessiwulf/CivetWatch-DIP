"""CivetWatch command line.  python -m civetwatch <command> --help"""
import argparse
import os
import sys

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")     # hide FFmpeg's harmless WebM tag warnings
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from .config import PROJECT_ROOT, load_config, parse_time


def _out(path):
    return path or os.path.join(PROJECT_ROOT, "runs", "latest")


def cmd_demo(a, cfg):
    from . import compare, pipeline, synthetic
    out = a.out or os.path.join(PROJECT_ROOT, "runs", "demo")
    if not a.nlm:
        cfg["preprocess"]["denoise"] = "bilateral"        # ~4x faster; --nlm for the full setting
    cfg["behavior"]["groom_without_pose"] = True          # the drawn civet has no pose model
    data = os.path.join(out, "data")
    print("1/3 Generating a synthetic 4-camera recording ...")
    manifest = synthetic.generate(data)
    print("2/3 Running CivetWatch ...")
    pipeline.run(manifest, cfg, out, clips=not a.no_clips)
    print("\n3/3 Success check against the scripted behavior:")
    table, overall = compare.compare_frames(os.path.join(out, "frames.csv"), os.path.join(data, "ground_truth.csv"))
    print(table.to_string(index=False))
    print("  " + "  ".join(f"{k}: {v:.0%}" for k, v in overall.items()))
    print(f"\nOpen the dashboard:  python -m civetwatch dashboard --runs {os.path.dirname(out)}")


def cmd_run(a, cfg):
    from . import pipeline
    pipeline.run(a.manifest, cfg, _out(a.out), start=parse_time(a.start) if a.start else None,
                 end=parse_time(a.end) if a.end else None, clips=not a.no_clips, fresh=a.fresh)


def cmd_report(a, cfg):
    """Re-run post-processing on frames.csv (fast: try new thresholds without re-analysing video)."""
    import pandas as pd
    from . import pipeline
    from .config import load_manifest
    from .features import FEATURES
    frames = pd.read_csv(a.frames, parse_dates=["t"])
    for col in frames.columns:
        if col not in FEATURES and frames[col].dtype == object:
            frames[col] = frames[col].fillna("")
    frames = frames.drop(columns=[c for c in ("voted", "state", "rest_type") if c in frames])
    cams = load_manifest(a.manifest)["cameras"] if a.manifest else []
    if a.relabel:
        if not cams:
            raise SystemExit("--relabel needs --manifest (for the zones)")
        print("Re-applying zones and behavior rules ...")
        frames = pipeline.relabel(frames, cfg, cams)
    info = {"detector": "from frames.csv", "cameras": sorted(c for c in frames["camera"].unique() if c)}
    pipeline.report(frames, cams, info, cfg, _out(a.out), clips=bool(cams) and not a.no_clips)


def cmd_prepare(a, cfg):
    from .video import prepare_video
    from .config import timestamp_from_filename
    start = parse_time(a.start) if a.start else timestamp_from_filename(a.video)
    if start is None:
        raise SystemExit("--start is needed: no time in the file name")
    for p in prepare_video(a.video, a.out, start, fps=cfg["sampling"]["fps"], width=cfg["sampling"]["process_width"],
                           hwaccel=a.hwaccel):
        print(p)


def cmd_zones(a, cfg):
    from .zones import import_cvat, save_zones
    if a.cvat:
        zones = import_cvat(a.cvat, a.image)
        save_zones(a.out, a.camera, zones)
        print(f"{len(zones)} zones -> {a.out}")
    else:
        from .zone_editor import edit
        edit(a.video, a.out, a.camera)


def cmd_extract(a, cfg):
    from .training import extract_frames
    print(f"{extract_frames(a.manifest, a.out, a.every, a.max)} frames saved to {a.out}")


def cmd_autolabel(a, cfg):
    from .training import autolabel_dataset, contact_sheets
    drop = []
    for item in a.drop or []:                       # cam2@2026-04-05 19:00:00@2026-04-05 19:10:00
        cam, start, end = item.split("@")
        drop.append((cam, start, end))
    index = autolabel_dataset(a.run, a.manifest, a.out, a.every, a.negatives, min_motion=a.min_motion,
                              min_size=a.min_size, min_conf=a.min_conf, drop=drop, cfg=cfg)
    print(f"{int(index['positive'].sum())} labelled civet images + {int((~index['positive']).sum())} empty images "
          f"-> {a.out}  (data.yaml ready for train-detector)")
    sheets = contact_sheets(a.out, os.path.join(a.out, "contact_sheets"))
    print(f"check the labels by eye: {len(sheets)} contact sheets in {os.path.join(a.out, 'contact_sheets')}")


def cmd_init_datasets(a, cfg):
    from .training import write_dataset_templates
    for p in write_dataset_templates():
        print(p)


def cmd_train_detector(a, cfg):
    from .training import train_detector
    print(train_detector(a.data, a.base, a.epochs, a.imgsz, a.device, resume=a.resume))


def cmd_train_pose(a, cfg):
    from .training import train_pose
    print(train_pose(a.data, a.base, a.epochs, a.imgsz, a.device, resume=a.resume))


def cmd_train_gru(a, cfg):
    from .gru import train
    from .config import resolve
    out = a.out or resolve(cfg["behavior"]["gru_weights"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report = train(a.frames, out, window=cfg["behavior"]["gru_window"], epochs=a.epochs)
    for k, v in report.items():
        print(f"  {k:22s} n={v['n']:5d}  recall={v['recall']:.0%}")
    print(f"saved {out}")


def cmd_export_labels(a, cfg):
    from .training import export_labels
    path, n = export_labels(a.run, a.out, a.only_reviewed)
    print(f"{path}  ({n} keeper decisions applied)")


def cmd_compare(a, cfg):
    import pandas as pd
    from . import compare
    pd.set_option("display.width", 140)
    if a.truth:
        table, overall = compare.compare_frames(a.auto, a.truth, cfg["behavior"]["vote_window"])
        print(table.to_string(index=False))
        print("  ".join(f"{k}: {v:.0%}" for k, v in overall.items()))
    elif a.keeper.lower().endswith(".xlsx"):
        print(compare.compare_sheets(a.keeper, a.auto).to_string(index=False))
    else:
        print(compare.compare_events(a.keeper, a.auto, a.tolerance).to_string(index=False))


def cmd_spotcheck(a, cfg):
    from . import spotcheck
    items = spotcheck.make_spotcheck(a.run, a.manifest, cfg, n=a.n, seed=a.seed)
    print(f"{len(items)} moments ready. Check them in the dashboard (Spot check link), "
          f"or fill {os.path.join(a.run, 'spotcheck', 'answers.json')}")


def cmd_spotcheck_report(a, cfg):
    import pandas as pd
    from . import spotcheck
    summary, df, confusion = spotcheck.score(a.run)
    for k, v in summary.items():
        if isinstance(v, float):
            v = "n/a (no such moments)" if v != v else f"{v:.0%}"
        print(f"  {k:32s} {v}")
    if len(confusion):
        pd.set_option("display.width", 200)
        print()
        print(confusion.to_string())


def cmd_dashboard(a, cfg):
    from .dashboard import serve
    serve(a.runs or os.path.join(PROJECT_ROOT, "runs"), a.port, cfg)


def main(argv=None):
    p = argparse.ArgumentParser(prog="civetwatch", description="Automatic behavior detection for the Large-spotted Civet")
    p.add_argument("--config", help="settings yaml (merged over config/default.yaml)")
    p.add_argument("--profile", choices=["laptop", "full"], help="resource limits (default: laptop)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="generate a synthetic recording and run the whole pipeline on it")
    s.add_argument("--out")
    s.add_argument("--no-clips", action="store_true")
    s.add_argument("--nlm", action="store_true", help="use Non-Local Means denoising (slower) like a real run")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("run", help="analyse one observation day")
    s.add_argument("--manifest", required=True, help="yaml listing the camera videos")
    s.add_argument("--out", help="output folder (default runs/latest)")
    s.add_argument("--from", dest="start", help='"YYYY-MM-DD HH:MM:SS"')
    s.add_argument("--to", dest="end")
    s.add_argument("--no-clips", action="store_true")
    s.add_argument("--fresh", action="store_true", help="ignore saved hours and analyse everything again")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="rebuild events/sheet/charts from frames.csv with the current settings")
    s.add_argument("--frames", required=True)
    s.add_argument("--manifest", help="needed for clips and --relabel")
    s.add_argument("--relabel", action="store_true", help="re-apply current zones + rules to the saved boxes")
    s.add_argument("--out")
    s.add_argument("--no-clips", action="store_true")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("prepare", help="FFmpeg: split a recording into 1-hour files at 2 fps")
    s.add_argument("--video", required=True)
    s.add_argument("--start", help="true start time of the recording (default: read from the file name)")
    s.add_argument("--hwaccel", help="FFmpeg hardware decoding, e.g. cuda")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("zones", help="draw a camera's zones, or import them from CVAT")
    s.add_argument("--camera", required=True)
    s.add_argument("--out", required=True, help="zones json to write")
    s.add_argument("--video", help="video or image to draw on")
    s.add_argument("--cvat", help="CVAT for images 1.1 XML export")
    s.add_argument("--image", help="image name inside the CVAT export")
    s.set_defaults(fn=cmd_zones)

    s = sub.add_parser("extract-frames", help="save frames for labelling in CVAT")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--every", type=float, default=10.0, help="seconds between frames (halved at night)")
    s.add_argument("--max", type=int, help="max frames per camera")
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("autolabel", help="bootstrap a YOLO civet dataset from a motion-detector run")
    s.add_argument("--run", required=True, help="run folder made with the motion detector")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--every", type=float, default=6.0, help="seconds between labelled frames")
    s.add_argument("--negatives", type=float, default=0.25, help="empty frames per labelled frame")
    s.add_argument("--min-motion", type=float, default=0.05, help="keep only moving-civet boxes")
    s.add_argument("--min-size", type=float, default=100, help="longest box side in pixels (960-wide frames)")
    s.add_argument("--min-conf", type=float, default=0.0, help="detector confidence (use ~0.6 for a YOLO run)")
    s.add_argument("--drop", nargs="*", help='wrong windows: "cam@start@end"')
    s.set_defaults(fn=cmd_autolabel)

    s = sub.add_parser("init-datasets", help="write YOLO dataset yaml templates")
    s.set_defaults(fn=cmd_init_datasets)

    for name, fn, base, imgsz in (("train-detector", cmd_train_detector, "yolo26n.pt", 960),
                                  ("train-pose", cmd_train_pose, "yolo26n-pose.pt", 640)):
        s = sub.add_parser(name, help=f"fine-tune {base} on the zoo dataset")
        s.add_argument("--data", required=True, help="dataset yaml")
        s.add_argument("--base", default=base)
        s.add_argument("--epochs", type=int, default=100)
        s.add_argument("--imgsz", type=int, default=imgsz)
        s.add_argument("--device", help="0 for the first GPU, cpu")
        s.add_argument("--resume", action="store_true", help="continue an interrupted training run")
        s.set_defaults(fn=fn)

    s = sub.add_parser("train-gru", help="train the grooming/toileting/marking sequence model")
    s.add_argument("--frames", nargs="+", required=True, help="frames_labelled.csv files")
    s.add_argument("--out")
    s.add_argument("--epochs", type=int, default=40)
    s.set_defaults(fn=cmd_train_gru)

    s = sub.add_parser("export-labels", help="frames.csv + keeper review -> training labels")
    s.add_argument("--run", required=True)
    s.add_argument("--out")
    s.add_argument("--only-reviewed", action="store_true")
    s.set_defaults(fn=cmd_export_labels)

    s = sub.add_parser("compare", help="success check: keeper vs CivetWatch")
    s.add_argument("--auto", required=True, help="CivetWatch sheet.xlsx / events.csv / frames.csv")
    s.add_argument("--keeper", help="keeper sheet.xlsx or events csv (behavior,start)")
    s.add_argument("--truth", help="per-sample ground truth csv (t,true_label)")
    s.add_argument("--tolerance", type=float, default=60)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("spotcheck", help="pick random moments for people to check by eye (no keeper sheet needed)")
    s.add_argument("--run", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--n", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_spotcheck)

    s = sub.add_parser("spotcheck-report", help="accuracy of CivetWatch against the spot-check answers")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_spotcheck_report)

    s = sub.add_parser("dashboard", help="keeper review dashboard in the browser")
    s.add_argument("--runs", help="folder containing run folders (default runs/)")
    s.add_argument("--port", type=int, default=8050)
    s.set_defaults(fn=cmd_dashboard)

    a = p.parse_args(argv)
    if a.cmd == "compare" and not (a.keeper or a.truth):
        p.error("compare needs --keeper or --truth")
    cfg = load_config(a.config)
    if a.profile:
        cfg.setdefault("resources", {})["profile"] = a.profile
    from . import resources
    r = resources.apply(cfg)
    if a.cmd not in ("dashboard", "compare", "spotcheck-report"):
        print(f"[resources] profile={r['profile']}  threads={r['threads'] or 'all'}  priority={r['priority']}  "
              f"duty={r['duty_cycle']:.0%}  GPU pause at {r['gpu_temp_pause']} C")
    a.fn(a, cfg)


if __name__ == "__main__":
    sys.exit(main())
