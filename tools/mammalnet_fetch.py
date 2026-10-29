"""Download a small, behavior-labelled subset of MammalNet (CVPR 2023, research use):
grooming / urinating / defecating clips of cats and other small carnivores, plus
'other behavior' clips of the same animals. Only the labelled seconds are kept,
at low resolution. Resumable: run again to continue.

python tools/mammalnet_fetch.py <annotation dir> <out dir>
"""
import json
import os
import random
import subprocess
import sys
import time

FAMILIES = {"felidae", "herpestidae", "viverridae", "procyonidae", "mephitidae", "mustelidae"}
TARGET = {"grooms/cleans itself or other animal": "grooming", "grooms": "grooming",
          "pees": "toileting", "poops": "toileting"}
OTHER = {"eats food", "drinks water", "sleeps", "hunts other animals", "fights against other animals"}
N_OTHER = 220
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"


def clip_path(out_dir, j):
    return os.path.join(out_dir, f"{j['label']}__{j['vid']}_{j['k']}.mp4")


def main(ann_dir, out_dir):
    db = json.load(open(os.path.join(ann_dir, "detection_annotations.json"), encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    jobs, other = [], []
    for vid, v in db.items():
        tax = v.get("taxnomy") or v.get("taxonomy") or [{}]
        if tax[0].get("family") not in FAMILIES:
            continue
        for k, a in enumerate(v["annotations"]):
            lab = a["label"].replace("_", " ")
            name = TARGET.get(lab) or (TARGET.get("grooms") if lab.startswith("grooms") else None)
            row = {"vid": vid, "k": k, "start": a["segment"][0], "end": a["segment"][1], "label": name or "other",
                   "orig": lab, "genus": tax[0].get("genus"), "subset": v.get("subset"), "url": v["url"]}
            if name:
                jobs.append(row)
            elif lab in OTHER:
                other.append(row)
    random.Random(0).shuffle(other)
    jobs += other[:N_OTHER]
    json.dump(jobs, open(os.path.join(out_dir, "clips.json"), "w"), indent=1)
    print(f"{len(jobs)} clips to fetch:", {l: sum(j['label'] == l for j in jobs) for l in ("grooming", "toileting", "other")},
          flush=True)
    from concurrent.futures import ThreadPoolExecutor
    flags = 0x00004000 if os.name == "nt" else 0          # below-normal priority
    todo = [j for j in jobs if not any(os.path.exists(clip_path(out_dir, j) + x) for x in ("", ".fail"))]
    print(f"{len(jobs) - len(todo)} already done, {len(todo)} to fetch (3 at a time)", flush=True)
    stats = {"ok": 0, "fail": 0}

    def fetch(j):
        path = clip_path(out_dir, j)
        a, b = max(0.0, j["start"] - 0.5), j["end"] + 0.5
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist", "--js-runtimes", "node",
               "--ffmpeg-location", FFMPEG,
               "-f", "bv*[height<=240][ext=mp4][protocol^=http]/bv*[height<=360][ext=mp4][protocol^=http]/b[height<=360]/bv*[height<=480]",
               "--download-sections", f"*{a:.1f}-{b:.1f}", "--force-keyframes-at-cuts", "-o", path, j["url"]]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, creationflags=flags)
            good = r.returncode == 0 and os.path.exists(path)
        except subprocess.TimeoutExpired:
            good = False
        if not good:
            open(path + ".fail", "w").close()          # do not retry deleted / private videos
        stats["ok" if good else "fail"] += 1
        time.sleep(float(os.environ.get("FETCH_PAUSE", 0)))     # be gentle: YouTube blocks fast downloaders
        time.sleep(float(os.environ.get("FETCH_PAUSE", 0)))     # be gentle: YouTube blocks fast downloaders
        n = stats["ok"] + stats["fail"]
        if n % 20 == 0:
            print(f"{n}/{len(todo)}  downloaded {stats['ok']}  unavailable {stats['fail']}", flush=True)

    workers = int(os.environ.get("FETCH_WORKERS", 3))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, todo))
    ok, fail = stats["ok"], stats["fail"]
    print(f"done: {ok} clips, {fail} unavailable", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
