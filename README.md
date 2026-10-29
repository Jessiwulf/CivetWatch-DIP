# CivetWatch

Automatic behavior detection for the Large-spotted Civet (*Viverra megaspila*).
Chiang Mai Zoo · Smart Zoo project · SE494 Digital Image Processing.

**Video in, observation sheet out.** CivetWatch finds the civet in all 4 camera
views (day colour or night infrared), follows it from room to room, names
each ethogram behavior and fills in the zoo's observation sheet. Keepers only
review the result.

```
0 VIDEO IN ─▶ 1 PREPARE ─▶ 2 FIND + FOLLOW ─▶ 3 READ BODY ─▶ 4 NAME BEHAVIOR ─▶ 5 CLEAN + COUNT
4 cameras     2 fps, CLAHE   YOLO26 detector    YOLO26-Pose     zone + timer rules   majority vote,
4 rooms       + NLM denoise, + ByteTrack,       11 body dots    + GRU (4 s window)   1-minute rule,
              fixed zones    one civet ID                                            bout counting
                                                   ─▶ events.csv · frames.csv · sheet.xlsx · charts.png · clips/
```

## Quick start

```bash
pip install -r requirements.txt
python -m civetwatch demo
python -m civetwatch dashboard
```

`demo` draws a synthetic 12-minute, 4-camera night recording with a scripted
civet (curling up, walking between rooms, feeding twice, drinking, grooming,
stretching out, sleeping in the box, hiding out of view), runs the full
pipeline on it and scores the result against the script. The dashboard then
opens at http://127.0.0.1:8050.

## Running a real observation day

1. **Split and sample the recordings** (optional but much faster): each NVR
   export becomes 1-hour files at 2 fps, named with their true start time.
   ```bash
   python -m civetwatch prepare --video exports/cam1.mp4 --start "2026-07-20 15:00:00" --out videos/cam1
   ```
2. **Draw the fixed zones once per camera** (rest box, door, fence, log, tyre;
   food/water too if you have no tray detector yet). Left-click points,
   right-click to close, then press the zone key; `s` saves.
   ```bash
   python -m civetwatch zones --camera cam1 --video videos/cam1/cam1_20260720_150000.mp4 --out days/zones/cam1.json
   ```
   Zones drawn in CVAT can be imported instead: `--cvat annotations.xml`.
3. **Write a manifest** for the day — see [`config/manifest.example.yaml`](config/manifest.example.yaml).
4. **Run**
   ```bash
   python -m civetwatch run --manifest days/2026-07-20.yaml --out runs/2026-07-20
   ```
5. **Review** in the dashboard: confirm or correct the events marked *please check*.

To try new thresholds without re-analysing the video:
```bash
python -m civetwatch report --frames runs/2026-07-20/frames.csv --manifest days/2026-07-20.yaml --out runs/2026-07-20b
```

## Checking accuracy yourselves (no keeper sheet needed)

```bash
python -m civetwatch spotcheck --run runs/2026-04-05 --manifest data/2026-04-05/manifest.yaml --n 120
python -m civetwatch dashboard          # -> "Spot check" link on the day's page
python -m civetwatch spotcheck-report --run runs/2026-04-05
```

`spotcheck` picks random moments of the day (60 % from times CivetWatch saw
the civet, 40 % from anywhere, at least 2 minutes apart) and shows every
camera 1 s before, at, and 1 s after each moment. CivetWatch's answer stays
hidden while you answer *where is the civet* and *what is it doing*. Two
people can split the list; answers are saved in `spotcheck/answers.json`.
The report gives: civet found when in view, no false civet when out of view,
right room, Inactive/Active right, behavior right, a confusion table, and a
"look again" link for every disagreement.

## Outputs

| File | What it is |
|---|---|
| `sheet.xlsx` | The zoo's observation sheet: Inactive / Active / Mating / Not visible Start-Stop rows, hourly frequency table (Feeding, Drinking, Territorial marking, Grooming, Urine, Feces), plus Events, Inactive detail, Summary and Alerts tabs |
| `events.csv` | One row per event: time · room/camera · behavior · duration · confidence · condition (light, visibility, quality) · review flag · clip |
| `frames.csv` | Every analysed sample (2 per second): box, body dots, features, AI label, voted label, state. Every number in the sheet links back here |
| `charts.png` | Day timeline, time per room, hourly counts heat map, time budget |
| `clips/` | Labelled WebM clips of events to review (box, ID, body dots, zones, behavior) |
| `summary.json` | Headline numbers and alerts (e.g. *no feeding after 22:00, please check*) |

## Behavior vs condition

*Behavior* is what the civet does; *condition* is how well the cameras can see
it. Both are saved for every sample, so bad video is never mistaken for a behavior:

* night infrared frames are brightened (CLAHE) and denoised (Non-Local Means);
* blurred, too dark or frozen frames are **poor image** and are not counted;
* no civet in any camera for ≥ 10 s is **Not visible**, never *resting* —
  unless it was last seen entering the **rest box**, which is *Sleep in a box*.

## Body points without labelling (ViTPose+)

Until a civet pose model is trained, CivetWatch uses the ready-made animal
pose model **ViTPose+ (AP-10K head, 54 mammal species)** on every frame where
the civet is visible. It finds the nose, eyes, neck, tail base, hips and paws
(no tail tip). Unsure poses (blur, fast movement), points outside the civet box
and bodies cut by the picture edge are ignored. The model (500 MB) is
downloaded once on the first run and then works offline. It adds about 20 %
to the run time.

The body points are used for feeding / drinking (nose at the dish or basin).
They are **not** used to count grooming, marking or toileting: seen from
above, the model often mixes up the civet's head and tail (tested on 5 Apr
2026), so those rules are off (`behavior.pose_rare_rules: false`).

## Toileting, marking and grooming: short stops + check

There are no latrine zones and no labelled examples, so these rare behaviors
are found in two steps (`civetwatch/stops.py`):

1. **Short-stop finder.** All three happen while the civet stands still for a
   few seconds between walking. CivetWatch lists every stop of 3-60 s in one
   camera with walking within 8 s before and after, away from the food and
   water (stillness = how far the box centre moves over 2 s; arriving from or
   leaving to another room also counts as walking). Tested on 40 busy minutes
   of 5 Apr 2026: 20 stops, all 3 toileting moments found by eye among them.
   Estimated about 160 stops for the whole day (`stops.count_room_changes:
   false` gives about 65, but misses stops right after the civet enters a room).
2. **Quick check by a person.** Stops in a doorway with another camera seeing
   the civet within 4 s are the civet passing between rooms and are skipped
   (5 Apr 2026: 30 of 191, none of the real events). Every other stop is shown
   in the dashboard's **Quick check** page as 4 pictures; press a key:
   `1` urinating, `2` defecating, `3` toileting (not sure which - 5 s or more
   counts as defecating), `4` scent marking, `5` grooming, `0` nothing special.
   Space enlarges, arrows move. About 160 stops a day, 2 s each.
3. **Rebuild the report** (`civetwatch report ...`) and the answers are counted
   in the sheet and charts. Nothing else is counted as these four behaviors.

We tried to let software decide instead, and measured each attempt against our
own check of every stop on 5 Apr 2026 - none was good enough:
a vision-language model (Qwen2.5-VL-3B: with the answer options reordered it
said "marking" for 108 of 127 stops whatever the picture - it was guessing),
motion / body-outline measurements, new marks on the floor, tail angle from
pose, zero-shot (SigLIP) and few-shot (DINOv2) image matching. The checked
answers collected by Quick check are the way forward: after a few days they
are real civet labels to train a classifier on.

Settings: `stops:` in `config/default.yaml`. The meaning of every word in the app is in the
dashboard ("What the words mean") and the *Read me* tab of `sheet.xlsx` (`civetwatch/glossary.py`).

## How each behavior is recognised

| Behavior | Counted as | Rule |
|---|---|---|
| Stretched out / Lying down / Sleep in a box / Curling up | Inactive · duration | box barely moves; pose + zone tell which (rest-box zone, tyre zone or compact ball, long body) |
| Locomotion / Walking | Active · duration | speed above `walk_speed` body lengths/s |
| Feeding | count | nose (or box) in the FOOD-tray zone, head down, small chewing motion, ≥ 3 s |
| Drinking | count | nose in the WATER-tray zone, head down, lapping motion, ≥ 2 s |
| Grooming, territorial marking, urination, defecation | listed to check, not counted automatically | short stop + vision-language pre-filter (see above); add confirmed ones to the sheet by hand |
| Not visible | duration | no civet box in any camera for ≥ 10 s |

Post-processing: a conf-weighted majority vote over 3 s (6 frames) removes odd
frames; the **1-minute rule** only switches Inactive ↔ Active when the new state
lasts ≥ 1 min; **bouts**: pauses shorter than 30 s stay in the same bout (one
trip to the tray = one count). All thresholds are in
[`config/default.yaml`](config/default.yaml); override them with `--config my.yaml`.

## Models and training

CivetWatch works at every stage of the project:

| Stage | Detector | Pose | Sequence |
|---|---|---|---|
| No labels yet | background subtraction (empty-room median per camera and light) | — | hand-written cues |
| Boxes labelled | `models/civet_det.pt` (YOLO26 + ByteTrack) | — | hand-written cues |
| Body dots labelled | 〃 | `models/civet_pose.pt` (YOLO26-Pose, 11 points) | hand-written cues |
| Keeper reviews collected | 〃 | 〃 | `models/civet_gru.pt` (GRU over 4 s) |

Models are picked up automatically when the files exist.

```bash
# 1 collect: frames for CVAT (night frames kept twice as often)
python -m civetwatch extract-frames --manifest days/2026-07-20.yaml --out datasets/frames --every 10
# 2 label in CVAT, export as YOLO; dataset yaml templates:
python -m civetwatch init-datasets
# 3 train (on the NVIDIA GPU: --device 0)
python -m civetwatch train-detector --data datasets/civet_det.yaml --device 0
python -m civetwatch train-pose --data datasets/civet_pose.yaml --device 0
# GRU: keeper review decisions become labels
python -m civetwatch export-labels --run runs/2026-07-20
python -m civetwatch train-gru --frames runs/2026-07-20/frames_labelled.csv
# 4-5 test on a new day and compare with the keeper's sheet
python -m civetwatch compare --keeper keeper_2026-07-21.xlsx --auto runs/2026-07-21/sheet.xlsx
```

### Bootstrapping the detector without hand-labelling (used for 5 Apr 2026)

The civet detector in `models/civet_det.pt` was trained this way, from the
zoo's own recordings, with no manual boxes:

```bash
# a) run with the motion detector (it needs no training)
python -m civetwatch --config data/2026-04-05/config.yaml run --manifest data/2026-04-05/manifest.yaml --out runs/2026-04-05_motion --no-clips
# b) its boxes of a *moving* civet become labels; contact sheets to check them by eye
python -m civetwatch autolabel --run runs/2026-04-05_motion --manifest data/2026-04-05/manifest.yaml --out datasets/civet_det_v1
#    delete wrong crops (or pass --drop "cam2@2026-04-05 23:07:40@2026-04-05 23:07:50")
# c) train, then repeat a)-b) with the trained YOLO model (--min-conf 0.6) to add
#    night and resting examples the motion detector could not label
python -m civetwatch train-detector --data datasets/civet_det_v1/data.yaml --base models/yolo26n.pt --imgsz 640 --device 0
```

Add zones of type `ignore` (key `i` in the zone tool) over the burned-in clock
and camera name so they never count as motion.

Detector classes: `civet`, `food_tray`, `water_tray` (mapped in
`detector.class_map`; COCO weights can bootstrap with `cat: civet`, `bowl: food`).
Pose keypoints: nose, left/right ear, withers, 4 paws, hips, tail base, tail tip.

The PyTorch installed by default is CPU-only. For the GPU, install the CUDA
build from https://pytorch.org/get-started/locally/.

## Speed, heat and crashes

Everything runs under a **resource profile** (`resources:` in `config/default.yaml`):

| | `laptop` (default) | `full` |
|---|---|---|
| CPU threads (OpenCV, PyTorch, FFmpeg) | 4 | all |
| Windows priority | below normal | normal |
| Duty cycle | work 70 %, idle 30 % | 100 % |
| Pause when the GPU reaches | 78 C (resume < 68 C) | 90 C |
| Training | batch 8, 1 data worker, 20 s rest per epoch | batch 16, 2 workers |

Measured on the i9-11900H / RTX 3050 Ti laptop, 4 cameras at 2 fps:
`laptop` 8.5 samples/s at about 8 % total CPU and 55 C GPU (24 h of video takes about 5.5 h);
`full` 15 samples/s (about 3.2 h). Use `--profile full` only on a well-cooled desktop.
Run one heavy job at a time (do not prepare videos while analysing).

**Crash-safe:** `run` saves every hour of analysed video in `<out>/parts/`.
After a crash, Ctrl+C or reboot, run the same command again and it continues
from the first unfinished hour (`--fresh` starts over). Interrupted training
continues with `train-detector ... --resume`.

## Code map

| Module | Role |
|---|---|
| `video.py` | shared clock across cameras, 2 fps sampling, CLAHE + NLM, image quality, FFmpeg prepare |
| `zones.py`, `zone_editor.py` | fixed + moving (tray) zones, point-in-polygon, CVAT import, drawing tool |
| `detect.py` | YOLO26 + ByteTrack, motion fallback, YOLO26-Pose |
| `fusion.py` | one civet across 4 cameras (global ID 1) |
| `features.py` | speed, motion, posture, nose zone … per sample |
| `rules.py`, `gru.py` | behavior rules, sequence model (heuristic / trained GRU) |
| `postprocess.py` | majority vote, 1-minute rule, bouts, not-visible rows, hourly table, alerts |
| `report.py`, `charts.py`, `clips.py` | sheet.xlsx, CSVs, charts.png, labelled clips |
| `dashboard.py` | keeper review web page |
| `compare.py` | success check: keeper vs CivetWatch |
| `synthetic.py` | synthetic 4-camera demo with ground truth |
