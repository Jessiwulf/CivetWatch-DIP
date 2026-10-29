"""Layer 2 · Find & track the civet (YOLO26 + ByteTrack) and layer 3 · read its
body (YOLO26-Pose).

Without trained weights CivetWatch falls back to a background-subtraction
detector, so the pipeline can run (and be tested) before any labelling.
"""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import resolve
from .ethogram import KEYPOINTS


@dataclass
class Detection:
    cls: str                     # civet | food | water
    box: tuple                   # x1, y1, x2, y2 (pixels of the processed frame)
    conf: float
    track_id: int = None
    keypoints: np.ndarray = field(default=None, repr=False)   # (K, 3) x, y, conf
    elongation: float = None     # long / short side of the rotated outline (motion detector)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(weights)


class YoloDetector:
    """YOLO26 detector retrained on zoo frames, with ByteTrack per camera
    (one model instance per camera so each keeps its own tracker state).
    With detector.batch: true, all cameras of one moment go to the GPU in a
    single call instead (faster; no track IDs - one civet needs none)."""

    name = "yolo"

    def __init__(self, cfg, cam_ids):
        d = cfg["detector"]
        self.weights = resolve(d["weights"])
        self.conf = d["conf"]
        self.tracker = d.get("tracker", "bytetrack.yaml")
        self.class_map = d.get("class_map") or {}
        self.batch = bool(d.get("batch", False))
        if self.batch:
            self.model = _load_yolo(self.weights)
        else:
            self.models = {cid: _load_yolo(self.weights) for cid in cam_ids}

    def detect_many(self, frames):
        """{cam_id: frame} -> {cam_id: [Detection]} in one batched forward pass."""
        ids = list(frames)
        if not ids:
            return {}
        results = self.model.predict([frames[c] for c in ids], conf=self.conf, verbose=False)
        return {c: self._parse(r) for c, r in zip(ids, results)}

    def __call__(self, cam_id, frame, gray=None, light=None):
        if self.batch:
            return self.detect_many({cam_id: frame})[cam_id]
        res = self.models[cam_id].track(frame, persist=True, tracker=self.tracker,
                                        conf=self.conf, verbose=False)[0]
        return self._parse(res)

    def _parse(self, res):
        out = []
        if res.boxes is None:
            return out
        names = res.names
        ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
        for xyxy, conf, cls, tid in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                        res.boxes.cls.int().tolist(), ids):
            mapped = self.class_map.get(names[cls])
            if mapped:
                out.append(Detection(mapped, tuple(xyxy), float(conf), tid))
        return out


class MotionDetector:
    """Fallback: civet = biggest blob that differs from the empty-room background.
    One background per camera and light mode (day colour / night IR), built from
    the median of frames spread over the recording, or from a given image."""

    name = "motion"

    def __init__(self, cfg, cam_ids):
        m = cfg["detector"]["motion"]
        self.min_area, self.max_area = m["min_area_frac"], m["max_area_frac"]
        self.thr = m["diff_threshold"]
        self.bg = {}            # (cam, light) -> float32 gray background
        self.ignore = {}        # cam -> uint8 mask, 1 = never a civet (clock text, lens dirt)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def set_ignore(self, cam_id, polygons, shape):
        mask = np.zeros(shape[:2], np.uint8)
        for poly in polygons:
            cv2.fillPoly(mask, [poly], 1)
        self.ignore[cam_id] = mask if mask.any() else None

    def set_background(self, cam_id, light, frames):
        stack = np.stack([cv2.GaussianBlur(f, (5, 5), 0) for f in frames]).astype(np.float32)
        self.bg[(cam_id, light)] = np.median(stack, axis=0)

    def __call__(self, cam_id, frame, gray=None, light="day_colour"):
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg = self.bg.get((cam_id, light))
        if bg is None:
            bg = next((v for (c, _), v in self.bg.items() if c == cam_id), None)
        blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        if bg is None or bg.shape != blur.shape:
            self.bg[(cam_id, light)] = blur
            return []
        diff = cv2.absdiff(blur, bg)
        # threshold adapts to the noise of this frame (median + 5 robust sigma)
        sample = diff[::4, ::4]
        med = float(np.median(sample))
        mad = float(np.median(np.abs(sample - med))) * 1.4826
        thr = max(self.thr, med + 5 * mad)
        mask = (cv2.GaussianBlur(diff, (9, 9), 0) > thr).astype(np.uint8)
        ign = self.ignore.get(cam_id)
        if ign is not None and ign.shape == mask.shape:
            mask[ign > 0] = 0
        # low-contrast fur breaks into spots and legs: close the gaps, then drop specks
        k = max(3, int(gray.shape[1] / 43) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_total = gray.shape[0] * gray.shape[1]
        best = None
        for c in contours:
            a = cv2.contourArea(c)
            if self.min_area * area_total <= a <= self.max_area * area_total and (best is None or a > best[0]):
                best = (a, c)
        if best is None:
            return []
        x, y, w, h = cv2.boundingRect(best[1])
        contrast = float(diff[y:y + h, x:x + w][mask[y:y + h, x:x + w] > 0].mean())
        conf = float(np.clip(0.4 + contrast / 120.0, 0, 0.99))
        # let the background follow slow light changes everywhere except the civet
        keep = np.ones_like(mask)
        keep[max(0, y - h // 2):y + h + h // 2, max(0, x - w // 2):x + w + w // 2] = 0
        cv2.accumulateWeighted(blur, bg, 0.01, mask=keep)
        (_, _), (rw, rh), _ = cv2.minAreaRect(best[1])
        elong = float(max(rw, rh) / max(1.0, min(rw, rh)))
        return [Detection("civet", (x, y, x + w, y + h), conf, 1, elongation=elong)]


class PoseEstimator:
    """YOLO26-Pose trained on civet keypoints (see ethogram.KEYPOINTS)."""

    name = "yolo26-pose"

    def __init__(self, cfg):
        p = cfg["pose"]
        self.model = _load_yolo(resolve(p["weights"]))
        self.conf, self.kp_conf = p["conf"], p["kp_conf"]

    def __call__(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * 0.25, (y2 - y1) * 0.25
        cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        cx2, cy2 = int(min(w, x2 + px)), int(min(h, y2 + py))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        res = self.model.predict(crop, conf=self.conf, verbose=False)[0]
        if res.keypoints is None or len(res.boxes) == 0:
            return None
        local = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        best = max(range(len(res.boxes)), key=lambda i: iou(res.boxes.xyxy[i].tolist(), local))
        kp = res.keypoints.data[best].cpu().numpy().astype(np.float32)
        if kp.shape[0] != len(KEYPOINTS):
            return None
        kp[:, 0] += cx1
        kp[:, 1] += cy1
        kp[kp[:, 2] < self.kp_conf] = np.nan
        return kp


class ViTPoseEstimator:
    """Ready-made animal pose, no civet labels needed: ViTPose+ with its AP-10K
    head (17 points, trained on 54 mammal species). Its points are mapped onto
    the civet keypoints; there is no tail tip. Points with a low score or far
    outside the civet box are dropped, and so is the whole pose when the model
    is unsure (blurred / fast frames)."""

    name = "vitpose+ (AP-10K, ready-made)"
    AP10K = {"nose": [2], "left_ear": [0], "right_ear": [1], "withers": [3],      # eyes stand in for ears
             "front_left_paw": [7], "front_right_paw": [10], "hind_left_paw": [13],
             "hind_right_paw": [16], "hips": [11, 14], "tail_base": [4], "tail_tip": []}

    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        from .resources import torch_threads
        torch_threads()
        p = cfg["pose"]
        name = p.get("vitpose_model", "usyd-community/vitpose-plus-base")
        try:                                   # offline: use the copy already downloaded
            self.proc = AutoProcessor.from_pretrained(name, local_files_only=True)
            self.model = VitPoseForPoseEstimation.from_pretrained(name, local_files_only=True)
        except OSError:
            self.proc = AutoProcessor.from_pretrained(name)
            self.model = VitPoseForPoseEstimation.from_pretrained(name)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.kp_conf = p.get("kp_conf", 0.4)
        self.min_mean = p.get("min_mean", 0.45)
        self.head = torch.tensor([p.get("vitpose_head", 3)], device=self.device)     # 3 = AP-10K animals

    def __call__(self, frame, box):
        from PIL import Image
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * 0.1, (y2 - y1) * 0.1
        bx1, by1, bx2, by2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
        coco = [bx1, by1, bx2 - bx1, by2 - by1]
        if coco[2] < 8 or coco[3] < 8:
            return None
        pil = Image.fromarray(frame[:, :, ::-1])
        inputs = self.proc(pil, boxes=[[coco]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs, dataset_index=self.head)
        res = self.proc.post_process_pose_estimation(out, boxes=[[coco]])[0][0]
        pts, sc = res["keypoints"].cpu().numpy(), res["scores"].cpu().numpy()
        if sc.mean() < self.min_mean:
            return None
        mx, my = (x2 - x1) * 0.25, (y2 - y1) * 0.25
        ok = (sc >= self.kp_conf) & (pts[:, 0] >= x1 - mx) & (pts[:, 0] <= x2 + mx) \
            & (pts[:, 1] >= y1 - my) & (pts[:, 1] <= y2 + my)
        kp = np.full((len(KEYPOINTS), 3), np.nan, np.float32)
        for i, name in enumerate(KEYPOINTS):
            src = [j for j in self.AP10K[name] if ok[j]]
            if src:
                kp[i, :2] = pts[src].mean(0)
                kp[i, 2] = sc[src].mean()
        return kp


def build_detector(cfg, cam_ids):
    d = cfg["detector"]
    kind = d.get("type", "auto")
    if kind == "auto":
        kind = "yolo" if os.path.exists(resolve(d["weights"])) else "motion"
    if kind == "yolo":
        return YoloDetector(cfg, cam_ids)
    return MotionDetector(cfg, cam_ids)


def build_pose(cfg):
    """auto: our own trained YOLO26-Pose if models/civet_pose.pt exists, else
    the ready-made ViTPose+ animal model (if `transformers` is installed)."""
    p = cfg["pose"]
    enabled = p.get("enabled", "auto")
    if enabled is False or enabled == "false":
        return None
    if os.path.exists(resolve(p["weights"])) and p.get("type", "auto") in ("auto", "yolo"):
        return PoseEstimator(cfg)
    if p.get("type", "auto") in ("auto", "vitpose"):
        try:
            return ViTPoseEstimator(cfg)
        except ImportError:
            print("[pose] transformers not installed - running without body points")
        except OSError as e:
            print(f"[pose] ViTPose model not available ({e.__class__.__name__}) - running without body points")
    return None
