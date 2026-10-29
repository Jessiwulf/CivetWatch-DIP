"""One civet, four rooms: pick the camera that sees it and keep one unbroken
path (global ID 1) as it moves from room to room."""


class CivetFusion:
    def __init__(self, cfg):
        self.margin = cfg["tracking"]["switch_margin"]
        self.cam = None
        self.last_track = {}      # cam -> local track id last followed

    def _best(self, cam, dets):
        dets = [d for d in dets if d.cls == "civet"]
        if not dets:
            return None
        tid = self.last_track.get(cam)
        same = [d for d in dets if tid is not None and d.track_id == tid]
        # stay with the same local track unless something far more certain appears
        if same and same[0].conf + self.margin >= max(d.conf for d in dets):
            return same[0]
        return max(dets, key=lambda d: d.conf)

    def select(self, per_cam):
        """per_cam: {cam_id: [Detection]} -> (cam_id, Detection) or (None, None).
        The current room keeps the civet unless another camera is clearly more
        certain (an animal in a doorway can show up in two views at once)."""
        best = {c: self._best(c, d) for c, d in per_cam.items()}
        best = {c: d for c, d in best.items() if d is not None}
        if not best:
            return None, None
        challenger = max(best, key=lambda c: best[c].conf)
        cur = best.get(self.cam)
        chosen = self.cam if cur is not None and best[challenger].conf < cur.conf + self.margin else challenger
        self.cam = chosen
        self.last_track[chosen] = best[chosen].track_id
        return chosen, best[chosen]
