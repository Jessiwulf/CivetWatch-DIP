"""Keep long jobs from overloading a laptop.

Hours of 100 % load on every core (plus the GPU) can overheat a thin laptop
and crash it (e.g. CLOCK_WATCHDOG_TIMEOUT). The `laptop` profile, the
default, therefore:

* caps the CPU threads used by OpenCV / NumPy / PyTorch / FFmpeg,
* lowers the process priority so Windows stays responsive,
* works in a duty cycle (e.g. 70 % work, 30 % idle) so the CPU can cool,
* pauses whenever the GPU (which shares the laptop's cooling) gets hot,
* lets long runs save every hour so a crash only loses the current hour.

`profile: full` removes the limits (desktop with good cooling).
"""
import os
import shutil
import subprocess
import sys
import time

LAPTOP = {"threads": 4, "priority": "below_normal", "duty_cycle": 0.7, "gpu_temp_pause": 78,
          "gpu_temp_resume": 68, "chunk_min": 60, "train_epoch_pause_s": 20, "train_batch": 8,
          "train_workers": 1}
FULL = {"threads": 0, "priority": "normal", "duty_cycle": 1.0, "gpu_temp_pause": 90,
        "gpu_temp_resume": 85, "chunk_min": 60, "train_epoch_pause_s": 0, "train_batch": 16,
        "train_workers": 2}

_settings = dict(LAPTOP)
BELOW_NORMAL = 0x00004000          # Windows BELOW_NORMAL_PRIORITY_CLASS


def settings():
    return _settings


def apply(cfg):
    """Call once at start-up, before OpenCV / PyTorch do any work."""
    r = cfg.get("resources") or {}
    base = FULL if r.get("profile") == "full" else LAPTOP
    _settings.clear()
    _settings.update(base)
    _settings.update({k: v for k, v in r.items() if k != "profile" and v is not None})
    _settings["profile"] = r.get("profile", "laptop")
    n = int(_settings["threads"] or 0)
    if n > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(n)
        import cv2
        cv2.setNumThreads(n)
        if "torch" in sys.modules:
            sys.modules["torch"].set_num_threads(n)
    if _settings["priority"] == "below_normal":
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        except Exception:
            pass
    return _settings


def torch_threads():
    """PyTorch reads OMP_NUM_THREADS at import; call after importing torch to be sure."""
    n = int(_settings.get("threads") or 0)
    if n > 0 and "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n)


def subprocess_flags():
    """creationflags for child processes (FFmpeg) so they inherit the low priority."""
    return BELOW_NORMAL if os.name == "nt" and _settings.get("priority") == "below_normal" else 0


def gpu_temperature():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, creationflags=subprocess_flags())
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Throttle:
    """tick() once per unit of work (one sample, one epoch). It sleeps so that
    work takes `duty_cycle` of the wall time, and waits while the GPU is hot."""

    def __init__(self, log=print, check_every_s=30):
        s = _settings
        self.duty = min(1.0, max(0.1, float(s["duty_cycle"])))
        self.hot, self.cool = s["gpu_temp_pause"], s["gpu_temp_resume"]
        self.check_every = check_every_s
        self.log = log
        self.last = time.perf_counter()
        self.next_check = time.time() + check_every_s
        self.paused_s = 0.0

    def tick(self):
        now = time.perf_counter()
        work = now - self.last
        if self.duty < 1.0:
            rest = work * (1.0 - self.duty) / self.duty
            time.sleep(rest)
            self.paused_s += rest
        if time.time() >= self.next_check:
            self.next_check = time.time() + self.check_every
            self.wait_if_hot()
        self.last = time.perf_counter()

    def wait_if_hot(self):
        temp = gpu_temperature()
        if temp is None or temp < self.hot:
            return
        if self.log:
            self.log(f"  [cool-down] GPU at {temp:.0f} C - pausing until it is below {self.cool} C", flush=True)
        t0 = time.time()
        while temp is not None and temp >= self.cool:
            time.sleep(15)
            temp = gpu_temperature()
        self.paused_s += time.time() - t0
        if self.log:
            self.log(f"  [cool-down] resumed after {time.time() - t0:.0f} s", flush=True)
