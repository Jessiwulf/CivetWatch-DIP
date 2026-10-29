"""Small GRU that reads 4 s of body-dot features (8 samples at 2 fps) and says
grooming / territorial marking / urination / defecation / other.

Training data = frames.csv files written by `civetwatch run`, with a
`true_label` column filled from keeper review (`civetwatch export-labels`)
or from CVAT behavior annotations.
"""
import numpy as np
import pandas as pd

from . import ethogram as E
from .features import FEATURES

CLASSES = ["other"] + E.SEQUENCE_LABELS


def _torch():
    import torch
    from torch import nn
    return torch, nn


def make_model(n_features, n_classes, hidden=64):
    torch, nn = _torch()

    class BehaviorGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, n_classes))

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])

    return BehaviorGRU()


def load_model(path):
    torch, _ = _torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(len(ckpt["features"]), len(ckpt["classes"]), ckpt.get("hidden", 64))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict(model, meta, window):
    torch, _ = _torch()
    x = (np.asarray(window, np.float32) - meta["mean"]) / meta["std"]
    with torch.no_grad():
        logits = model(torch.from_numpy(x[None]))
        return torch.softmax(logits, dim=1)[0].numpy()


def build_windows(frames, window):
    """Sliding windows over consecutive samples; label = label of the last sample."""
    X, y = [], []
    for _, run in frames.groupby((frames["t"].diff().dt.total_seconds() > 1.0).cumsum()):
        feats = run[FEATURES].to_numpy(np.float32)
        labels = run["true_label"].tolist()
        for end in range(window, len(run) + 1):
            lab = labels[end - 1]
            if not isinstance(lab, str) or lab == E.UNKNOWN:
                continue
            X.append(feats[end - window:end])
            y.append(CLASSES.index(lab) if lab in CLASSES else 0)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def train(csv_paths, out_path, window=8, epochs=40, lr=1e-3, hidden=64, seed=0):
    torch, nn = _torch()
    torch.manual_seed(seed)
    frames = pd.concat([pd.read_csv(p, parse_dates=["t"]) for p in csv_paths], ignore_index=True)
    if "true_label" not in frames:
        raise ValueError("frames.csv needs a 'true_label' column (run `civetwatch export-labels` after review)")
    frames = frames.sort_values("t").reset_index(drop=True)
    for f in FEATURES:
        frames[f] = frames.get(f, 0.0)
    X, y = build_windows(frames, window)
    if len(X) < 20:
        raise ValueError(f"Only {len(X)} labelled windows - label more clips first")

    split = int(len(X) * 0.8)           # time-ordered split: test on later footage
    mean = X[:split].reshape(-1, X.shape[2]).mean(0)
    std = X[:split].reshape(-1, X.shape[2]).std(0) + 1e-6
    norm = lambda a: torch.from_numpy((a - mean) / std)
    Xtr, ytr, Xva, yva = norm(X[:split]), torch.from_numpy(y[:split]), norm(X[split:]), torch.from_numpy(y[split:])

    counts = np.bincount(y[:split], minlength=len(CLASSES)).astype(np.float32)
    weights = torch.tensor(np.where(counts > 0, counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), 0.0))
    model = make_model(X.shape[2], len(CLASSES), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights.float())

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xva).argmax(1) == yva).float().mean().item() if len(Xva) else float("nan")
            print(f"epoch {epoch + 1:3d}  loss {loss.item():.3f}  val acc {acc:.3f}")

    model.eval()
    report = {}
    with torch.no_grad():
        pred = model(Xva).argmax(1).numpy() if len(Xva) else np.array([])
    for k, name in enumerate(CLASSES):
        m = y[split:] == k
        if m.any():
            report[name] = {"n": int(m.sum()), "recall": float((pred[m] == k).mean())}
    torch.save({"state_dict": model.state_dict(), "features": FEATURES, "classes": CLASSES,
                "mean": mean, "std": std, "window": window, "hidden": hidden}, out_path)
    return report
