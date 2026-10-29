"""charts.png: one-day ethogram timeline, time per room, hourly frequency
heat map and the time budget."""
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from . import ethogram as E

# palette validated for colour-blind separation (dataviz validate_palette.js)
INACTIVE_C = "#2A6FA0"
ACTIVE_C = "#D0773A"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BAR = "#3d5a3a"
HEAT = ["#fcfcfb", "#d6e6f2", "#a9cbe4", "#6fa6cf", "#2A6FA0", "#164a70"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.title.set_color(INK)


def make_charts(result, path, title):
    blocks, events, hourly, s = result["blocks"], result["events"], result["hourly"], result["summary"]
    fig = plt.figure(figsize=(14, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], width_ratios=[1.2, 1.4, 1], hspace=0.42, wspace=0.35, bottom=0.12)
    fig.suptitle(title, x=0.06, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.06, 0.925, f"Active {s['active_frac']:.0%} of the time it was seen  ·  "
             f"{s['counts'].get(E.FEEDING, 0)} feeding visits  ·  {s['to_review']} events to check by a person",
             fontsize=10, color=INK2)
    fig.text(0.06, 0.02, "Filled dot = counted.  Hollow dot = the app is unsure, check it in the dashboard.  "
             "Grooming, scent marking, urinating and defecating are only counted after a person confirms them "
             "(dashboard > Quick check).", fontsize=8.5, color=INK2)

    # ---- 1. timeline ----
    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    freq_rows = E.FREQUENCY_LABELS
    lanes = ["State", "Not visible"] + [E.DISPLAY[b] for b in freq_rows]
    y_of = {name: len(lanes) - 1 - i for i, name in enumerate(lanes)}
    span_days = max(1e-9, mdates.date2num(s["end"]) - mdates.date2num(s["start"]))
    axes_px = ax.get_position().width * fig.get_figwidth() * 140          # saved at dpi=140
    for _, b in blocks.iterrows():
        c = {E.INACTIVE: INACTIVE_C, E.ACTIVE: ACTIVE_C}.get(b["state"], MUTED)
        x0, x1 = mdates.date2num(b["start"]), mdates.date2num(b["stop"])
        ax.barh(y_of["State"], x1 - x0, left=x0, height=0.7, color=c, edgecolor=SURFACE, linewidth=1.5)
        # label only when the text fits inside its own bar (short neighbours overlapped before)
        label = f"{b['state']} {b['duration_s'] / 60:.0f} min"
        bar_px = (x1 - x0) / span_days * axes_px
        if bar_px >= len(label) * 7.5 + 16:
            ax.text((x0 + x1) / 2, y_of["State"], label, ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    if len(events):
        for _, ev in events[events["behavior"] == E.NOT_VISIBLE].iterrows():
            x0, x1 = mdates.date2num(ev["start"]), mdates.date2num(ev["stop"])
            ax.barh(y_of["Not visible"], x1 - x0, left=x0, height=0.5, color=MUTED)
        for beh in freq_rows:
            sel = events[events["behavior"] == beh]
            if len(sel):
                review = sel["needs_review"].astype(bool)
                ax.scatter(mdates.date2num(sel.loc[~review, "start"]), [y_of[E.DISPLAY[beh]]] * int((~review).sum()),
                           s=46, color=BAR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
                ax.scatter(mdates.date2num(sel.loc[review, "start"]), [y_of[E.DISPLAY[beh]]] * int(review.sum()),
                           s=46, facecolor=SURFACE, edgecolor=BAR, linewidth=1.5, zorder=3)
    ax.set_yticks(list(y_of.values()), list(y_of.keys()))
    ax.set_xlim(mdates.date2num(s["start"]), mdates.date2num(s["end"]))
    span_h = (s["end"] - s["start"]).total_seconds() / 3600
    if span_h > 3:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(span_h // 12))))
    else:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, int(span_h * 60 // 12))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("One day of behavior  (filled dot = counted, hollow = please check)", loc="left", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=INACTIVE_C), plt.Rectangle((0, 0), 1, 1, color=ACTIVE_C),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    ax.legend(handles, ["Inactive", "Active", "Not visible"], loc="upper right", bbox_to_anchor=(1, 1.16),
              ncol=3, frameon=False, fontsize=8, labelcolor=INK2)

    # ---- 2. time per room ----
    ax = fig.add_subplot(gs[1, 0])
    _style(ax)
    rooms = sorted(s["room_share"], key=lambda r: (r == "Not seen", r))
    vals = [s["room_share"][r] * 100 for r in rooms]
    ypos = np.arange(len(rooms))[::-1]
    ax.barh(ypos, vals, color=[MUTED if r == "Not seen" else BAR for r in rooms], height=0.55)
    for y, v in zip(ypos, vals):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ypos, rooms)
    ax.set_xlim(0, max(vals + [10]) * 1.2)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Time per room  (Not seen = no camera saw it)", loc="left", fontsize=11)

    # ---- 3. hourly frequency heat map ----
    ax = fig.add_subplot(gs[1, 1])
    _style(ax)
    cols = ["Feeding", "Drinking", "Territorial marking", "Grooming", "Urine", "Feces"]
    if len(hourly):
        data = hourly[cols].to_numpy().T.astype(float)
        vmax = max(1.0, data.max())
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", HEAT)
        ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j]:
                    ax.text(j, i, int(data[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if data[i, j] > vmax * 0.55 else INK)
        ax.set_xticks(range(len(hourly)), [h.split("-")[0].split(".")[0] for h in hourly["hour"]], fontsize=7)
        names = {"Urine": "Urinating", "Feces": "Defecating"}
        ax.set_yticks(range(len(cols)), [names.get(c, c) for c in cols], fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(hourly)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cols)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    ax.set_xlabel("hour of the day (number in a box = how many times it started in that hour)", fontsize=8, color=MUTED)
    ax.set_title("How many times, per hour", loc="left", fontsize=11)

    # ---- 4. time budget ----
    ax = fig.add_subplot(gs[1, 2])
    _style(ax)
    budget = {E.DISPLAY[k]: v / 60 for k, v in sorted(s["rest_types_s"].items(), key=lambda kv: -kv[1])}
    budget["Active"] = s["active_s"] / 60
    names = list(budget)[::-1]
    vals = [budget[n] for n in names]
    colors = [ACTIVE_C if n == "Active" else INACTIVE_C for n in names]
    ax.barh(range(len(names)), vals, color=colors, height=0.55)
    for y, v in enumerate(vals):
        ax.text(v + max(vals + [1]) * 0.02, y, f"{v:.0f} min", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, max(vals + [1]) * 1.3)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Time budget  (while it was seen)", loc="left", fontsize=11)

    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
