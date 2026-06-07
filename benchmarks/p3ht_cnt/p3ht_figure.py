"""
P3HT-CNT benchmark figure: Regret vs Cost + Facility Sessions.

"""

import sys, pickle, io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

# Google Colab file upload helper
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False          # running locally - set PKL_PATH below instead

PKL_PATH = "p3ht_results.pkl"   # pickle, change if running locally

# ── 1. Load data ─────────────────────────────────────────────────────────────

class _SafeUnpickler(pickle.Unpickler):
    """Loads the pkl even when the original benchmark module is absent."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (), {
                "__repr__": lambda self: f"<{name}>",
                "__module__": module,
            })

if IN_COLAB:
    print("▶  Please upload your  p3ht_results.pkl  file …")
    uploaded = files.upload()
    pkl_bytes = next(iter(uploaded.values()))
    raw = _SafeUnpickler(io.BytesIO(pkl_bytes)).load()
else:
    with open(PKL_PATH, "rb") as fh:
        raw = _SafeUnpickler(fh).load()

results = raw["results"]          # dict  method → list[list[IterationLog]]

# ── 2. Nature-style rcParams (Liberation Sans) ───────────────────────────────
# Nature requires Helvetica/Arial; Liberation Sans is metric-compatible with
# Arial. Fall back gracefully if not installed.
_available = {f.name for f in fm.fontManager.ttflist}
for _candidate in ("Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"):
    if _candidate in _available:
        FONT = _candidate
        break
else:
    FONT = "DejaVu Sans"
print(f"Using font: {FONT}")

SZ_SM  = 7
SZ_MD  = 8
SZ_LG  = 9

mpl.rcParams.update({
    # font
    "font.family":        "sans-serif",
    "font.sans-serif":    [FONT, "Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          SZ_MD,
    "axes.titlesize":     SZ_LG,
    "axes.labelsize":     SZ_MD,
    "xtick.labelsize":    SZ_SM,
    "ytick.labelsize":    SZ_SM,
    "legend.fontsize":    SZ_SM,
    # lines / markers
    "lines.linewidth":    1.2,
    "lines.markersize":   3.5,
    # axes
    "axes.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    # grid
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.4,
    "grid.color":         "#b0b0b0",
    # layout
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype":       42,      # embeds fonts - required by Nature
    "ps.fonttype":        42,
})

# ── 3. Colour / style palette ────────────────────────────────────────────────
PALETTE = {
    "QS-MFBO+FAVOP": {"color": "#d62728", "ls": "-",  "lw": 1.4, "zorder": 4},
    "QS-MFBO":        {"color": "#2ca02c", "ls": "--", "lw": 1.2, "zorder": 3},
    "MF-MES":         {"color": "#1f77b4", "ls": ":",  "lw": 1.2, "zorder": 2},
}
FILL_ALPHA = 0.15
BAR_COLORS = {"MF": "#f5a623", "HF": "#9467bd"}   # orange / purple

METHODS    = ["QS-MFBO+FAVOP", "QS-MFBO", "MF-MES"]
FIDELITIES = {0.5: "MF", 1.0: "HF"}               # key → label

DISPLAY_METHOD_NAMES = {
    "QS-MFBO+FAVOP": "QS-MFBO + FAVP",
    "QS-MFBO":       "QS-MFBO",
    "MF-MES":        "MFBO",
}

# ── 4. Helper: build regret-vs-cost curves ───────────────────────────────────

def build_regret_curves(results, methods, n_grid=300, cost_min=25, cost_max=270,
                         log_floor=1e-3):

    cost_grid = np.linspace(cost_min, cost_max, n_grid)
    out = {}
    for m in methods:
        runs = results[m]
        interp_regrets = []
        for run in runs:
            costs   = np.array([it.cumulative_cost for it in run])
            regrets = np.array([it.true_regret     for it in run])
            # running minimum (regret is monotone non-increasing in practice)
            regrets = np.minimum.accumulate(regrets)
            # clamp small/zero regrets to a floor for log-scale plotting,
            # without dropping them from the median (NaN would do that).
            regrets = np.maximum(regrets, log_floor)
            fn = interp1d(costs, regrets, kind="previous",
                          bounds_error=False,
                          fill_value=(regrets[0], regrets[-1]))
            interp_regrets.append(fn(cost_grid))

        stack  = np.array(interp_regrets)   # (n_runs, n_grid)
        median = np.nanmedian(stack, axis=0)
        lo     = np.nanpercentile(stack, 25, axis=0)
        hi     = np.nanpercentile(stack, 75, axis=0)
        out[m] = (cost_grid, median, lo, hi)
    return out

# ── 5. Helper: compute session counts ────────────────────────────────────────

def compute_sessions(results, methods, fidelities):

    out = {}
    for m in methods:
        out[m] = {}
        for fid_key, fid_label in fidelities.items():
            counts = []
            for run in results[m]:
                last = run[-1]
                ns   = last.n_sessions if hasattr(last, "n_sessions") else {}
                counts.append(ns.get(fid_key, 0))
            out[m][fid_label] = (np.mean(counts), np.std(counts))
    return out

# ── 6. Helper: per-seed final regret (for annotation) ────────────────────────

def final_regrets(results, method):
    return np.array([run[-1].true_regret for run in results[method]])


def build_per_seed_trajectories(results, method, cost_grid, log_floor=1e-3):

    runs = results[method]
    out = []
    for run in runs:
        costs   = np.array([it.cumulative_cost for it in run])
        regrets = np.array([it.true_regret     for it in run])
        regrets = np.minimum.accumulate(regrets)
        regrets = np.maximum(regrets, log_floor)
        fn = interp1d(costs, regrets, kind="previous",
                      bounds_error=False,
                      fill_value=(regrets[0], regrets[-1]))
        out.append(fn(cost_grid))
    return np.array(out)

# ── 7. Build data ────────────────────────────────────────────────────────────
COST_MAX  = 270   # budget=250 plus flush overhead

LOG_FLOOR = 0.04

curves   = build_regret_curves(results, METHODS, cost_min=25, cost_max=COST_MAX,
                                log_floor=LOG_FLOOR)
sessions = compute_sessions(results, METHODS, FIDELITIES)


favp_grid = curves["QS-MFBO+FAVOP"][0]
favp_per_seed = build_per_seed_trajectories(results, "QS-MFBO+FAVOP",
                                             favp_grid, log_floor=LOG_FLOOR)

favp_finals = final_regrets(results, "QS-MFBO+FAVOP")
favp_median = float(np.median(favp_finals))
print(f"QS-MFBO+FAVP per-seed final regrets: "
      f"{', '.join(f'{r:.3f}' for r in favp_finals)}")
print(f"Median final regret (QS-MFBO+FAVP): {favp_median:.4f}")

# ── 8. Figure ────────────────────────────────────────────────────────────────
FIG_W_IN = 220 / 25.4   # ≈ 7.2 in (Nature double-column)
FIG_H_IN = 70  / 25.4   # ≈ 2.2 in

fig, axes = plt.subplots(
    1, 2,
    figsize=(FIG_W_IN, FIG_H_IN),
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.38},
)

# ─── Panel A: Regret vs Cost ─────────────────────────────────────────────────
ax = axes[0]

# Plot baselines (QS-MFBO and MF-MES) with shaded IQR bands as before.
for m in ("QS-MFBO", "MF-MES"):
    cost_grid, median, lo, hi = curves[m]
    s = PALETTE[m]
    ax.plot(cost_grid, median,
            color=s["color"], ls=s["ls"], lw=s["lw"],
            zorder=s["zorder"], label=DISPLAY_METHOD_NAMES[m])
    ax.fill_between(cost_grid, lo, hi,
                    color=s["color"], alpha=FILL_ALPHA, zorder=s["zorder"] - 1)


favp_style = PALETTE["QS-MFBO+FAVOP"]
for s_idx in range(favp_per_seed.shape[0]):
    ax.plot(favp_grid, favp_per_seed[s_idx, :],
            color=favp_style["color"], lw=0.5, alpha=0.25,
            zorder=favp_style["zorder"] - 1)

# Bold median on top
cost_grid, median, _, _ = curves["QS-MFBO+FAVOP"]
ax.plot(cost_grid, median,
        color=favp_style["color"], ls=favp_style["ls"], lw=favp_style["lw"],
        zorder=favp_style["zorder"] + 1,
        label=DISPLAY_METHOD_NAMES["QS-MFBO+FAVOP"])

ax.set_yscale("log")
ax.set_xlim(25, COST_MAX + 5)
ax.set_ylim(LOG_FLOOR, 4.0)
ax.set_xlabel("Cumulative cost", labelpad=3)
ax.set_ylabel("True regret",     labelpad=3)
ax.set_title("P3HT-CNT: Regret vs Cost", pad=4)

# minor grid on log axis
ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(subs="all", numticks=10))
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.grid(which="minor", linewidth=0.2, alpha=0.25)

# panel label
ax.text(-0.12, 1.04, "a", transform=ax.transAxes,
        fontsize=10, va="top", fontweight="bold")


favp_color = PALETTE["QS-MFBO+FAVOP"]["color"]
final_x = COST_MAX
ax.annotate(f"median = {favp_median:.2f}",
            xy=(final_x - 5, favp_median),
            xytext=(final_x - 80, 0.13),
            fontsize=6,
            color=favp_color,
            ha="left",
            arrowprops=dict(arrowstyle="-", color=favp_color, lw=0.5))

# ─── Panel B: Facility Sessions bar chart ────────────────────────────────────
ax2 = axes[1]

n_methods  = len(METHODS)
n_fid      = len(FIDELITIES)
bar_width  = 0.30
group_gap  = 0.10
x_centers  = np.arange(n_methods) * (n_fid * bar_width + group_gap + 0.10)

fid_labels = list(FIDELITIES.values())   # ["MF", "HF"]

for fi, fid_label in enumerate(fid_labels):
    offset = (fi - (n_fid - 1) / 2) * bar_width
    means  = [sessions[m][fid_label][0] for m in METHODS]
    stds   = [sessions[m][fid_label][1] for m in METHODS]
    color  = BAR_COLORS[fid_label]

    ax2.bar(
        x_centers + offset, means,
        width=bar_width,
        color=color, alpha=0.88,
        label=f"{fid_label} sessions",
        zorder=3,
    )
    ax2.errorbar(
        x_centers + offset, means, yerr=stds,
        fmt="none", color="k", capsize=2.5,
        linewidth=0.8, capthick=0.8, zorder=4,
    )

ax2.set_xticks(x_centers)
ax2.set_xticklabels([DISPLAY_METHOD_NAMES[m] for m in METHODS],
                     fontsize=SZ_SM - 0.5)
ax2.set_ylabel("Sessions", labelpad=3)
ax2.set_title("Facility Sessions", pad=4)
ax2.grid(axis="x", visible=False)

ax2.text(-0.18, 1.04, "b", transform=ax2.transAxes,
         fontsize=10, va="top", fontweight="bold")

# ── 9. Shared horizontal legend below both panels 
legend_elements = [
    Line2D([0], [0], color=PALETTE[m]["color"],
           ls=PALETTE[m]["ls"], lw=PALETTE[m]["lw"],
           label=DISPLAY_METHOD_NAMES[m])
    for m in METHODS
] + [
    mpl.patches.Patch(color=BAR_COLORS[fl], alpha=0.88,
                      label=f"{fl} sessions")
    for fl in fid_labels
]

fig.legend(legend_elements, [h.get_label() for h in legend_elements],
           loc="lower center", ncol=len(legend_elements),
           bbox_to_anchor=(0.5, -0.08), frameon=True,
           framealpha=0.9, edgecolor="#dddddd", fontsize=SZ_SM)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18, wspace=0.38)

# ── 10. Save
PDF_NAME = "p3ht_cnt_benchmark.pdf"
PNG_NAME = "p3ht_cnt_benchmark.png"

fig.savefig(PDF_NAME, format="pdf")
fig.savefig(PNG_NAME, format="png", dpi=300)
print(f"Saved  {PDF_NAME}  and  {PNG_NAME}")
plt.show()

#this is googlecolab code
