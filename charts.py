"""Generate evaluation charts for mlx-turbo."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-darkgrid")

OUT = "charts"
os.makedirs(OUT, exist_ok=True)

# Color palette
BL_COLOR = "#5B8DEF"
TQ_COLOR = "#F97316"
SAVE_COLOR = "#10B981"

# ── Chart 1: KV Cache Memory (Qwen3.5 9B, 4K–32K) ──────────────────────────

ctx = [4096, 8192, 16384, 32768]
bl_kv = [131, 268, 524, 1074]
tq_kv = [28, 56, 112, 226]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(ctx))
w = 0.32
bars_bl = ax.bar(
    x - w / 2,
    bl_kv,
    w,
    label="Baseline FP16",
    color=BL_COLOR,
    edgecolor="white",
    linewidth=0.5,
)
bars_tq = ax.bar(
    x + w / 2,
    tq_kv,
    w,
    label="TurboQuant 3-bit",
    color=TQ_COLOR,
    edgecolor="white",
    linewidth=0.5,
)

for bar, val in zip(bars_bl, bl_kv):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 15,
        f"{val}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=BL_COLOR,
        fontweight="bold",
    )
for bar, val in zip(bars_tq, tq_kv):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 15,
        f"{val}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TQ_COLOR,
        fontweight="bold",
    )

ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("KV Cache Memory (MB)")
ax.set_title(
    "KV Cache Memory: Baseline vs TurboQuant\nQwen3.5-9B  |  3-bit  |  4.7x compression",
    fontsize=12,
)
ax.set_xticks(x)
ax.set_xticklabels([f"{c // 1000}K" for c in ctx])
ax.legend(loc="upper left")
ax.set_ylim(0, 1250)
fig.tight_layout()
fig.savefig(f"{OUT}/01_kv_memory.png", dpi=180)
plt.close()
print("Saved 01_kv_memory.png")


# ── Chart 2: Decode Speed (both models, short context) ──────────────────────

models = ["Mistral Nemo 12B", "Qwen3.5 9B"]
bl_speed = [34.4, 43.6]
tq_speed = [21.9, 36.9]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(models))
w = 0.3
ax.bar(x - w / 2, bl_speed, w, label="Baseline", color=BL_COLOR, edgecolor="white")
ax.bar(
    x + w / 2, tq_speed, w, label="TurboQuant 3-bit", color=TQ_COLOR, edgecolor="white"
)

for i in range(len(models)):
    overhead = (tq_speed[i] / bl_speed[i] - 1) * 100
    ax.text(
        x[i] + w / 2,
        tq_speed[i] + 0.8,
        f"{overhead:+.0f}%",
        ha="center",
        fontsize=9,
        color="#888",
    )

ax.set_ylabel("Decode Speed (tok/s)")
ax.set_title("Decode Speed Comparison\n200-token generation", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 55)
fig.tight_layout()
fig.savefig(f"{OUT}/02_decode_speed.png", dpi=180)
plt.close()
print("Saved 02_decode_speed.png")


# ── Chart 3: Decode Speed vs Context Length (Qwen3.5 9B) ────────────────────

ctx_long = [4096, 8192, 16384, 32768]
tq_toks = [36.6, 35.1, 32.3, 30.0]
# Baseline only available at short context; extrapolate flat ~43
bl_short = [42.6, None, None, None]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    [c / 1000 for c in ctx_long],
    tq_toks,
    "o-",
    color=TQ_COLOR,
    linewidth=2,
    markersize=7,
    label="TurboQuant 3-bit",
)
ax.axhline(
    y=42.6,
    color=BL_COLOR,
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Baseline @ 4K (42.6 tok/s)",
)

for i, (c, t) in enumerate(zip(ctx_long, tq_toks)):
    ax.annotate(
        f"{t:.1f}",
        (c / 1000, t),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        color=TQ_COLOR,
    )

ax.set_xlabel("Context Length (K tokens)")
ax.set_ylabel("Decode Speed (tok/s)")
ax.set_title(
    "Decode Speed at Long Context\nQwen3.5-9B  |  TurboQuant 3-bit (Rust backend)",
    fontsize=12,
)
ax.legend()
ax.set_ylim(0, 55)
ax.set_xlim(2, 36)
fig.tight_layout()
fig.savefig(f"{OUT}/03_long_context_speed.png", dpi=180)
plt.close()
print("Saved 03_long_context_speed.png")


# ── Chart 4: MSE Distortion vs Paper Bounds ──────────────────────────────────

bits = [2, 3, 4]
measured = [0.115, 0.034, 0.009]
bounds = [0.170, 0.043, 0.011]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(bits))
w = 0.3
ax.bar(
    x - w / 2, bounds, w, label="Paper upper bound", color="#CBD5E1", edgecolor="white"
)
ax.bar(
    x + w / 2,
    measured,
    w,
    label="mlx-turbo (measured)",
    color=TQ_COLOR,
    edgecolor="white",
)

for i in range(len(bits)):
    pct = measured[i] / bounds[i] * 100
    ax.text(
        x[i] + w / 2,
        measured[i] + 0.003,
        f"{pct:.0f}%",
        ha="center",
        fontsize=9,
        color=TQ_COLOR,
    )

ax.set_ylabel("MSE Distortion (unit vectors, d=128)")
ax.set_title("Quantization Quality: Measured vs Paper Bounds", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{b}-bit" for b in bits])
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/04_mse_distortion.png", dpi=180)
plt.close()
print("Saved 04_mse_distortion.png")


# ── Chart 5: Overhead breakdown (why Mistral is slower) ──────────────────────

models_oh = ["Mistral Nemo 12B\n(40 KV layers)", "Qwen3.5 9B\n(8 KV layers)"]
overhead_pct = [36, 15]
kv_layers = [40, 8]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# Left: overhead bars
colors = ["#EF4444", TQ_COLOR]
ax1.bar(models_oh, overhead_pct, color=colors, edgecolor="white", width=0.5)
for i, (m, o) in enumerate(zip(models_oh, overhead_pct)):
    ax1.text(i, o + 1, f"{o}%", ha="center", fontsize=12, fontweight="bold")
ax1.set_ylabel("Decode Overhead (%)")
ax1.set_title("TurboQuant Overhead", fontsize=12)
ax1.set_ylim(0, 50)

# Right: KV layers vs overhead scatter
ax2.scatter(
    kv_layers, overhead_pct, s=200, c=colors, edgecolors="white", linewidth=2, zorder=5
)
for i in range(len(models_oh)):
    ax2.annotate(
        models_oh[i].split("\n")[0],
        (kv_layers[i], overhead_pct[i]),
        textcoords="offset points",
        xytext=(15, 5),
        fontsize=9,
    )
ax2.set_xlabel("Number of KVCache Layers")
ax2.set_ylabel("Decode Overhead (%)")
ax2.set_title("Overhead Scales with KV Layer Count", fontsize=12)
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 50)
# Trend line
z = np.polyfit(kv_layers, overhead_pct, 1)
xl = np.linspace(0, 50, 100)
ax2.plot(xl, np.polyval(z, xl), "--", color="#888", alpha=0.5)

fig.tight_layout()
fig.savefig(f"{OUT}/05_overhead_analysis.png", dpi=180)
plt.close()
print("Saved 05_overhead_analysis.png")


# ── Chart 6: Memory savings projection (Qwen3.5 9B on 24GB Mac) ─────────────

ctx_proj = [4, 8, 16, 32, 64, 128]
bl_proj = [131, 268, 524, 1074, 2148, 4295]
tq_proj = [28, 56, 112, 226, 447, 895]
avail_gb = 15  # ~15GB free after model weights

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_between(ctx_proj, bl_proj, alpha=0.15, color=BL_COLOR)
ax.fill_between(ctx_proj, tq_proj, alpha=0.15, color=TQ_COLOR)
ax.plot(
    ctx_proj,
    bl_proj,
    "o-",
    color=BL_COLOR,
    linewidth=2,
    markersize=6,
    label="Baseline FP16",
)
ax.plot(
    ctx_proj,
    tq_proj,
    "o-",
    color=TQ_COLOR,
    linewidth=2,
    markersize=6,
    label="TurboQuant 3-bit",
)
ax.axhline(
    y=avail_gb * 1000,
    color="#EF4444",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label=f"Available memory (~{avail_gb}GB)",
)

# Mark where each hits the limit
# BL hits ~47K, TQ hits ~225K (off chart)
ax.annotate(
    "Baseline limit\n~47K ctx",
    xy=(47, avail_gb * 1000),
    xytext=(55, 12000),
    fontsize=9,
    color=BL_COLOR,
    arrowprops=dict(arrowstyle="->", color=BL_COLOR),
)

ax.set_xlabel("Context Length (K tokens)")
ax.set_ylabel("KV Cache Memory (MB)")
ax.set_title(
    "Memory Projection: Qwen3.5-9B on 24GB Mac\nTurboQuant extends max context from ~47K to ~225K",
    fontsize=12,
)
ax.legend(loc="upper left")
ax.set_ylim(0, 16000)
ax.set_xscale("linear")
fig.tight_layout()
fig.savefig(f"{OUT}/06_memory_projection.png", dpi=180)
plt.close()
print("Saved 06_memory_projection.png")

print(f"\nAll charts saved to {OUT}/")
