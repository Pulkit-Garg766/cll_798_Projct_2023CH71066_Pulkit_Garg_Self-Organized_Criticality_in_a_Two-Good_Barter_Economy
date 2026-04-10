"""
Main experiment: connectivity sweep and plot generation.
Figures are saved next to this script.

Usage:
    python experiment.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from soc_sim import run_simulation, extract_cascades, fit_powerlaw, logbin_pdf

plt.rcParams.update({
    'font.size': 11, 'figure.dpi': 110,
    'axes.grid': True, 'grid.alpha': 0.3,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- EXPERIMENT PARAMETERS ----------------
N = 100          # agents
T = 1000       # ticks per run
RUNS = 200       # runs per k value (avalanches pooled across runs)
K_VALUES = [2, 4, 8]
WINDOW = 1       # avalanche grouping window (ticks)
K_MAIN = 4       # which k to highlight in fig1

print(f"Running {len(K_VALUES)} k-values x {RUNS} runs x {T} ticks x N={N}")
print("=" * 60)

# ---------------- RUN THE SWEEP ----------------
results = {}
for k in K_VALUES:
    all_avalanches = []
    total_deaths = 0
    for run in range(RUNS):
        seed = 100 * k + run
        deaths = run_simulation(N=N, k=k, T=T, seed=seed)
        avalanches = extract_cascades(deaths, window=WINDOW)
        all_avalanches.extend(avalanches)
        total_deaths += len(deaths)
    results[k] = all_avalanches
    alpha, _ = fit_powerlaw(all_avalanches, xmin=2)
    mx = max(all_avalanches) if all_avalanches else 0
    a_str = f"alpha={alpha:.3f}" if alpha else "fit failed"
    print(f"k={k:3d} | deaths={total_deaths:6d} | avalanches={len(all_avalanches):6d} | max={mx:4d} | {a_str}")

# ---------------- FIGURE 1: Main distribution with power-law fit (tail excluded) ----------------
fig, ax = plt.subplots(figsize=(7, 5))
sizes = results[K_MAIN]
centers, pdf = logbin_pdf(sizes, nbins=20)

# Plot all data points
ax.loglog(centers, pdf, 'o', markersize=8, color='#185FA5',
          label=f'Simulation (k={K_MAIN})', markeredgecolor='white')

# --- Tail exclusion: fit only the well-populated bulk ---
# A bin is considered "tail" if its pdf value is more than 10x below the peak,
# OR if it is one of the last 3 points (typically sparse).
if len(pdf) > 4:
    peak = pdf.max()
    # Keep bins whose pdf is at least 1% of peak, and drop the last 2 bins
    bulk_mask = (pdf >= peak * 0.01)
    bulk_mask[-2:] = False   # always drop the last two noisy points
    fit_centers = centers[bulk_mask]
    fit_pdf = pdf[bulk_mask]
else:
    fit_centers = centers
    fit_pdf = pdf

# Fit a straight line on log-log axes using only the bulk points
if len(fit_centers) >= 3:
    log_x = np.log10(fit_centers)
    log_y = np.log10(fit_pdf)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    tau = -slope
    xs = np.logspace(np.log10(fit_centers.min()), np.log10(fit_centers.max()), 50)
    ys = 10**intercept * xs**slope
    ax.loglog(xs, ys, '--', color='#A32D2D', linewidth=2.2,
              label=f'Power-law fit (bulk): tau = {tau:.2f}')
    # Highlight which points were used in the fit
    ax.loglog(fit_centers, fit_pdf, 'o', markersize=8,
              markerfacecolor='none', markeredgecolor='#A32D2D', markeredgewidth=1.5,
              label='Points used in fit')

ax.set_xlabel('Avalanche size $s$')
ax.set_ylabel('Probability density $P(s)$')
ax.set_title(f'Death avalanche size distribution (N={N}, k={K_MAIN})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig1_distribution.png")

# ---------------- FIGURE 2: All k distributions overlaid ----------------
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(K_VALUES)))
for k, c in zip(K_VALUES, colors):
    sizes = results[k]
    if len(sizes) < 5:
        continue
    centers, pdf = logbin_pdf(sizes, nbins=18)
    if len(centers) > 0:
        ax.loglog(centers, pdf, 'o-', color=c, label=f'k = {k}',
                  markersize=6, linewidth=1.2, alpha=0.9)
ax.set_xlabel('Avalanche size $s$')
ax.set_ylabel('Probability density $P(s)$')
ax.set_title('Avalanche distribution vs. network connectivity')
ax.legend(title='Mean degree')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_connectivity_sweep.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig2_connectivity_sweep.png")

# ---------------- FIGURE 3: Exponent and moments vs k ----------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

alphas, means, vars_, maxs = [], [], [], []
for k in K_VALUES:
    sizes = results[k]
    a, _ = fit_powerlaw(sizes, xmin=2)
    alphas.append(a if a else np.nan)
    means.append(np.mean(sizes) if sizes else 0)
    vars_.append(np.var(sizes) if sizes else 0)
    maxs.append(max(sizes) if sizes else 0)

axes[0].plot(K_VALUES, alphas, 'o-', color='#185FA5', markersize=9)
axes[0].set_xlabel('Mean degree $k$')
axes[0].set_ylabel('Power-law exponent tau')
axes[0].set_title('Critical exponent vs connectivity')
axes[0].set_xscale('log')

axes[1].plot(K_VALUES, means, 's-', color='#0F6E56', markersize=9, label='mean')
axes[1].plot(K_VALUES, maxs, '^-', color='#A32D2D', markersize=9, label='max')
axes[1].set_xlabel('Mean degree $k$')
axes[1].set_ylabel('Avalanche size')
axes[1].set_title('Mean and max avalanche size')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend()

axes[2].plot(K_VALUES, vars_, 'D-', color='#993C1D', markersize=9)
axes[2].set_xlabel('Mean degree $k$')
axes[2].set_ylabel('Variance of avalanche size')
axes[2].set_title('Variance (susceptibility proxy)')
axes[2].set_xscale('log')
axes[2].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_summary.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig3_summary.png")

# ---------------- FIGURE 4: CCDF ----------------
fig, ax = plt.subplots(figsize=(7, 5))
for k, c in zip(K_VALUES, colors):
    sizes = np.array(sorted(results[k], reverse=True))
    if len(sizes) < 5:
        continue
    ccdf = np.arange(1, len(sizes) + 1) / len(sizes)
    ax.loglog(sizes, ccdf, '-', color=c, label=f'k = {k}', linewidth=1.5)
ax.set_xlabel('Avalanche size $s$')
ax.set_ylabel(r'$P(S \geq s)$  (CCDF)')
ax.set_title('Complementary cumulative distribution of avalanches')
ax.legend(title='Mean degree')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_ccdf.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig4_ccdf.png")

# ---------------- SUMMARY TABLE ----------------
print("\nSummary:")
print(f"{'k':>4} {'alpha':>8} {'mean':>8} {'var':>12} {'max':>6}")
for k, a, m, v, mx in zip(K_VALUES, alphas, means, vars_, maxs):
    a_s = f"{a:.3f}" if a and not np.isnan(a) else "  -  "
    print(f"{k:>4} {a_s:>8} {m:>8.2f} {v:>12.2f} {mx:>6}")