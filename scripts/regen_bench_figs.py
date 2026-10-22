#!/usr/bin/env python
"""
Regenerate benchmark figures using saved results (Track B only, no anchor).
Produces publication-ready PNGs in results/trussart/.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

REPO = Path('/Users/ronguy/Dropbox/Work/CyTOF/Code/som_ot_warp_repo_anchor')
RESULTS_DIR = REPO / 'results' / 'trussart'
FIG_DIR = Path('/Users/ronguy/Dropbox/Apps/Overleaf/CyPrep/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load saved metrics ──
def load_metrics(method_key):
    p = RESULTS_DIR / method_key / 'metrics.json'
    with open(p) as f:
        return json.load(f)

# ── Load corrected data ──
def load_corrected(method_key):
    p = RESULTS_DIR / method_key / 'corrected.parquet'
    return pd.read_parquet(p).values

# ── Load original data ──
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from benchmark_public import load_trussart

DATA_DIR = '/Users/ronguy/Dropbox/Work/CyTOF/TestDir/trussart_cytofruv/Norm_Raw_RUV1b_RUV3b_simult'
X_full, y_batch_full, y_sample_full, y_celltype_full, markers = load_trussart(DATA_DIR)

# Subsample same as notebook
SUBSAMPLE = 20000
rng = np.random.RandomState(42)
keep = []
for b in np.unique(y_batch_full):
    idx = np.where(y_batch_full == b)[0]
    n = min(SUBSAMPLE, len(idx))
    keep.append(rng.choice(idx, n, replace=False))
keep = np.sort(np.concatenate(keep))
X = X_full[keep].copy()
y_batch = y_batch_full[keep].copy()
y_sample = y_sample_full[keep].copy()
y_celltype = y_celltype_full[keep].copy()

# Drop Unassigned
keep_ct = y_celltype != 'Unassigned'
X = X[keep_ct]; y_batch = y_batch[keep_ct]
y_sample = y_sample[keep_ct]; y_celltype = y_celltype[keep_ct]
print(f'Loaded {len(X):,} cells, {len(markers)} markers')

# ── Build Track B results dict ──
track_b_keys = ['somot_noanchor', 'combat', 'mmdresnet']
display_names = {
    'somot_noanchor': 'SOM-OT-Warp',
    'combat': 'ComBat',
    'mmdresnet': 'MMD-ResNet',
}

all_results = {}
all_corrected = {}
metrics_before = None

for key in track_b_keys:
    try:
        m = load_metrics(key)
        all_results[key] = {
            'display_name': display_names[key],
            'elapsed_sec': m.get('elapsed_sec', 0),
            'before': m.get('before', {}),
            'after': m.get('after', {}),
        }
        all_corrected[key] = load_corrected(key)
        if metrics_before is None:
            metrics_before = m['before']
        print(f'  Loaded {key}')
    except Exception as e:
        print(f'  SKIP {key}: {e}')

method_colors = {
    'somot_noanchor': '#6495ED',
    'combat':         '#F08080',
    'mmdresnet':      '#90EE90',
}

# ════════════════════════════════════════════════════
# Figure 1: Bar chart metrics
# ════════════════════════════════════════════════════
print('\nGenerating metrics bar chart...')
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

plot_metrics = [
    ('knn_entropy', 'kNN Entropy (↑ better)'),
    ('knn_ilisi', 'iLISI (↑ better)'),
    ('silhouette_batch', 'Sil. Batch (↓ better)'),
    ('silhouette_bio', 'Sil. Bio (↑ better)'),
]

for ax, (mkey, title) in zip(axes, plot_metrics):
    names = ['Uncorrected']
    vals = [float(metrics_before.get(mkey, 0))]
    colors = ['#B0B0B0']

    for key, res in all_results.items():
        after = res.get('after', {})
        if mkey in after:
            names.append(res['display_name'])
            vals.append(float(after[mkey]))
            colors.append(method_colors.get(key, '#999999'))

    bars = ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
out = FIG_DIR / 'bench_metrics.png'
plt.savefig(str(out), dpi=200, bbox_inches='tight')
plt.close()
print(f'  → {out}')

# ════════════════════════════════════════════════════
# Figure 2: Per-marker EMD heatmap
# ════════════════════════════════════════════════════
print('Generating EMD heatmap...')
emd_before = metrics_before['per_marker_emd']['per_marker']

reduction_rows = []
for key, res in all_results.items():
    after = res.get('after', {})
    if not after:
        continue
    emd_after = after.get('per_marker_emd', {}).get('per_marker', {})
    label = res['display_name']
    for m in markers:
        bv = float(emd_before.get(m, 0))
        av = float(emd_after.get(m, 0))
        reduction_rows.append({
            'Method': label, 'Marker': m,
            'EMD_reduction': (bv - av) / (bv + 1e-12)
        })

red_df = pd.DataFrame(reduction_rows)
pivot = red_df.pivot(index='Marker', columns='Method', values='EMD_reduction')
pivot = pivot.loc[markers]

fig, ax = plt.subplots(figsize=(max(6, len(all_results) * 2.5),
                                max(8, len(markers) * 0.35)))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'EMD Reduction'},
            annot_kws={'fontsize': 8})
ax.set_title('EMD Reduction per Marker  (green = improved, red = worsened)')
ax.set_ylabel('')
plt.tight_layout()
out = FIG_DIR / 'bench_emd_heatmap.png'
plt.savefig(str(out), dpi=200, bbox_inches='tight')
plt.close()
print(f'  → {out}')

# ════════════════════════════════════════════════════
# Figure 3: Runtime comparison
# ════════════════════════════════════════════════════
print('Generating runtime chart...')
fig, ax = plt.subplots(figsize=(8, max(2.5, len(all_results) * 0.7)))
names = []; times = []; colors = []
for key, res in all_results.items():
    if not res.get('after'):
        continue
    names.append(res['display_name'])
    times.append(res['elapsed_sec'])
    colors.append(method_colors.get(key, '#999999'))

bars = ax.barh(names, times, color=colors)
ax.set_xlabel('Seconds')
ax.set_title('Runtime (fit + transform)', fontsize=12, fontweight='bold')
for bar, t in zip(bars, times):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
            f'{t:.1f}s', va='center', fontsize=10)
plt.tight_layout()
out = FIG_DIR / 'bench_runtime.png'
plt.savefig(str(out), dpi=200, bbox_inches='tight')
plt.close()
print(f'  → {out}')

# ════════════════════════════════════════════════════
# Figure 4: Per-marker distribution overlays
# ════════════════════════════════════════════════════
print('Generating distribution overlays...')
n_show = min(6, len(markers))
show_markers = markers[:n_show]

col_names = ['Uncorrected']
col_data  = [X]
for key, res in all_results.items():
    if not res.get('after'):
        continue
    col_names.append(res['display_name'])
    col_data.append(all_corrected[key])

batches = np.unique(y_batch)
fig, axes = plt.subplots(n_show, len(col_names),
                         figsize=(3.5 * len(col_names), 2.2 * n_show),
                         sharex='row')

for col, (cname, Xm) in enumerate(zip(col_names, col_data)):
    for row, mk in enumerate(show_markers):
        ax = axes[row, col]
        j = markers.index(mk)
        for b in batches:
            mask = y_batch == b
            ax.hist(Xm[mask, j], bins=80, density=True, histtype='step', label=b, linewidth=1)
        if row == 0: ax.set_title(cname, fontsize=10, fontweight='bold')
        if col == 0: ax.set_ylabel(mk, fontsize=9)
        if row == 0 and col == len(col_names) - 1: ax.legend(fontsize=7)
        ax.tick_params(labelsize=6)

plt.suptitle('Per-Marker Distributions by Batch', fontsize=12, y=1.01)
plt.tight_layout()
out = FIG_DIR / 'bench_distributions.png'
plt.savefig(str(out), dpi=200, bbox_inches='tight')
plt.close()
print(f'  → {out}')

# ════════════════════════════════════════════════════
# Figures 5-6: UMAP (batch + cell type)
# ════════════════════════════════════════════════════
print('Generating UMAP plots (this takes a minute)...')
import umap
from sklearn.preprocessing import StandardScaler

n_umap = min(15000, len(X))
rng2 = np.random.RandomState(42)
umap_idx = rng2.choice(len(X), n_umap, replace=False)
batch_sub = y_batch[umap_idx]
ct_sub = y_celltype[umap_idx]

umap_datasets = {'Uncorrected': X}
for key, res in all_results.items():
    if res.get('after') and key in all_corrected:
        umap_datasets[res['display_name']] = all_corrected[key]

umap_embeddings = {}
for name, data in umap_datasets.items():
    print(f'  UMAP: {name}...')
    Xs = StandardScaler().fit_transform(data[umap_idx])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
    umap_embeddings[name] = reducer.fit_transform(Xs)

n_plots = len(umap_embeddings)

# Row 1: colored by Batch
fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4))
if n_plots == 1: axes = [axes]

batch_colors_map = {b: plt.cm.Set1(i) for i, b in enumerate(np.unique(batch_sub))}
for ax, (name, emb) in zip(axes, umap_embeddings.items()):
    for b in np.unique(batch_sub):
        mask = batch_sub == b
        ax.scatter(emb[mask, 0], emb[mask, 1], s=0.3, alpha=0.25,
                   label=b, color=batch_colors_map[b])
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=10, fontsize=7)
plt.suptitle('UMAP colored by Batch', fontsize=12, y=1.02)
plt.tight_layout()
out = FIG_DIR / 'bench_umap_batch.png'
plt.savefig(str(out), dpi=200, bbox_inches='tight')
plt.close()
print(f'  → {out}')

# Row 2: colored by Cell Type
fig2, axes2 = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4))
if n_plots == 1: axes2 = [axes2]

ct_colors_map = {ct: plt.cm.tab10(i) for i, ct in enumerate(sorted(np.unique(ct_sub)))}
for ax, (name, emb) in zip(axes2, umap_embeddings.items()):
    for ct in sorted(np.unique(ct_sub)):
        mask = ct_sub == ct
        ax.scatter(emb[mask, 0], emb[mask, 1], s=0.3, alpha=0.25,
                   label=ct, color=ct_colors_map[ct])
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=10, fontsize=7, title='Cell type')
plt.suptitle('UMAP colored by Cell Type', fontsize=12, y=1.02)
plt.tight_layout()
out = FIG_DIR / 'bench_umap_celltype.png'
plt.savefig(str(out), dpi=200, bbox_inches='tight')
plt.close()
print(f'  → {out}')

print('\n✓ All figures regenerated!')
