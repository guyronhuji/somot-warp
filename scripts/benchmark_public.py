#!/usr/bin/env python
"""
Benchmark SOM-OT Warp against competing CyTOF batch correction methods
on public datasets.

Usage:
    python scripts/benchmark_public.py --dataset trussart --datadir ./data/trussart/
    python scripts/benchmark_public.py --dataset trussart --datadir ./data/trussart/ --methods somot,combat
"""

import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── Metrics (self-contained, adapted from som_ot_warp_repo/somot_warp/metrics.py) ───

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.stats import ks_2samp, wasserstein_distance


def _neighbor_indices(X, k=30):
    nn = NearestNeighbors(n_neighbors=min(k, len(X))).fit(X)
    return nn.kneighbors(return_distance=False)


def knn_batch_entropy(X, batch_labels, k=30):
    """Shannon entropy of batch composition in k-NN neighborhoods (higher = better mixing)."""
    X = np.asarray(X); b = np.asarray(batch_labels)
    idx = _neighbor_indices(X, k=k)
    _, inv = np.unique(b, return_inverse=True)
    B = inv.max() + 1
    ent = []
    for row in idx:
        counts = np.bincount(inv[row], minlength=B).astype(float)
        p = counts / (counts.sum() + 1e-12)
        e = -(p[p > 0] * np.log2(p[p > 0])).sum()
        ent.append(e)
    return float(np.mean(ent))


def knn_batch_ilisi(X, batch_labels, k=30):
    """Inverse Simpson index of batch k-NN composition (higher = better mixing)."""
    X = np.asarray(X); b = np.asarray(batch_labels)
    idx = _neighbor_indices(X, k=k)
    _, inv = np.unique(b, return_inverse=True)
    B = inv.max() + 1
    il = []
    for row in idx:
        counts = np.bincount(inv[row], minlength=B).astype(float)
        p = counts / (counts.sum() + 1e-12)
        inv_simpson = 1.0 / np.sum(p**2 + 1e-12)
        il.append(inv_simpson)
    return float(np.mean(il))


def graph_connectivity(X, bio_labels, k=30):
    """Fraction of largest connected component per cell type (higher = better preservation)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    X = np.asarray(X); y = np.asarray(bio_labels)
    idx = _neighbor_indices(X, k=k)
    n = len(X)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    data = np.ones_like(cols, dtype=float)
    G = csr_matrix((data, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    out = []
    for lab in np.unique(y):
        nodes = np.where(y == lab)[0]
        sub = G[nodes[:, None], nodes]
        _, labels = connected_components(sub, directed=False)
        if len(nodes) == 0:
            continue
        largest = np.bincount(labels).max() if labels.size else 0
        out.append(largest / max(1, len(nodes)))
    return float(np.mean(out)) if out else float("nan")


def per_marker_emd(X, batch_labels, markers=None):
    """Median EMD across all batch pairs, per marker."""
    X = np.asarray(X); b = np.asarray(batch_labels)
    batches = np.unique(b)
    pairs = [(a, bb) for i, a in enumerate(batches) for bb in batches[i + 1:]]
    p = X.shape[1]
    emds = np.zeros(p)
    for (a, bb) in pairs:
        for j in range(p):
            emds[j] += wasserstein_distance(X[b == a, j], X[b == bb, j])
    emds /= max(1, len(pairs))
    result = {"median_EMD": float(np.median(emds)), "mean_EMD": float(np.mean(emds))}
    if markers is not None:
        result["per_marker"] = {m: float(emds[j]) for j, m in enumerate(markers)}
    return result


def compute_all_metrics(X, batch_labels, bio_labels=None, markers=None, pca_dims=30):
    """Compute all benchmark metrics on expression matrix X."""
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=min(pca_dims, Z.shape[1]))
    Zp = pca.fit_transform(Z).astype(np.float32)

    out = {
        "knn_entropy": knn_batch_entropy(Zp, batch_labels, k=30),
        "knn_ilisi": knn_batch_ilisi(Zp, batch_labels, k=30),
        "silhouette_batch": float(silhouette_score(
            Zp[::max(1, len(Zp)//20000)],
            np.asarray(batch_labels)[::max(1, len(Zp)//20000)]
        )),
        "per_marker_emd": per_marker_emd(X, batch_labels, markers),
    }
    if bio_labels is not None:
        sub = max(1, len(Zp) // 20000)
        out["silhouette_bio"] = float(silhouette_score(
            Zp[::sub], np.asarray(bio_labels)[::sub]
        ))
        out["graph_connectivity"] = graph_connectivity(Zp, bio_labels, k=30)
    return out


# ─── Data Loaders ───────────────────────────────────────────────────────────────

def read_fcs_to_df(fcs_path):
    """Read a single FCS file and return a pandas DataFrame."""
    try:
        import readfcs
        adata = readfcs.read(fcs_path)
        df = pd.DataFrame(adata.X, columns=list(adata.var_names))
        return df
    except Exception:
        pass
    try:
        import fcsparser
        _, df = fcsparser.parse(fcs_path, reformat_meta=True)
        return df
    except Exception:
        pass
    raise ImportError("Install readfcs or fcsparser to read FCS files: pip install readfcs fcsparser")


def load_trussart(datadir):
    """
    Load the Trussart / CytofRUV dataset (FR-FCM-Z2L2).

    Expected directory structure (pre-gated FCS files):
        datadir/
            RUV1b/           -- Batch 1 (Run 1b)
                A1.fcs ... A5.fcs     (cell type A, samples 1-5)
                B1.fcs ... B5.fcs     (cell type B)
                C1.fcs ... C5.fcs     (cell type C)
                D1.fcs ... D5.fcs     (cell type D)
                Unassigned.fcs
            RUV3b/           -- Batch 2 (Run 3b)
                Run3_A1.fcs ... Run3_A5.fcs
                Run3_B1.fcs ... Run3_B5.fcs
                Run3_C1.fcs ... Run3_C5.fcs
                Run3_D1.fcs ... Run3_D5.fcs
                Run3_Unassigned.fcs

    Cell type labels: letter prefix (A/B/C/D)
    Sample labels: numeric suffix (1-5), shared across both batches
    Anchor: any sample measured in both batches (e.g., sample '1')

    Returns: X, y_batch, y_sample, y_celltype, markers
    """
    import re

    cache = Path(datadir) / "_cache_trussart.parquet"
    if cache.exists():
        print(f"  Loading cached data from {cache}")
        df = pd.read_parquet(cache)
        markers = [c for c in df.columns if c not in ('batch', 'sample', 'celltype')]
        y_ct = df['celltype'].values if 'celltype' in df.columns else None
        return df[markers].values, df['batch'].values, df['sample'].values, y_ct, markers

    base = Path(datadir)

    # Auto-detect batch directories
    batch_dirs = {}
    for candidate in ['RUV1b', 'RUV3b']:
        d = base / candidate
        if d.is_dir():
            batch_dirs[candidate] = d
    # Fallback: look inside Norm_Raw_RUV1b_RUV3b_simult/
    if not batch_dirs:
        norm_dir = base / 'Norm_Raw_RUV1b_RUV3b_simult'
        if norm_dir.is_dir():
            for candidate in ['RUV1b', 'RUV3b']:
                d = norm_dir / candidate
                if d.is_dir():
                    batch_dirs[candidate] = d
    # Fallback: flat FCS files in datadir
    if not batch_dirs:
        flat_fcs = sorted(base.glob("*.fcs"))
        if flat_fcs:
            batch_dirs = {'flat': base}

    if not batch_dirs:
        raise FileNotFoundError(
            f"No FCS files found in {datadir}. Expected structure:\n"
            f"  {datadir}/RUV1b/*.fcs and {datadir}/RUV3b/*.fcs\n"
            "Download from: http://flowrepository.org/id/FR-FCM-Z2L2"
        )

    print(f"  Found batch directories: {list(batch_dirs.keys())}")
    frames = []
    for batch_name, batch_dir in sorted(batch_dirs.items()):
        fcs_files = sorted(batch_dir.glob("*.fcs"))
        print(f"  {batch_name}: {len(fcs_files)} files")
        for f in fcs_files:
            df = read_fcs_to_df(str(f))
            fname = f.stem

            # Strip "Run3_" prefix from RUV3b filenames to get base name
            base_name = re.sub(r'^Run\d+_', '', fname)

            # Parse cell type (letter) and sample number
            m = re.match(r'^([A-Z])(\d+)$', base_name)
            if m:
                celltype = m.group(1)
                sample = m.group(2)
            elif 'unassigned' in base_name.lower():
                celltype = 'Unassigned'
                sample = 'Unassigned'
            else:
                celltype = 'Unknown'
                sample = base_name

            df['batch'] = batch_name
            df['sample'] = sample
            df['celltype'] = celltype
            df['_file'] = fname
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Identify protein marker columns (exclude metadata/scatter/time)
    exclude_patterns = ['time', 'event_length', 'center', 'offset', 'width',
                        'residual', 'beadDist', 'barcode', 'batch', 'sample',
                        'celltype', '_file', 'File', 'DNA', 'Viability',
                        'Live_Dead', 'Cisplatin']
    all_cols = combined.columns.tolist()
    markers = [c for c in all_cols
               if not any(p.lower() in c.lower() for p in exclude_patterns)
               and c not in ('batch', 'sample', 'celltype', '_file')]

    # Filter to numeric marker columns only
    markers = [m for m in markers if pd.api.types.is_numeric_dtype(combined[m])]

    print(f"  Found {len(markers)} markers, {len(combined):,} total cells")
    print(f"  Batches: {combined['batch'].value_counts().to_dict()}")
    print(f"  Samples: {sorted(combined['sample'].unique())}")
    print(f"  Cell types: {combined['celltype'].value_counts().to_dict()}")

    # Cache
    out_df = combined[markers + ['batch', 'sample', 'celltype']].copy()
    out_df.to_parquet(cache)

    X = combined[markers].values.astype(np.float32)
    return X, combined['batch'].values, combined['sample'].values, combined['celltype'].values, markers


def load_dataset(dataset_name, datadir):
    """Dispatch to the appropriate loader."""
    loaders = {
        'trussart': load_trussart,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    return loaders[dataset_name](datadir)


# ─── Method Wrappers ────────────────────────────────────────────────────────────

def run_somot(X, y_batch, markers, anchor_label, **kwargs):
    """Run SOM-OT Warp batch correction."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    batches = np.unique(y_batch)

    if len(batches) == 2:
        from somot_warp.sklearn_api import SOMOTWarpSKExact
        model = SOMOTWarpSKExact(
            anchor='A', output_scale='A',
            transform_kind='arcsinh', pca_components=None,
            som_grid=(8, 8), som_iters=8000,
            n_landmarks_total=1200,
            q_grid=np.array([0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]),
            alpha_ot=0.7, alpha_warp=None,
            som_verbose=True,
        )
        # Map labels so anchor = 'A'
        y_mapped = np.where(y_batch == anchor_label, 'A', 'B')
        model.fit(X, y_mapped)
        X_corr = model.transform(X, y_mapped)
    else:
        from somot_warp.multi_batch import MultiBatchSOMOTWarp
        model = MultiBatchSOMOTWarp(
            anchor_label=anchor_label,
            n_jobs=1,
            transform_kind='arcsinh', pca_components=None,
            som_grid=(8, 8), som_iters=8000,
            n_landmarks_total=1200,
            q_grid=np.array([0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]),
            alpha_ot=0.7, alpha_warp=None,
            som_verbose=True,
        )
        model.fit(X, y_batch)
        X_corr = model.transform(X, y_batch)
    return X_corr


def run_combat(X, y_batch, markers, anchor_label, **kwargs):
    """Run ComBat batch correction via InMoose."""
    try:
        from inmoose.pycombat import pycombat_norm
    except ImportError:
        raise ImportError("Install inmoose: pip install inmoose")

    # pycombat_norm expects features x samples
    batch_list = list(y_batch)
    X_corr = pycombat_norm(X.T, batch_list)
    return np.asarray(X_corr.T, dtype=np.float32)


def run_cytonorm(X, y_batch, markers, anchor_label, y_sample=None, **kwargs):
    """Run CytoNormPy batch correction.

    CytoNormPy API requires:
      1. Create AnnData with data in a layer
      2. Set obs columns: batch, reference (ref/other), file_name (sample ID)
      3. cn.add_clusterer(...)
      4. cn.run_anndata_setup(...)
      5. cn.run_clustering(...)
      6. cn.calculate_quantiles()
      7. cn.calculate_splines()
      8. cn.normalize_data()
      9. Retrieve from adata.layers['cyto_normalized']
    """
    try:
        import cytonormpy as cnp
        import anndata as ad
    except ImportError:
        raise ImportError(
            "Install CytoNormPy: pip install cytonormpy  "
            "(or: pip install git+https://github.com/TarikExner/CytoNormPy@main)"
        )

    # ── Build AnnData ──
    adata = ad.AnnData(X.copy().astype(np.float64))
    adata.var_names = list(markers)

    # Store original data in a layer (CytoNormPy reads from layers, not .X)
    adata.layers['compensated'] = adata.X.copy()

    adata.obs['batch'] = pd.Categorical(y_batch)

    # Build a unique sample identifier per batch+sample combination
    if y_sample is not None:
        # Create file-like IDs: "batch_sample" to be unique per batch
        file_names = [f"{b}_{s}" for b, s in zip(y_batch, y_sample)]
        adata.obs['file_name'] = pd.Categorical(file_names)
    else:
        # No sample info: treat each batch as one "file"
        adata.obs['file_name'] = pd.Categorical(y_batch)

    # Mark reference samples: those present in multiple batches
    if y_sample is not None:
        sample_batches = pd.DataFrame({'sample': y_sample, 'batch': y_batch})
        ref_samples = (sample_batches.groupby('sample')['batch'].nunique()
                       .loc[lambda x: x > 1].index.tolist())
        if ref_samples:
            # Reference = cells whose sample ID is in the shared set
            adata.obs['reference'] = pd.Categorical(
                ['ref' if s in ref_samples else 'other'
                 for s in y_sample]
            )
        else:
            # Fallback: use anchor batch as reference
            adata.obs['reference'] = pd.Categorical(
                ['ref' if b == anchor_label else 'other' for b in y_batch]
            )
    else:
        adata.obs['reference'] = pd.Categorical(
            ['ref' if b == anchor_label else 'other' for b in y_batch]
        )

    # ── Setup CytoNorm ──
    cn = cnp.CytoNorm()

    # Add clusterer (FlowSOM preferred, fall back to KMeans)
    try:
        clusterer = cnp.FlowSOM(n_clusters=20)
    except Exception:
        clusterer = cnp.KMeans(n_clusters=20)
    cn.add_clusterer(clusterer)

    # Setup data handler from AnnData
    cn.run_anndata_setup(
        adata,
        layer='compensated',
        reference_column='reference',
        reference_value='ref',
        batch_column='batch',
        sample_identifier_column='file_name',
        channels=list(markers),
    )

    # ── Run normalization pipeline ──
    cn.run_clustering(markers=list(markers))
    cn.calculate_quantiles()
    cn.calculate_splines()
    cn.normalize_data()

    # Retrieve normalized data
    return adata.layers['cyto_normalized'].astype(np.float32)


def run_cycombine(X, y_batch, markers, anchor_label, **kwargs):
    """Run cyCombine batch correction via rpy2."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
    except ImportError:
        raise ImportError("Install rpy2: pip install rpy2 (requires R)")

    try:
        cycombine = importr('cyCombine')
    except Exception:
        raise ImportError(
            "Install cyCombine in R: devtools::install_github('biosurf/cyCombine')"
        )

    df = pd.DataFrame(X, columns=markers)
    df['batch'] = y_batch

    r_df = pandas2ri.py2rpy(df)
    markers_r = ro.StrVector(list(markers))

    corrected_r = cycombine.batch_correct(
        df=r_df,
        markers=markers_r,
        norm_method="scale",
        seed=42
    )
    corrected = pandas2ri.rpy2py(corrected_r)
    return corrected[markers].values.astype(np.float32)


def run_mmdresnet(X, y_batch, markers, anchor_label, **kwargs):
    """Run MMD-ResNet batch correction (requires PyTorch repo clone)."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        raise ImportError("Install PyTorch: pip install torch")

    batches = np.unique(y_batch)
    anchor_mask = y_batch == anchor_label
    X_anchor = X[anchor_mask].astype(np.float32)
    X_corr = X.copy().astype(np.float32)

    # Simple MMD-ResNet implementation for benchmark
    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)

        def forward(self, x):
            r = torch.relu(self.bn1(self.fc1(x)))
            r = self.bn2(self.fc2(r))
            return x + r

    class MMDResNet(nn.Module):
        def __init__(self, dim, n_blocks=3):
            super().__init__()
            self.blocks = nn.Sequential(*[ResBlock(dim) for _ in range(n_blocks)])

        def forward(self, x):
            return self.blocks(x)

    def mmd_rbf(x, y, sigma=1.0):
        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)
        kxx = torch.exp(-xx / (2 * sigma**2)).mean()
        kyy = torch.exp(-yy / (2 * sigma**2)).mean()
        kxy = torch.exp(-xy / (2 * sigma**2)).mean()
        return kxx + kyy - 2 * kxy

    target_t = torch.tensor(X_anchor, dtype=torch.float32)
    dim = X.shape[1]

    for b in batches:
        if b == anchor_label:
            continue
        mask = y_batch == b
        source = X[mask].astype(np.float32)
        source_t = torch.tensor(source, dtype=torch.float32)

        model = MMDResNet(dim, n_blocks=3)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

        model.train()
        for epoch in range(200):
            # Mini-batch
            idx_s = np.random.choice(len(source), min(1000, len(source)), replace=False)
            idx_t = np.random.choice(len(X_anchor), min(1000, len(X_anchor)), replace=False)
            xs = source_t[idx_s]
            xt = target_t[idx_t]

            out = model(xs)
            loss = mmd_rbf(out, xt, sigma=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_corr[mask] = model(source_t).numpy()

    return X_corr


# ─── Method Registry ────────────────────────────────────────────────────────────

METHOD_REGISTRY = {
    'somot':     ('SOM-OT Warp',  run_somot),
    'combat':    ('ComBat',       run_combat),
    'cytonorm':  ('CytoNormPy',   run_cytonorm),
    'cycombine': ('cyCombine',    run_cycombine),
    'mmdresnet': ('MMD-ResNet',   run_mmdresnet),
}


# ─── Visualization ──────────────────────────────────────────────────────────────

def plot_benchmark_results(results, markers, outdir):
    """Generate comparison figures."""
    import matplotlib.pyplot as plt

    figdir = Path(outdir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    methods_with_before = ['Uncorrected'] + methods

    # ── 1. Bar chart: key metrics side by side ──
    metric_keys = ['knn_entropy', 'knn_ilisi']
    has_bio = any('silhouette_bio' in results[m].get('after', {}) for m in methods)
    if has_bio:
        metric_keys.append('silhouette_bio')
    metric_keys.append('silhouette_batch')

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 5))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, mk in zip(axes, metric_keys):
        vals = []
        labels = []
        for m in methods:
            before_val = results[m].get('before', {}).get(mk, np.nan)
            after_val = results[m].get('after', {}).get(mk, np.nan)
            if m == methods[0]:
                vals.append(before_val)
                labels.append('Uncorrected')
            vals.append(after_val)
            labels.append(results[m]['display_name'])

        colors = ['gray'] + [plt.cm.Set2(i) for i in range(len(methods))]
        ax.barh(range(len(vals)), vals, color=colors[:len(vals)])
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(mk)
        ax.set_title(mk.replace('_', ' ').title())

    plt.tight_layout()
    plt.savefig(figdir / "metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── 2. Per-marker EMD comparison ──
    fig, ax = plt.subplots(figsize=(10, max(6, len(markers) * 0.3)))
    x = np.arange(len(markers))
    width = 0.8 / (len(methods_with_before))

    for i, m in enumerate(methods):
        before_emd = results[m].get('before', {}).get('per_marker_emd', {}).get('per_marker', {})
        after_emd = results[m].get('after', {}).get('per_marker_emd', {}).get('per_marker', {})

        if i == 0 and before_emd:
            vals_before = [before_emd.get(mk, 0) for mk in markers]
            ax.barh(x - width * (len(methods)) / 2, vals_before, width,
                    label='Uncorrected', color='gray', alpha=0.7)

        if after_emd:
            vals_after = [after_emd.get(mk, 0) for mk in markers]
            ax.barh(x + width * (i + 0.5 - len(methods) / 2), vals_after, width,
                    label=results[m]['display_name'], alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(markers, fontsize=8)
    ax.set_xlabel('Earth Mover Distance')
    ax.set_title('Per-Marker EMD by Method')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "per_marker_emd.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── 3. Timing comparison ──
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [results[m]['display_name'] for m in methods]
    times = [results[m].get('elapsed_sec', 0) for m in methods]
    colors = [plt.cm.Set2(i) for i in range(len(methods))]
    ax.barh(names, times, color=colors)
    ax.set_xlabel('Seconds')
    ax.set_title('Runtime Comparison')
    plt.tight_layout()
    plt.savefig(figdir / "timing.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figures saved to {figdir}/")


def print_summary_table(results, outdir):
    """Print and save a summary comparison table."""
    rows = []
    for method_key, res in results.items():
        row = {'Method': res['display_name'], 'Time (s)': f"{res.get('elapsed_sec', 0):.1f}"}
        after = res.get('after', {})
        before = res.get('before', {})
        row['kNN Entropy (before)'] = f"{before.get('knn_entropy', 0):.3f}"
        row['kNN Entropy (after)'] = f"{after.get('knn_entropy', 0):.3f}"
        row['iLISI (before)'] = f"{before.get('knn_ilisi', 0):.3f}"
        row['iLISI (after)'] = f"{after.get('knn_ilisi', 0):.3f}"
        row['Sil. Batch (before)'] = f"{before.get('silhouette_batch', 0):.3f}"
        row['Sil. Batch (after)'] = f"{after.get('silhouette_batch', 0):.3f}"
        if 'silhouette_bio' in after:
            row['Sil. Bio (before)'] = f"{before.get('silhouette_bio', 0):.3f}"
            row['Sil. Bio (after)'] = f"{after.get('silhouette_bio', 0):.3f}"
        if 'graph_connectivity' in after:
            row['Graph Conn. (before)'] = f"{before.get('graph_connectivity', 0):.3f}"
            row['Graph Conn. (after)'] = f"{after.get('graph_connectivity', 0):.3f}"
        row['Median EMD (before)'] = f"{before.get('per_marker_emd', {}).get('median_EMD', 0):.3f}"
        row['Median EMD (after)'] = f"{after.get('per_marker_emd', {}).get('median_EMD', 0):.3f}"
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = Path(outdir) / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nSaved to {summary_path}")
    return summary


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SOM-OT Warp against competing CyTOF batch correction methods."
    )
    parser.add_argument("--dataset", required=True, choices=["trussart"],
                        help="Public dataset to benchmark on")
    parser.add_argument("--datadir", required=True,
                        help="Directory containing downloaded FCS files")
    parser.add_argument("--outdir", default="results",
                        help="Output directory for results")
    parser.add_argument("--methods", default="somot,combat,mmdresnet",
                        help="Comma-separated list of methods to run "
                             "(somot,combat,cytonorm,cycombine,mmdresnet)")
    parser.add_argument("--anchor", default=None,
                        help="Anchor batch label (default: auto-detect)")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Subsample cells per batch for faster testing (0 = no subsample)")
    args = parser.parse_args()

    outdir = Path(args.outdir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"\n{'='*60}")
    print(f"Loading dataset: {args.dataset}")
    print(f"{'='*60}")
    X, y_batch, y_sample, y_celltype, markers = load_dataset(args.dataset, args.datadir)
    print(f"  Shape: {X.shape}, Markers: {len(markers)}")
    print(f"  Batches: {np.unique(y_batch)}")

    # Auto-detect anchor
    anchor = args.anchor
    if anchor is None:
        # Use the first batch as anchor
        anchor = np.unique(y_batch)[0]
    print(f"  Anchor batch: {anchor}")

    # Subsample if requested
    if args.subsample > 0:
        rng = np.random.RandomState(42)
        keep = []
        for b in np.unique(y_batch):
            idx = np.where(y_batch == b)[0]
            n = min(args.subsample, len(idx))
            keep.append(rng.choice(idx, n, replace=False))
        keep = np.concatenate(keep)
        X = X[keep]
        y_batch = y_batch[keep]
        y_sample = y_sample[keep] if y_sample is not None else None
        y_celltype = y_celltype[keep] if y_celltype is not None else None
        print(f"  Subsampled to {len(X)} cells ({args.subsample} per batch)")

    # ── Compute uncorrected metrics ──
    print(f"\nComputing uncorrected (baseline) metrics...")
    metrics_before = compute_all_metrics(X, y_batch, y_celltype, markers)

    # ── Run each method ──
    method_list = [m.strip() for m in args.methods.split(",")]
    results = {}

    for method_key in method_list:
        if method_key not in METHOD_REGISTRY:
            print(f"\n  WARNING: Unknown method '{method_key}', skipping.")
            continue

        display_name, run_fn = METHOD_REGISTRY[method_key]
        print(f"\n{'─'*60}")
        print(f"Running: {display_name}")
        print(f"{'─'*60}")

        try:
            t0 = time.perf_counter()
            X_corr = run_fn(
                X.copy(), y_batch.copy(), markers, anchor,
                y_sample=y_sample
            )
            elapsed = time.perf_counter() - t0
            print(f"  Completed in {elapsed:.1f}s")

            # Compute post-correction metrics
            print(f"  Computing metrics...")
            metrics_after = compute_all_metrics(X_corr, y_batch, y_celltype, markers)

            results[method_key] = {
                'display_name': display_name,
                'elapsed_sec': elapsed,
                'before': metrics_before,
                'after': metrics_after,
            }

            # Save corrected data and per-method metrics
            method_dir = outdir / method_key
            method_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(X_corr, columns=markers).to_parquet(method_dir / "corrected.parquet")
            with open(method_dir / "metrics.json", "w") as f:
                json.dump({
                    'elapsed_sec': elapsed,
                    'before': metrics_before,
                    'after': metrics_after
                }, f, indent=2, default=str)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[method_key] = {
                'display_name': display_name,
                'elapsed_sec': 0,
                'before': metrics_before,
                'after': {},
                'error': str(e),
            }

    # ── Summary and plots ──
    if results:
        print_summary_table(results, outdir)
        try:
            plot_benchmark_results(results, markers, outdir)
        except Exception as e:
            print(f"  Plotting failed: {e}")

    print(f"\nAll results saved to {outdir}/")


if __name__ == "__main__":
    main()
