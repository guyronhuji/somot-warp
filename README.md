# SOMOTwarp

Batch correction for cytometry (CyTOF / flow) data: **SOM-clustered landmark optimal transport + per-marker quantile
warping**, with a scikit-learn API and a learned mapping that can be applied to new cells.

Typical use: learn the batch effect from a reference sample measured in every batch (e.g. untreated cells), then
apply the learned mapping to all cells of that batch, including conditions that were never used for fitting.

## How it works

For two batches A and B:

1. **Stabilize** each marker (`arcsinh(x / cofactor)` or `log1p`, or none if the data are already transformed).
2. **Z-score per batch**, then optionally **PCA** on the pooled cells.
3. Train a **self-organizing map** (MiniSom) on the pooled cells; each SOM node is a local region of phenotype space.
4. In every node with enough cells from both batches, summarize each batch by **k-means landmarks** and solve an
   **entropic (unbalanced) optimal-transport** problem between them. Cells are moved toward the transported landmark
   images (soft k-NN interpolation), blended with their own position by `alpha_ot`.
5. Map back to marker space and apply a **per-marker monotone quantile warp** inside each node.
6. Invert the scaling / transform, giving corrected values in the original marker space.

`anchor='A'` keeps batch A fixed and maps B onto it. `MultiBatchSOMOTWarp` does this for any number of batches
(star topology: every batch is mapped onto one anchor batch).

## Install

```bash
pip install git+https://github.com/guyronhuji/somot-warp
# or, for development
git clone https://github.com/guyronhuji/somot-warp && cd somot-warp && pip install -e .
```

Requires Python ≥ 3.9 with numpy, scipy, pandas, scikit-learn, MiniSom, POT, tqdm, joblib and matplotlib.

## Quick start

### Two batches

```python
import numpy as np
from somot_warp import SOMOTWarpSKExact

# X: (n_cells, n_markers) raw intensities; y: batch label per cell, exactly two values
model = SOMOTWarpSKExact(anchor="A", output_scale="A", alpha_ot=0.7, som_grid=(6, 6), pca_components=None)
model.fit(X, y)                      # labels are sorted: the first one is batch "A" (the anchor)

X_new_corr = model.transform(X_new, y=np.array(["B"] * len(X_new)))   # apply the learned map to new B cells
```

### Several batches onto one anchor

```python
from somot_warp import MultiBatchSOMOTWarp

model = MultiBatchSOMOTWarp(anchor_label="B1", n_jobs=-1, alpha_ot=0.7, som_grid=(6, 6), pca_components=None)
model.fit(X_train, batch_train)              # e.g. untreated cells of every batch
X_corr = model.transform(X_all, batch_all)   # every cell of every batch; anchor-batch cells pass through unchanged
```

## Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `transform_kind`, `cofactor` | `'arcsinh'`, `5.0` | Use `transform_kind=None` if the input is already arcsinh-transformed / normalized. |
| `pca_components` | `30` | `None` works on all markers directly. |
| `som_grid` | `(10, 10)` | Pick it so a typical node holds ≥ ~100 cells of the smaller batch; nodes below `min_node_size` (default 80) in either batch get no OT/warp map (only the per-batch z-score). |
| `n_landmarks_total` | `1200` | Landmarks per node ≈ this / number of nodes (min 20 per batch). |
| `reg` | `0.1` | Sinkhorn entropy, **relative to the median landmark cost of each node** (`cost_scale='median'`). |
| `cost_scale` | `'median'` | `'median'`, `'mean'`, or `None` for the legacy absolute `reg` (see changelog). |
| `ot_method` | `'sinkhorn_stabilized'` | POT solver; log-stabilized by default. |
| `unbalanced`, `reg_m` | `True`, `5.0` | Unbalanced OT and its marginal relaxation. |
| `alpha_ot` | `None` | How far cells move toward their transported images (with `anchor='A'`, `None` = 1.0 for B). **0.7 is recommended**: full transport (1.0) pulls cells onto landmark images and shrinks within-node variation. |
| `alpha_warp` | `None` | Blend for the per-marker quantile warp (`None` = 1.0 for the moving batch). |
| `q_grid` | 1…99 percentiles | Include 0 and 100 to warp the full range instead of clamping the tails. |
| `anchor`, `output_scale` | `'barycentric'`, `'batch'` | `'A'` / `'A'` to map B onto A and report everything on A's scale. |

After fitting, `model.nodes_` holds the per-node OT maps and `model.n_unmatched_landmarks_` counts landmarks that
received no transport mass (they keep their own position).

## Command line

```bash
# fit on two batches, write corrected tables and a report
somot-batch --batchA A.csv --batchB B.csv --anchor A --output_scale A --alpha_ot 0.7 --som 6x6 \
            --save_model model.pkl

# apply a saved model to new cells of batch B
somot-apply --model model.pkl --batch B_new.csv --which B --out B_new_corrected.csv
```

`somot-batch --help` lists all options (PCA, SOM size/iterations, landmarks, `--reg`, `--cost_scale`, …).

## Benchmarks

`scripts/` contains the benchmark on the Trussart et al. CytofRUV data (FlowRepository FR-FCM-Z2L2) against
ComBat and MMD-ResNet (`benchmark_public.py`, `benchmark_notebook.ipynb`, `benchmark_generalization.ipynb`,
`regen_bench_figs.py`). Place the pre-gated FCS files in `data/trussart/RUV1b/` and `data/trussart/RUV3b/`
(layout documented in `load_trussart`), or point `SOMOT_TRUSSART_DIR` at them. Figures go to `results/figures/`
unless `SOMOT_FIG_DIR` is set. Run the notebooks from `scripts/` (or set `SOMOT_REPO`).

## Changelog

**0.3.0**
* Fix: the OT cost matrix is now scaled per SOM node (`cost_scale='median'`) and solved with the log-stabilized
  Sinkhorn. Previously `reg` was absolute; with unscaled squared distances in many dimensions (e.g. ~36 markers
  without PCA) `exp(-M / reg)` underflowed, transport plans came back empty, and all cells of the affected nodes were
  mapped to the same point (duplicate cells, isolated islands in UMAPs). `reg` is now relative: the default 0.1
  corresponds to `reg ≈ 2` absolute on typical z-scored CyTOF data. Use `cost_scale=None` for the old behaviour.
* Landmarks that receive no transport mass keep their own position instead of collapsing to the origin.
* `MultiBatchSOMOTWarp` exported at package level; CLI gains `--cost_scale` and `--ot_method`.

## License

MIT — see [LICENSE](LICENSE). If you use SOMOTwarp, please cite it (see [CITATION.cff](CITATION.cff)).
