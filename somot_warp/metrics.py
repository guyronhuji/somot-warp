from minisom import MiniSom
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plot
def assign_som_nodes(df, features, som_x=8, som_y=8, iters=5000, random_state=0):
    X = df[features].values.astype(np.float32)

    som = MiniSom(
        x=som_x, y=som_y, input_len=len(features),
        sigma=1.0, learning_rate=0.5, random_seed=random_state
    )
    som.random_weights_init(X)
    som.train_random(X, num_iteration=iters, verbose=True)

    # assign BMU node index to each cell
    labels = []
    for x in X:
        i, j = som.winner(x)
        labels.append(i * som_y + j)
    return np.array(labels)


import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, median_abs_deviation

def emd_mad_scores(df_uncorr, df_corr, batch_col, node_col, features, bin_size=0.1):
    """
    Compute cyCombine-like EMD reduction and MAD score.

    Parameters
    ----------
    df_uncorr : pd.DataFrame
        Uncorrected dataframe (must include batch_col, node_col, features).
    df_corr : pd.DataFrame
        Corrected dataframe (same shape, same columns).
    batch_col : str
        Name of column with batch labels.
    node_col : str
        Name of column with SOM node labels (must match in both dfs).
    features : list of str
        Marker/features to evaluate.
    bin_size : float
        Bin size for histogram-based EMD (default 0.1).
    
    Returns
    -------
    emd_reduction : float
        Global EMD reduction score.
    mad_score : float
        Global MAD score.
    emd_table : pd.DataFrame
        Detailed per-feature/node/batchpair EMDs before and after.
    mad_table : pd.DataFrame
        Detailed per-feature/node/batch MAD before and after.
    """
    batches = df_uncorr[batch_col].unique()
    pairs = [(a, b) for i, a in enumerate(batches) for b in batches[i+1:]]
    emd_records = []
    mad_records = []

    # --- EMD ---
    for f in tqdm(features):
        for node in np.unique(df_uncorr[node_col]):
            dfn_u = df_uncorr[df_uncorr[node_col] == node]
            dfn_c = df_corr[df_corr[node_col] == node]
            for (a, b) in pairs:
                x_u = dfn_u.loc[dfn_u[batch_col] == a, f].values
                y_u = dfn_u.loc[dfn_u[batch_col] == b, f].values
                x_c = dfn_c.loc[dfn_c[batch_col] == a, f].values
                y_c = dfn_c.loc[dfn_c[batch_col] == b, f].values

                if len(x_u) == 0 or len(y_u) == 0: 
                    continue
                if len(x_c) == 0 or len(y_c) == 0:
                    continue

                # binning
                lo = min(x_u.min(), y_u.min(), x_c.min(), y_c.min())
                hi = max(x_u.max(), y_u.max(), x_c.max(), y_c.max())
                bins = np.arange(lo, hi + bin_size, bin_size)

                def hist_emd(x, y):
                    hx, _ = np.histogram(x, bins=bins, density=True)
                    hy, _ = np.histogram(y, bins=bins, density=True)
                    cdfx = np.cumsum(hx) / np.sum(hx)
                    cdfy = np.cumsum(hy) / np.sum(hy)
                    return np.sum(np.abs(cdfx - cdfy))

                emd_u = hist_emd(x_u, y_u)
                emd_c = hist_emd(x_c, y_c)

                if emd_u < 2 and emd_c < 2:
                    continue

                emd_records.append(dict(
                    feature=f, node=node, pair=f"{a}-{b}",
                    EMD_uncorr=emd_u, EMD_corr=emd_c
                ))

    emd_table = pd.DataFrame(emd_records)
    emd_reduction = ((emd_table["EMD_uncorr"] - emd_table["EMD_corr"]).sum() /
                     emd_table["EMD_uncorr"].sum())

    # --- MAD ---
    for f in tqdm(features):
        for node in np.unique(df_uncorr[node_col]):
            dfn_u = df_uncorr[df_uncorr[node_col] == node]
            dfn_c = df_corr[df_corr[node_col] == node]
            for b in batches:
                xu = dfn_u.loc[dfn_u[batch_col] == b, f].values
                xc = dfn_c.loc[dfn_c[batch_col] == b, f].values
                if len(xu) == 0 or len(xc) == 0:
                    continue
                mad_u = median_abs_deviation(xu, scale=1.0, nan_policy='omit')
                mad_c = median_abs_deviation(xc, scale=1.0, nan_policy='omit')
                mad_records.append(dict(
                    feature=f, node=node, batch=b,
                    MAD_uncorr=mad_u, MAD_corr=mad_c,
                    MAD_diff=np.abs(mad_u - mad_c)
                ))

    mad_table = pd.DataFrame(mad_records)
    mad_score = mad_table["MAD_diff"].median()

    return emd_reduction, mad_score, emd_table, mad_table


def plot_emd_reduction(emd_table):
    import matplotlib.pyplot as plt
    """
    Barplot of mean EMD reduction per marker, similar to cyCombine paper.
    """
    # reduction per row
    emd_table["reduction"] = (
        (emd_table["EMD_uncorr"] - emd_table["EMD_corr"]) / emd_table["EMD_uncorr"]
    )

    # aggregate per marker
    per_marker = emd_table.groupby("feature")["reduction"].mean().sort_values()

    plt.figure(figsize=(8, 5))
    per_marker.plot(kind="barh", color="steelblue")
    plt.xlabel("Mean EMD reduction")
    plt.ylabel("Marker")
    plt.title("Per-marker EMD reduction after batch correction")
    plt.axvline(0, color="k", linewidth=1)
    plt.tight_layout()
    plt.show()
