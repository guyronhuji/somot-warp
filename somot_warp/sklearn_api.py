
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from minisom import MiniSom
import ot
from tqdm import tqdm

def _cdist2(A, B):
    A2 = np.sum(A*A, axis=1, keepdims=True)
    B2 = np.sum(B*B, axis=1, keepdims=True).T
    return A2 + B2 - 2*A@B.T

def _cost_scale(M, how):
    """Divisor that puts an OT cost matrix on a unit scale, making `reg` relative (None -> 1: absolute `reg`)."""
    if how is None:
        return 1.0
    pos = M[M > 0]
    if pos.size == 0:
        return 1.0
    if how == 'median':
        return float(np.median(pos))
    if how == 'mean':
        return float(np.mean(pos))
    raise ValueError("cost_scale must be 'median', 'mean' or None")

def _quantile_warp_fit(x_ref, x_src, q_grid):
    xr = np.percentile(x_ref, q_grid)
    xs = np.percentile(x_src, q_grid)
    eps = 1e-9
    xs = np.maximum.accumulate(xs + np.linspace(0, eps*(len(q_grid)-1), len(q_grid)))
    return xs, xr

def _quantile_warp_apply(x, x_knots, y_knots):
    return np.interp(x, x_knots, y_knots, left=y_knots[0], right=y_knots[-1])

class SOMOTWarpSKExact(BaseEstimator, TransformerMixin):
    def __init__(self,
        transform_kind='arcsinh', cofactor=5.0,
        pca_components=30,
        som_grid=(10,10), som_sigma=1.5, som_learning_rate=0.5,
        som_iters=8000, som_random_state=0, som_verbose=False,
        n_landmarks_total=1200, reg=0.1, unbalanced=True, reg_m=5.0,
        cost_scale='median', ot_method='sinkhorn_stabilized',
        interp='soft', soft_k=5, soft_tau=0.5, midpoint_ot=True,
        q_grid=None, midpoint_warp=True,
        min_node_size=80,
        anchor='barycentric', alpha_ot=None, alpha_warp=None,
        output_scale='batch'
    ):
        self.transform_kind = transform_kind
        self.cofactor = cofactor
        self.pca_components = pca_components
        self.som_grid = som_grid
        self.som_sigma = som_sigma
        self.som_learning_rate = som_learning_rate
        self.som_iters = som_iters
        self.som_random_state = som_random_state
        self.som_verbose = som_verbose
        self.n_landmarks_total = n_landmarks_total
        self.reg = reg
        self.unbalanced = unbalanced
        self.reg_m = reg_m
        self.cost_scale = cost_scale
        self.ot_method = ot_method
        self.interp = interp
        self.soft_k = soft_k
        self.soft_tau = soft_tau
        self.midpoint_ot = midpoint_ot
        self.q_grid = (np.array([1,5,10,20,30,40,50,60,70,80,90,95,99], float)
                       if q_grid is None else np.asarray(q_grid, float))
        self.midpoint_warp = midpoint_warp
        self.min_node_size = min_node_size
        self.anchor = anchor
        self.alpha_ot = alpha_ot
        self.alpha_warp = alpha_warp
        self.output_scale = output_scale

        # learned
        self.sa_ = None; self.sb_ = None
        self.pca_ = None
        self.som_ = None
        self.mx_ = None; self.my_ = None
        self.label_a_ = None; self.label_b_ = None
        self.nodes_ = {}
        self.y_ = None
        self.n_features_in_ = None
        self.Xa_corr_ = None; self.Xb_corr_ = None

    # --- helpers ---
    def _stabilize(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.transform_kind == 'arcsinh':
            return np.arcsinh(X / self.cofactor).astype(np.float32)
        if self.transform_kind == 'log1p':
            return np.log1p(X).astype(np.float32)
        return X

    def _destabilize(self, Xt):
        if self.transform_kind == 'arcsinh':
            return (np.sinh(Xt) * self.cofactor).astype(np.float32)
        if self.transform_kind == 'log1p':
            return np.expm1(Xt).astype(np.float32)
        return Xt.astype(np.float32)

    def _two_batches(self, y):
        y = np.asarray(y)
        uniq = np.unique(y)
        if uniq.size != 2:
            raise ValueError(f"Expected exactly 2 batches; got {uniq.size}: {uniq}")
        self.label_a_, self.label_b_ = uniq[0], uniq[1]
        return (y == self.label_a_), (y == self.label_b_)

    def _alphas_for_batches(self):
        if self.anchor == 'A':
            return 0.0, (1.0 if self.alpha_ot is None else float(self.alpha_ot))
        if self.anchor == 'B':
            return (1.0 if self.alpha_ot is None else float(self.alpha_ot)), 0.0
        a = 0.5 if self.alpha_ot is None else float(self.alpha_ot)
        return a, a

    def _alphas_warp_for_batches(self):
        if self.anchor == 'A':
            return 0.0, (1.0 if self.alpha_warp is None else float(self.alpha_warp))
        if self.anchor == 'B':
            return (1.0 if self.alpha_warp is None else float(self.alpha_warp)), 0.0
        a = 0.5 if self.alpha_warp is None else float(self.alpha_warp)
        return a, a

    # --- fit ---
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        self.n_features_in_ = X.shape[1]; self.y_ = y.copy()

        maskA, maskB = self._two_batches(y)
        Xa0, Xb0 = X[maskA], X[maskB]

        Xa_t = self._stabilize(Xa0); Xb_t = self._stabilize(Xb0)

        self.sa_, self.sb_ = StandardScaler(), StandardScaler()
        Za_s = self.sa_.fit_transform(Xa_t).astype(np.float32)
        Zb_s = self.sb_.fit_transform(Xb_t).astype(np.float32)

        if self.pca_components is not None:
            self.pca_ = PCA(n_components=self.pca_components, random_state=self.som_random_state)
            Zp_pool = self.pca_.fit_transform(np.vstack([Za_s, Zb_s])).astype(np.float32)
            Za_p, Zb_p = Zp_pool[:len(Za_s)], Zp_pool[len(Za_s):]
        else:
            self.pca_ = None; Za_p, Zb_p = Za_s, Zb_s

        self.mx_, self.my_ = self.som_grid
        self.som_ = MiniSom(x=self.mx_, y=self.my_, input_len=Za_p.shape[1], sigma=self.som_sigma,
                            learning_rate=self.som_learning_rate, neighborhood_function='gaussian',
                            random_seed=self.som_random_state)
        self.som_.random_weights_init(np.vstack([Za_p, Zb_p]))
        self.som_.train_random(data=np.vstack([Za_p, Zb_p]), num_iteration=self.som_iters, verbose=self.som_verbose)

        def bmu_labels(Z):
            labs = []
            for z in Z:
                i, j = self.som_.winner(z)
                labs.append(i * self.my_ + j)
            return np.array(labs, int)

        lab_a = bmu_labels(Za_p); lab_b = bmu_labels(Zb_p)

        Za_corr = Za_p.copy(); Zb_corr = Zb_p.copy()
        n_nodes = self.mx_ * self.my_
        self.nodes_.clear()
        self.n_unmatched_landmarks_ = 0
        alphaA, alphaB = self._alphas_for_batches()

        for node in tqdm(range(n_nodes), disable=not self.som_verbose):
            Ia = np.where(lab_a == node)[0]; Ib = np.where(lab_b == node)[0]
            if len(Ia) < self.min_node_size or len(Ib) < self.min_node_size:
                continue
            Za_k, Zb_k = Za_p[Ia], Zb_p[Ib]

            total_here = len(Ia) + len(Ib)
            base = self.n_landmarks_total / max(1, n_nodes)
            ka = min(max(20, int(base * (len(Ia)/total_here))), len(Ia))
            kb = min(max(20, int(base * (len(Ib)/total_here))), len(Ib))

            kma = KMeans(n_clusters=ka, n_init=5, random_state=self.som_random_state).fit(Za_k)
            kmb = KMeans(n_clusters=kb, n_init=5, random_state=self.som_random_state).fit(Zb_k)
            Ca, Cb = kma.cluster_centers_.astype(np.float32), kmb.cluster_centers_.astype(np.float32)
            a_mass = np.bincount(kma.labels_, minlength=ka).astype(float); a_mass /= a_mass.sum()
            b_mass = np.bincount(kmb.labels_, minlength=kb).astype(float); b_mass /= b_mass.sum()

            # Costs are scaled per node (cost_scale) so `reg` is relative to the typical landmark distance; with an
            # absolute reg the Sinkhorn kernel exp(-M/reg) underflows in high dimensions and the plan comes back empty.
            M = np.maximum(_cdist2(Ca, Cb).astype(np.float64), 0.0)
            M_ot = M / _cost_scale(M, self.cost_scale)
            if self.unbalanced:
                T = ot.unbalanced.sinkhorn_unbalanced(a_mass, b_mass, M_ot, reg=self.reg, reg_m=self.reg_m,
                                                      method=self.ot_method)
            else:
                T = ot.sinkhorn(a_mass, b_mass, M_ot, reg=self.reg, method=self.ot_method)
            T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

            # A landmark that receives no transport mass keeps its own position (instead of collapsing to 0)
            rows = T.sum(axis=1, keepdims=True)
            cols = T.sum(axis=0, keepdims=True)
            tiny = 1e-12 * max(float(T.sum()), 1e-300)
            ok_a = rows[:, 0] > tiny
            ok_b = cols[0] > tiny
            Ca_to_B = np.where(ok_a[:, None], (T @ Cb) / np.maximum(rows, 1e-300), Ca)
            Cb_to_A = np.where(ok_b[:, None], (T.T @ Ca) / np.maximum(cols.T, 1e-300), Cb)
            self.n_unmatched_landmarks_ += int((~ok_a).sum() + (~ok_b).sum())

            nna = NearestNeighbors(n_neighbors=min(self.soft_k, ka)).fit(Ca)
            nnb = NearestNeighbors(n_neighbors=min(self.soft_k, kb)).fit(Cb)

            if self.interp == 'soft':
                da, ia = nna.kneighbors(Za_k, return_distance=True)
                wa = np.exp(-da / (self.soft_tau + 1e-12)); wa /= wa.sum(axis=1, keepdims=True) + 1e-12
                Za_img = (wa[:, :, None] * Ca_to_B[ia]).sum(axis=1)

                db, ib = nnb.kneighbors(Zb_k, return_distance=True)
                wb = np.exp(-db / (self.soft_tau + 1e-12)); wb /= wb.sum(axis=1, keepdims=True) + 1e-12
                Zb_img = (wb[:, :, None] * Cb_to_A[ib]).sum(axis=1)
            else:
                ia = nna.kneighbors(Za_k, n_neighbors=1, return_distance=False).squeeze(1)
                ib = nnb.kneighbors(Zb_k, n_neighbors=1, return_distance=False).squeeze(1)
                Za_img = Ca_to_B[ia]; Zb_img = Cb_to_A[ib]

            Za_corr[Ia] = (1 - alphaA) * Za_k + alphaA * Za_img
            Zb_corr[Ib] = (1 - alphaB) * Zb_k + alphaB * Zb_img

            self.nodes_[node] = dict(
                Ca=Ca, Cb=Cb,
                Ca_to_B=Ca_to_B.astype(np.float32),
                Cb_to_A=Cb_to_A.astype(np.float32),
                nna=nna, nnb=nnb
            )

        ZsA = self.pca_.inverse_transform(Za_corr) if self.pca_ is not None else Za_corr
        ZsB = self.pca_.inverse_transform(Zb_corr) if self.pca_ is not None else Zb_corr
        if self.output_scale == 'A':
            Xa_t_corr = self.sa_.inverse_transform(ZsA)
            Xb_t_corr = self.sa_.inverse_transform(ZsB)
        elif self.output_scale == 'B':
            Xa_t_corr = self.sb_.inverse_transform(ZsA)
            Xb_t_corr = self.sb_.inverse_transform(ZsB)
        else:
            Xa_t_corr = self.sa_.inverse_transform(ZsA)
            Xb_t_corr = self.sb_.inverse_transform(ZsB)

        if self.q_grid is None:
            self.q_grid = np.array([1,5,10,20,30,40,50,60,70,80,90,95,99], float)
        p = Xa_t_corr.shape[1]
        alphaAw, alphaBw = self._alphas_warp_for_batches()
        n_nodes = self.mx_ * self.my_
        for node in range(n_nodes):
            Ia = np.where(lab_a == node)[0]; Ib = np.where(lab_b == node)[0]
            if len(Ia) < self.min_node_size or len(Ib) < self.min_node_size:
                continue
            A = Xa_t_corr[Ia]; B = Xb_t_corr[Ib]
            knots_B2A = []; knots_A2B = []
            for j in range(p):
                xk, yk = _quantile_warp_fit(A[:, j], B[:, j], self.q_grid)
                xkA, ykA = _quantile_warp_fit(B[:, j], A[:, j], self.q_grid)
                Bmap = _quantile_warp_apply(B[:, j], xk, yk)
                Amap = _quantile_warp_apply(A[:, j], xkA, ykA)
                Xa_t_corr[Ia, j] = (1 - alphaAw) * A[:, j] + alphaAw * Amap
                Xb_t_corr[Ib, j] = (1 - alphaBw) * B[:, j] + alphaBw * Bmap
                knots_B2A.append((xk.astype(np.float32), yk.astype(np.float32)))
                knots_A2B.append((xkA.astype(np.float32), ykA.astype(np.float32)))
            if node in self.nodes_:
                self.nodes_[node]['knots_B2A'] = knots_B2A
                self.nodes_[node]['knots_A2B'] = knots_A2B

        self.Xa_corr_ = self._destabilize(Xa_t_corr)
        self.Xb_corr_ = self._destabilize(Xb_t_corr)
        return self

    # --- transform ---
    def transform(self, X, y):
        check_is_fitted(self, ["sa_", "sb_", "som_", "label_a_", "label_b_", "nodes_"])
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        if not np.all(np.isin(y, [self.label_a_, self.label_b_])):
            raise ValueError("transform y contains unseen batch labels.")
        X_out = np.empty_like(X, dtype=np.float32)
        alphaA, alphaB = self._alphas_for_batches()
        alphaAw, alphaBw = self._alphas_warp_for_batches()

        for label in [self.label_a_, self.label_b_]:
            mask = (y == label)
            if not np.any(mask): 
                continue
            isA = (label == self.label_a_)

            Xt = self._stabilize(X[mask])
            scaler = self.sa_ if isA else self.sb_
            Zs = scaler.transform(Xt).astype(np.float32)
            Zp = self.pca_.transform(Zs).astype(np.float32) if self.pca_ is not None else Zs

            nodes = []
            for z in Zp:
                i, j = self.som_.winner(z)
                nodes.append(i * self.my_ + j)
            nodes = np.array(nodes, int)

            Zp_corr = Zp.copy()
            for node in np.unique(nodes):
                idx = np.where(nodes == node)[0]
                info = self.nodes_.get(node)
                if info is None or len(idx)==0:
                    continue
                if isA:
                    nn = info['nna']; mapped = info['Ca_to_B']; a = alphaA
                else:
                    nn = info['nnb']; mapped = info['Cb_to_A']; a = alphaB
                d, ii = nn.kneighbors(Zp_corr[idx], return_distance=True)
                w = np.exp(-d / (self.soft_tau + 1e-12)); w /= (w.sum(axis=1, keepdims=True) + 1e-12)
                Z_img = (w[:, :, None] * mapped[ii]).sum(axis=1)
                Zp_corr[idx] = (1 - a) * Zp_corr[idx] + a * Z_img

            Zs_corr = self.pca_.inverse_transform(Zp_corr).astype(np.float32) if self.pca_ is not None else Zp_corr
            if self.output_scale == 'A':
                Xt_corr = self.sa_.inverse_transform(Zs_corr).astype(np.float32)
            elif self.output_scale == 'B':
                Xt_corr = self.sb_.inverse_transform(Zs_corr).astype(np.float32)
            else:
                Xt_corr = (self.sa_ if isA else self.sb_).inverse_transform(Zs_corr).astype(np.float32)

            Xw = Xt_corr.copy()
            for node in np.unique(nodes):
                idx = np.where(nodes == node)[0]
                info = self.nodes_.get(node)
                if info is None or len(idx)==0:
                    continue
                knots = info['knots_A2B'] if isA else info['knots_B2A']
                a_w = alphaAw if isA else alphaBw
                for j, (xk, yk) in enumerate(knots):
                    mapped = _quantile_warp_apply(Xw[idx, j], xk, yk)
                    Xw[idx, j] = (1 - a_w) * Xw[idx, j] + a_w * mapped

            X_out[mask] = self._destabilize(Xw)
        return X_out

    def fit_transform(self, X, y):
        self.fit(X, y)
        return type(self).transform(self, X, y)






