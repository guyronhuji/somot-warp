
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed

from .sklearn_api import SOMOTWarpSKExact


def _fit_one_pair(anchor_data, other_data, kwargs):
    X_pair = np.vstack([anchor_data, other_data])
    y_pair = np.array(['__anchor__'] * len(anchor_data)
                       + ['__other__'] * len(other_data))
    model = SOMOTWarpSKExact(anchor='A', output_scale='A', **kwargs)
    model.fit(X_pair, y_pair)
    return model


class MultiBatchSOMOTWarp(BaseEstimator, TransformerMixin):
    """Star-topology multi-batch correction using SOMOTWarpSKExact.

    Corrects N batches toward a single anchor batch by running N-1
    independent pairwise corrections (anchor vs each non-anchor batch).

    Parameters
    ----------
    anchor_label : str or int
        The batch label to use as the reference. Must be present in y.
    n_jobs : int, default=1
        Number of parallel jobs for fitting pairwise models.
        Use -1 for all CPUs.
    **kwargs
        All other keyword arguments are forwarded to each
        SOMOTWarpSKExact instance. Note that ``anchor``, ``output_scale``,
        ``alpha_ot``, and ``alpha_warp`` are overridden internally.
    """

    def __init__(self, anchor_label, n_jobs=1, **kwargs):
        self.anchor_label = anchor_label
        self.n_jobs = n_jobs

        # warn and strip params that are forced by the star topology
        for key in ('anchor', 'output_scale'):
            if key in kwargs:
                warnings.warn(
                    f"'{key}' is ignored in MultiBatchSOMOTWarp "
                    f"(forced to anchor='A', output_scale='A')."
                )
                kwargs.pop(key)
        self.kwargs = kwargs

    def fit(self, X, y):
        """Fit pairwise models for each non-anchor batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Batch labels. Must contain ``anchor_label``.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        labels = np.unique(y)
        if self.anchor_label not in labels:
            raise ValueError(
                f"anchor_label '{self.anchor_label}' not found in y. "
                f"Available labels: {labels.tolist()}"
            )
        if labels.size < 2:
            raise ValueError(
                f"Need at least 2 batches; got {labels.size}: {labels.tolist()}"
            )

        self.labels_ = labels
        anchor_mask = y == self.anchor_label
        anchor_data = X[anchor_mask]
        self.n_features_in_ = X.shape[1]

        other_labels = [l for l in labels if l != self.anchor_label]

        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_pair)(
                anchor_data, X[y == label], dict(self.kwargs)
            )
            for label in other_labels
        )

        self.models_ = dict(zip(other_labels, fitted))
        return self

    def transform(self, X, y):
        """Apply learned corrections to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Batch labels. Must be anchor or a label seen during fit.

        Returns
        -------
        X_corr : ndarray of shape (n_samples, n_features)
        """
        check_is_fitted(self, ["models_", "labels_"])
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        unseen = set(np.unique(y)) - set(self.labels_)
        if unseen:
            raise ValueError(f"transform y contains unseen batch labels: {unseen}")

        X_out = np.empty_like(X, dtype=np.float32)

        # anchor rows pass through unchanged
        anchor_mask = y == self.anchor_label
        if np.any(anchor_mask):
            X_out[anchor_mask] = X[anchor_mask]

        # each non-anchor batch: use its pairwise model
        for label, model in self.models_.items():
            mask = y == label
            if not np.any(mask):
                continue
            Xb = X[mask]
            y_pair = np.array(['__other__'] * len(Xb))
            X_out[mask] = model.transform(Xb, y_pair)

        return X_out

    def fit_transform(self, X, y):
        """Fit and return corrected data.

        Uses the already-corrected arrays stored during fit
        (``Xa_corr_`` for anchor, ``Xb_corr_`` for non-anchor)
        so that the result is identical to the training correction.
        """
        self.fit(X, y)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        X_out = np.empty_like(X, dtype=np.float32)

        # anchor rows: take corrected anchor from any pairwise model
        anchor_mask = y == self.anchor_label
        if np.any(anchor_mask):
            any_model = next(iter(self.models_.values()))
            X_out[anchor_mask] = any_model.Xa_corr_

        # non-anchor rows: take corrected B from the corresponding model
        for label, model in self.models_.items():
            mask = y == label
            if np.any(mask):
                X_out[mask] = model.Xb_corr_

        return X_out
