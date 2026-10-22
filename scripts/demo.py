import numpy as np
from sklearn.datasets import make_blobs
from somot_warp.sklearn_api import SOMOTWarpSKExact
X, y_true = make_blobs(n_samples=2000, centers=4, n_features=10, random_state=0)
A = X + 0.5
B = X*1.1 - 0.4
Xall = np.vstack([A,B])
y = np.array(['A']*len(A) + ['B']*len(B))
model = SOMOTWarpSKExact(som_grid=(8,8), som_iters=3000, anchor='A', output_scale='A', som_verbose=True)
Xcorr = model.fit_transform(Xall, y)
print('Corrected:', Xcorr.shape)
