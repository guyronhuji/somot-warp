# SOM-OT Warp (sklearn API, Anchorable)

Exact workflow: arcsinh/log1p → per-batch z-score → pooled PCA → MiniSom(x,y) → per-node KMeans landmarks → entropic OT → latent blend (barycentric/anchored) → invert → per-marker quantile warp blend → inverse.
